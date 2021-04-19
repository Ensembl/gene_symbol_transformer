#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Evaluate a trained network or get statistics for existing symbol assignments.

Pass the --assignments_csv and --ensembldb_database arguments to compare
the assignments in the `assignments_csv` CSV file with the ones in the `ensembldb_database`
Ensembl database,
or
pass the --checkpoint argument to evaluate a trained network by assigning symbols
to the canonical translations of protein sequences of annotations in the latest
Ensembl release and comparing them to the existing symbol assignments.

A trained network can be evaluated by assigning gene symbols to the assembly annotations on the main Ensembl website and comparing them with the existing gene symbols.
"""


# standard library imports
import argparse
import csv
import gzip
import pathlib
import sys

# third party imports
import ensembl_rest
import pandas as pd
import pymysql
import requests

from icecream import ic
from loguru import logger

# project imports
from fully_connected_pipeline import FullyConnectedNetwork
from utils import (
    PrettySimpleNamespace,
    data_directory,
    load_checkpoint,
    read_fasta_in_chunks,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

sequences_directory = data_directory / "protein_sequences"
sequences_directory.mkdir(exist_ok=True)


get_xref_symbols_for_canonical_gene_transcripts = """
-- Xref symbols for canonical translations
SELECT
  translation.stable_id AS translation_stable_id,
  xref.display_label AS Xref_symbol
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
WHERE gene.biotype = 'protein_coding';
"""

get_entrezgene_symbols = """
-- EntrezGene symbols for translations with no Xref symbols using a subquery
SELECT
  translation.stable_id AS translation_stable_id,
  xref.display_label AS EntrezGene_symbol
FROM gene
INNER JOIN object_xref
  ON gene.gene_id = object_xref.ensembl_id
INNER JOIN xref
  ON object_xref.xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
WHERE gene.gene_id IN (
  -- ensembl_id of canonical translations without Xref symbols
  SELECT
    gene.gene_id
  FROM gene
  INNER JOIN transcript
    ON gene.canonical_transcript_id = transcript.transcript_id
  INNER JOIN translation
    ON transcript.canonical_translation_id = translation.translation_id
  WHERE gene.biotype = 'protein_coding'
  AND gene.display_xref_id IS NULL
)
AND gene.biotype = 'protein_coding'
AND object_xref.ensembl_object_type = 'Gene'
AND external_db.db_name = 'EntrezGene';
"""

get_uniprot_gn_symbols = """
-- Uniprot_gn symbols for translations with no Xref and no EntrezGene symbols
SELECT
  translation.stable_id AS translation_stable_id,
  xref.display_label AS Uniprot_gn_symbol
FROM gene
INNER JOIN object_xref
  ON gene.gene_id = object_xref.ensembl_id
INNER JOIN xref
  ON object_xref.xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
WHERE gene.gene_id IN (
  SELECT
    gene.gene_id
  FROM gene
  INNER JOIN transcript
    ON gene.canonical_transcript_id = transcript.transcript_id
  INNER JOIN translation
    ON transcript.canonical_translation_id = translation.translation_id
  WHERE gene.biotype = 'protein_coding'
  AND gene.display_xref_id IS NULL
  AND gene.gene_id NOT IN (
    -- ensembl_id of canonical translations without Xref or EntrezGene symbols
    SELECT
      gene.gene_id
    FROM gene
    INNER JOIN object_xref
      ON gene.gene_id = object_xref.ensembl_id
    INNER JOIN xref
      ON object_xref.xref_id = xref.xref_id
    INNER JOIN external_db
      ON xref.external_db_id = external_db.external_db_id
    INNER JOIN transcript
      ON gene.canonical_transcript_id = transcript.transcript_id
    INNER JOIN translation
      ON transcript.canonical_translation_id = translation.translation_id
    WHERE gene.gene_id IN (
      -- ensembl_id of canonical translations without Xref symbols
      SELECT
        gene.gene_id
      FROM gene
      INNER JOIN transcript
        ON gene.canonical_transcript_id = transcript.transcript_id
      INNER JOIN translation
        ON transcript.canonical_translation_id = translation.translation_id
      WHERE gene.biotype = 'protein_coding'
      AND gene.display_xref_id IS NULL
    )
    AND gene.biotype = 'protein_coding'
    AND object_xref.ensembl_object_type = 'Gene'
    AND external_db.db_name = 'EntrezGene'
  )
)
AND gene.biotype = 'protein_coding'
AND object_xref.ensembl_object_type = 'Gene'
AND external_db.db_name = 'Uniprot_gn';
"""


def get_genomes_metadata():
    """
    Get metadata for all genomes in the latest Ensembl release.

    The metadata are loaded from the `species_EnsemblVertebrates.txt` file of
    the latest Ensembl release.

    It would have been more elegant to get the genome metadata from the Ensembl
    REST API `/info/species` endpoint but that lacks the core database name.
    https://rest.ensembl.org/documentation/info/species

    The metadata REST API could also be used when it's been updated.
    """
    # get the version number of the latest Ensembl release
    ensembl_release = ensembl_rest.software()["release"]

    # download the `species_EnsemblVertebrates.txt` file
    species_data_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/species_EnsemblVertebrates.txt"
    species_data_path = data_directory / "species_EnsemblVertebrates.txt"
    if not species_data_path.exists():
        response = requests.get(species_data_url)
        with open(species_data_path, "wb+") as f:
            f.write(response.content)
        logger.info(f"downloaded {species_data_path}")

    genomes_df = pd.read_csv(species_data_path, delimiter="\t", index_col=False)
    genomes_df = genomes_df.rename(columns={"#name": "name"})
    genomes = [
        PrettySimpleNamespace(**genome_row._asdict())
        for genome_row in genomes_df.itertuples()
    ]

    return genomes


def fix_assembly(assembly):
    """
    Fixes for cases that the FASTA pep file naming doesn't mirror the assembly name.
    """
    # fix for a few assembly names
    # http://ftp.ensembl.org/pub/release-103/fasta/erinaceus_europaeus/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/homo_sapiens/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/loxodonta_africana/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/poecilia_formosa/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/sorex_araneus/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/tetraodon_nigroviridis/pep/
    # http://ftp.ensembl.org/pub/release-103/fasta/tupaia_belangeri/pep/
    names_map = {
        "eriEur1": "HEDGEHOG",
        "GRCh38.p13": "GRCh38",
        "Loxafr3.0": "loxAfr3",
        "Poecilia_formosa-5.1.2": "PoeFor_5.1.2",
        "sorAra1": "COMMON_SHREW1",
        "TETRAODON 8.0": "TETRAODON8",
        "tupBel1": "TREESHREW",
    }

    if assembly in names_map:
        return names_map[assembly]

    # remove spaces in the assembly name
    return assembly.replace(" ", "")


def evaluate_network(checkpoint_path):
    """
    Evaluate a trained network by downloading FASTA files with protein sequences
    for the annotated genome assemblies in the latest Ensembl release, assigning
    gene symbols to the sequences, and comparing the assigned symbols to the current
    public Xref symbol assignments.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]
    training_session = checkpoint["training_session"]

    ensembl_release = ensembl_rest.software()["release"]

    base_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/fasta/"

    genomes = get_genomes_metadata()
    for genome in genomes:
        # download archived protein sequences FASTA file
        archived_fasta_filename = "{}.{}.pep.all.fa.gz".format(
            genome.species.capitalize(),
            fix_assembly(genome.assembly),
        )

        archived_fasta_url = f"{base_url}{genome.species}/pep/{archived_fasta_filename}"

        archived_fasta_path = sequences_directory / archived_fasta_filename
        if not archived_fasta_path.exists():
            response = requests.get(archived_fasta_url)
            with open(archived_fasta_path, "wb+") as f:
                f.write(response.content)
            logger.info(f"downloaded {archived_fasta_filename}")

        # extract archived protein sequences FASTA file
        fasta_path = archived_fasta_path.with_suffix("")
        if not fasta_path.exists():
            with gzip.open(archived_fasta_path, "rb") as f:
                file_content = f.read()
            with open(fasta_path, "wb+") as f:
                f.write(file_content)
            logger.info(f"extracted {fasta_path}")

        # assign symbols
        assignments_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {fasta_path}")
            assign_symbols(network, checkpoint_path, fasta_path)

        comparisons_csv_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
        )
        if not comparisons_csv_path.exists():
            compare_with_database(
                assignments_csv_path,
                genome.core_db,
                genome.species,
            )


def assign_symbols(network, checkpoint_path, sequences_fasta):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    fasta_path = pathlib.Path(sequences_fasta)
    assignments_csv_path = pathlib.Path(
        f"{checkpoint_path.parent}/{fasta_path.stem}_symbols.csv"
    )

    # read the FASTA file in chunks and assign symbols
    with open(assignments_csv_path, "w+") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol"]
        csv_writer = csv.writer(csv_file, delimiter="\t")
        csv_writer.writerow(field_names)

        for fasta_entries in read_fasta_in_chunks(fasta_path):
            if fasta_entries[-1] is None:
                fasta_entries = [
                    fasta_entry
                    for fasta_entry in fasta_entries
                    if fasta_entry is not None
                ]

            stable_ids = [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]

            assignments = network.predict(sequences)

            # save assignments to the CSV file
            csv_writer.writerows(zip(stable_ids, assignments))
    logger.info(f"symbol assignments saved at {assignments_csv_path}")


def compare_with_database(
    assignments_csv,
    ensembldb_database,
    scientific_name=None,
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Compare classifier assignments with the gene symbols in the genome database on
    the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    host = "ensembldb.ensembl.org"
    user = "anonymous"
    connection = pymysql.connect(
        host=host,
        user=user,
        database=ensembldb_database,
        # cursorclass=pymysql.cursors.DictCursor,
    )

    sql_queries = [
        get_xref_symbols_for_canonical_gene_transcripts,
    ]

    if EntrezGene:
        sql_queries.append(get_entrezgene_symbols)

    if Uniprot_gn:
        sql_queries.append(get_uniprot_gn_symbols)

    db_responses_dict = {}

    with connection:
        for sql_query in sql_queries:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                db_response = cursor.fetchall()

            db_response_dict = dict(db_response)
            assert set(db_responses_dict.keys()).isdisjoint(db_response_dict.keys())
            db_responses_dict.update(db_response_dict)

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]

            translation_stable_id = csv_stable_id[:-2]

            if translation_stable_id in db_responses_dict:
                xref_symbol = db_responses_dict[translation_stable_id]
                comparisons.append((csv_stable_id, classifier_symbol, xref_symbol))

    dataframe_columns = [
        "csv_stable_id",
        "classifier_symbol",
        "xref_symbol",
    ]
    comparisons_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    comparisons_df.to_csv(comparisons_csv_path, sep="\t", index=False)
    logger.info(f"comparisons CSV saved at {comparisons_csv_path}")

    num_assignments = len(comparisons_df)

    comparisons_df["classifier_symbol_lowercase"] = comparisons_df[
        "classifier_symbol"
    ].str.lower()
    comparisons_df["xref_symbol_lowercase"] = comparisons_df["xref_symbol"].str.lower()

    num_equal_assignments = (
        comparisons_df["classifier_symbol_lowercase"]
        .eq(comparisons_df["xref_symbol_lowercase"])
        .sum()
    )

    matching_percentage = (num_equal_assignments / num_assignments) * 100
    if scientific_name is not None:
        message = f"{scientific_name}: "
    else:
        message = ""
    message += f"{num_equal_assignments} matching out of {num_assignments} assignments ({matching_percentage:.2f}%)"
    logger.info(message)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--assignments_csv",
        help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--ensembldb_database",
        help="genome database name on the public Ensembl MySQL server",
    )
    argument_parser.add_argument("--checkpoint", help="training session checkpoint path")

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)

    if args.assignments_csv and args.ensembldb_database:
        assignments_csv_path = pathlib.Path(args.assignments_csv)
        log_file_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
        )

        logger.add(log_file_path, format=LOGURU_FORMAT)
        compare_with_database(args.assignments_csv, args.ensembldb_database)
    elif args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        log_file_path = pathlib.Path(
            f"{checkpoint_path.parent}/{checkpoint_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=LOGURU_FORMAT)

        evaluate_network(checkpoint_path)
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
