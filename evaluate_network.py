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
pass the --checkpoint and --species_data arguments to evaluate a trained network
from scratch, by downloading FASTA files with protein sequences for a list of annotated
genome assemblies, assigning gene symbols to the sequences, and comparing the assigned
symbols to the current public Xref symbol assignments.
"""


# standard library imports
import argparse
import csv
import gzip
import pathlib
import sys

# third party imports
import pandas as pd
import pymysql
import requests
import yaml

from loguru import logger

# project imports
from fully_connected_pipeline import FullyConnectedNetwork
from pipeline_abstractions import (
    PrettySimpleNamespace,
    data_directory,
    load_checkpoint,
    read_fasta_in_chunks,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

sequences_directory = data_directory / "protein_sequences"
sequences_directory.mkdir(exist_ok=True)


get_xref_symbols_for_canonical_gene_transcripts = """
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


def evaluate_network(checkpoint_path, species_data_path):
    """
    Evaluate a trained network by downloading FASTA files with protein sequences
    for a list of annotated genome assemblies, assigning gene symbols to the sequences,
    and comparing the assigned symbols to the current public Xref symbol assignments.
    """
    with open(species_data_path) as f:
        species_data_list = yaml.safe_load(f)

    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]
    training_session = checkpoint["training_session"]

    for species_data_dict in species_data_list:
        species_data = PrettySimpleNamespace(**species_data_dict)

        # download archived protein sequences FASTA file
        archived_fasta_filename = species_data.protein_sequences.split("/")[-1]
        archived_fasta_path = sequences_directory / archived_fasta_filename
        if not archived_fasta_path.exists():
            response = requests.get(species_data.protein_sequences)
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
            f"{checkpoint_path.parent}/{checkpoint_path.stem}/{fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {fasta_path}")
            assign_symbols(network, checkpoint_path, fasta_path)

        compare_with_database(
            assignments_csv_path,
            species_data.ensembldb_database,
            species_data.scientific_name,
        )


def assign_symbols(network, checkpoint_path, sequences_fasta):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    fasta_path = pathlib.Path(sequences_fasta)
    assignments_csv_path = pathlib.Path(
        f"{checkpoint_path.parent}/{checkpoint_path.stem}/{fasta_path.stem}_symbols.csv"
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


def compare_with_database(assignments_csv, ensembldb_database, scientific_name=None):
    """
    Compare classifier assignments with the gene symbols in the species database on
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

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(get_xref_symbols_for_canonical_gene_transcripts)
            db_response = cursor.fetchall()

    db_response_dict = dict(db_response)

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]

            translation_stable_id = csv_stable_id[:-2]

            if translation_stable_id in db_response_dict:
                xref_symbol = db_response_dict[translation_stable_id]
                comparisons.append((csv_stable_id, classifier_symbol, xref_symbol))

    dataframe_columns = [
        "csv_stable_id",
        "classifier_symbol",
        "xref_symbol",
    ]
    comparisons_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare_database.csv"
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
        "--assignments_csv", help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--ensembldb_database",
        help="species database name on the public Ensembl MySQL server",
    )
    argument_parser.add_argument("--checkpoint", help="training session checkpoint path")
    argument_parser.add_argument("--species_data", help="species data YAML file path")

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
    elif args.checkpoint and args.species_data:
        checkpoint_path = pathlib.Path(args.checkpoint)
        species_data_path = pathlib.Path(args.species_data)
        log_file_path = pathlib.Path(
            f"{species_data_path.parent}/{checkpoint_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=LOGURU_FORMAT)

        evaluate_network(checkpoint_path, species_data_path)
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
        sys.exit()
