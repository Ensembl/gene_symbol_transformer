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
Generate training dataset of protein sequences for canonical translations from
assemblies in the latest Ensembl release and useful metadata, generate statistics,
and create auxiliary development dataset files containing a subset of samples from
the full dataset for faster prototyping.
"""


# standard library imports
import argparse
import gzip
import pathlib
import pickle
import sys
import time

# third party imports
import ensembl_rest
import pandas as pd
import pymysql
import requests

from icecream import ic
from loguru import logger

# project imports
from utils import (
    PrettySimpleNamespace,
    data_directory,
    fasta_to_dict,
    load_dataset,
    dev_datasets_symbol_frequency,
    sizeof_fmt,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

sequences_directory = data_directory / "protein_sequences"
sequences_directory.mkdir(parents=True, exist_ok=True)

get_xref_symbols_for_canonical_gene_transcripts = """
-- Xref symbols for canonical translations
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'Xref_symbol',
  external_db.db_display_name AS 'external_db.db_display_name'
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
WHERE gene.biotype = 'protein_coding';
"""

get_entrezgene_symbols = """
-- EntrezGene symbols for translations with no Xref symbols
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'EntrezGene_symbol',
  external_db.db_display_name AS 'external_db.db_display_name'
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
  -- ensembl_id of canonical translations with no Xref symbols
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
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'Uniprot_gn_symbol',
  external_db.db_display_name AS 'external_db.db_display_name'
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
    -- ensembl_id of canonical translations with no Xref and no EntrezGene symbols
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
      -- ensembl_id of canonical translations with no Xref symbols
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


def get_ensembl_release():
    """
    retrieve the version number of the latest Ensembl release
    """
    # https://rest.ensembl.org/documentation/info/data
    ensembl_release = max(ensembl_rest.data()["releases"])
    return ensembl_release


def dataframe_to_fasta(df, fasta_path):
    """
    Save a dataframe containing entries of sequences and metadata to a FASTA file.
    """
    with open(fasta_path, "w+") as fasta_file:
        for index, values in df.iterrows():
            row_dict = values.to_dict()
            description_text = "\t".join(
                f"{key}:{value}"
                for key, value in row_dict.items()
                if key not in {"Index", "sequence"}
            )

            description = ">" + description_text

            sequence = row_dict["sequence"]
            fasta_entry = f"{description}\n{sequence}\n"
            fasta_file.write(fasta_entry)


def save_dev_datasets(num_samples=100):
    """
    Generate and save subsets of the full dataset for faster loading during development,
    the datasets as FASTA files, and FASTA files with a small number of sample sequences
    for quick reference.
    """
    dataset = load_dataset()

    symbol_counts = dataset["symbol"].value_counts()

    for num_symbols, max_frequency in dev_datasets_symbol_frequency.items():
        # verify that max_frequency is the cutoff limit for the selected symbols
        assert all(
            symbol_counts[:num_symbols] == symbol_counts[symbol_counts >= max_frequency]
        )
        assert symbol_counts[num_symbols] < max_frequency

        dev_dataset = dataset[dataset["symbol"].isin(symbol_counts[:num_symbols].index)]

        # save dataframe to a pickle file
        pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
        dev_dataset.to_pickle(pickle_path)
        logger.info(
            f"{num_symbols} most frequent symbols dev dataset saved at {pickle_path}"
        )

        # save sequences to a FASTA file
        fasta_path = data_directory / f"{num_symbols}_symbols.fasta"

        dataframe_to_fasta(dev_dataset, fasta_path)
        logger.info(
            f"{num_symbols} most frequent symbols dev dataset FASTA file saved at {fasta_path}"
        )

        # pick random sample sequences
        samples = dev_dataset.sample(num_samples)
        samples = samples.sort_index()

        # save sample sequences to a FASTA file
        fasta_path = data_directory / f"{num_symbols}_symbols-{num_samples}_samples.fasta"
        dataframe_to_fasta(samples, fasta_path)
        logger.info(
            f"{num_symbols} most frequent symbols {num_samples} samples FASTA file saved at {fasta_path}"
        )


def download_file(file_url, file_path):
    """
    Download a file from the Ensembl HTTP server.

    Requesting the file includes a retry loop, because sometimes an erroneous
    "404 Not Found" response is issued by the server for actually existing files,
    which subsequently are correctly downloaded on a following request.

    Args:
        file_url (str): URL of the file to be downloaded
        file_path (Path or str): path to save the downloaded file
    """
    while not file_path.exists():
        response = requests.get(file_url)
        if response.ok:
            with open(file_path, "wb+") as f:
                f.write(response.content)
        else:
            # delay retry
            time.sleep(5)


def get_assemblies_metadata():
    """
    Get metadata for all genome assemblies in the latest Ensembl release.

    The metadata are loaded from the `species_EnsemblVertebrates.txt` file of
    the latest Ensembl release.

    It would have been more elegant to get the metadata from the Ensembl REST API
    `/info/species` endpoint but that lacks the core database name.
    https://rest.ensembl.org/documentation/info/species

    The metadata REST API could also be used when it's been updated.
    """
    ensembl_release = get_ensembl_release()

    # download the `species_EnsemblVertebrates.txt` file
    species_data_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/species_EnsemblVertebrates.txt"
    species_data_path = data_directory / "species_EnsemblVertebrates.txt"
    if not species_data_path.exists():
        download_file(species_data_url, species_data_path)
        logger.info(f"downloaded {species_data_path}")
    else:
        logger.info(f"{species_data_path} in place")

    assemblies_df = pd.read_csv(species_data_path, delimiter="\t", index_col=False)
    assemblies_df = assemblies_df.rename(columns={"#name": "name"})
    assemblies = [
        PrettySimpleNamespace(**genome_row._asdict())
        for genome_row in assemblies_df.itertuples()
    ]

    return assemblies


def fix_assembly(assembly):
    """
    Fixes for cases that the FASTA pep file naming doesn't mirror the assembly name.
    """
    # fix for a few assembly names
    # http://ftp.ensembl.org/pub/release-104/fasta/erinaceus_europaeus/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/loxodonta_africana/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/poecilia_formosa/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/sorex_araneus/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/tetraodon_nigroviridis/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/tupaia_belangeri/pep/
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


def download_protein_sequences_fasta(assembly, ensembl_release):
    """
    Download and extract the archived protein sequences FASTA file for the species
    described in the assembly object.
    """
    base_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/fasta/"

    # download archived protein sequences FASTA file
    archived_fasta_filename = "{}.{}.pep.all.fa.gz".format(
        assembly.species.capitalize(),
        fix_assembly(assembly.assembly),
    )
    archived_fasta_url = f"{base_url}{assembly.species}/pep/{archived_fasta_filename}"
    archived_fasta_path = sequences_directory / archived_fasta_filename
    if not archived_fasta_path.exists():
        download_file(archived_fasta_url, archived_fasta_path)
        logger.info(f"downloaded {archived_fasta_filename}")
    else:
        logger.info(f"{archived_fasta_filename} in place")

    # extract archived protein sequences FASTA file
    fasta_path = archived_fasta_path.with_suffix("")
    if not fasta_path.exists():
        with gzip.open(archived_fasta_path, "rb") as f:
            file_content = f.read()
        with open(fasta_path, "wb+") as f:
            f.write(file_content)
        logger.info(f"extracted {fasta_path}")

    return fasta_path


def get_canonical_translations(ensembldb_database, EntrezGene=False, Uniprot_gn=False):
    """
    Get canonical translation sequences from the genome assembly with
    the ensembldb_database core database.
    """
    host = "ensembldb.ensembl.org"
    user = "anonymous"
    connection = pymysql.connect(
        host=host,
        user=user,
        database=ensembldb_database,
        cursorclass=pymysql.cursors.DictCursor,
    )

    sql_queries = [
        get_xref_symbols_for_canonical_gene_transcripts,
    ]

    if EntrezGene:
        sql_queries.append(get_entrezgene_symbols)

    if Uniprot_gn:
        sql_queries.append(get_uniprot_gn_symbols)

    columns = [
        "gene.stable_id",
        "gene.version",
        "translation.stable_id",
        "translation.version",
        "Xref_symbol",
        "external_db.db_display_name",
        "sequence",
    ]
    canonical_translations_df = pd.DataFrame(columns=columns)

    canonical_translations_list = []
    with connection:
        for sql_query in sql_queries:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                response = cursor.fetchall()
            canonical_translations_list.extend(response)

    canonical_translations_df = pd.concat(
        [canonical_translations_df, pd.DataFrame(canonical_translations_list)],
        ignore_index=True,
    )

    return canonical_translations_df


def get_sequence_from_assembly_fasta_dict(df_row, assembly_fasta_dict):
    """
    Retrieve the sequence in the assembly FASTA dictionary that the translations
    dataframe row corresponds to.
    """
    if df_row["translation.version"] is None:
        translation_stable_id_version = df_row["translation.stable_id"]
    else:
        translation_stable_id_version = "{}.{}".format(
            df_row["translation.stable_id"], df_row["translation.version"]
        )
    sequence = assembly_fasta_dict[translation_stable_id_version]["sequence"]

    return sequence


def generate_dataset():
    """
    Download canonical translations of protein coding genes from all genome assemblies
    in the latest Ensembl release.
    """
    ensembl_release = get_ensembl_release()
    logger.info(f"Ensembl release {ensembl_release}")

    assemblies = get_assemblies_metadata()
    logger.info(f"downloading protein sequences FASTA files")
    for assembly in assemblies:
        fasta_path = download_protein_sequences_fasta(assembly, ensembl_release)
        assembly.fasta_path = str(fasta_path.resolve())
    logger.info("protein sequences FASTA files in place")

    logger.info(f"retrieving assemblies metadata from the Ensembl REST API")
    metadata = []
    for assembly in assemblies:
        # skip assembly if assembly_accession is missing
        # (the value is converted to a `nan` float)
        if type(assembly.assembly_accession) is float:
            continue

        # delay between REST API calls
        time.sleep(0.2)

        # retrieve additional information for the assembly from the REST API
        # https://rest.ensembl.org/documentation/info/info_genomes_assembly
        response = ensembl_rest.info_genomes_assembly(assembly.assembly_accession)
        rest_assembly = PrettySimpleNamespace(**response)

        assembly_metadata = PrettySimpleNamespace()

        assembly_metadata.assembly_accession = assembly.assembly_accession
        assembly_metadata.scientific_name = rest_assembly.scientific_name
        assembly_metadata.common_name = rest_assembly.display_name
        assembly_metadata.taxonomy_id = assembly.taxonomy_id
        assembly_metadata.core_db = assembly.core_db
        assembly_metadata.sequences_fasta_path = assembly.fasta_path

        metadata.append(assembly_metadata)

        species = assembly.species.replace("_", " ").capitalize()
        logger.info(
            f"retrieved metadata for {assembly.name}, {species}, {assembly.assembly_accession}"
        )

    logger.info(
        f"retrieving canonical translation stable IDs and metadata from the Ensembl MySQL server"
    )
    canonical_translations_list = []
    for assembly in metadata:
        # delay between SQL queries
        time.sleep(0.1)

        assembly_translations = get_canonical_translations(assembly.core_db)
        assembly_fasta_dict = fasta_to_dict(assembly.sequences_fasta_path)

        assembly_translations["sequence"] = assembly_translations.apply(
            lambda x: get_sequence_from_assembly_fasta_dict(x, assembly_fasta_dict),
            axis=1,
        )

        assembly_translations["assembly_accession"] = assembly.assembly_accession
        assembly_translations["scientific_name"] = assembly.scientific_name
        assembly_translations["common_name"] = assembly.common_name
        assembly_translations["taxonomy_id"] = assembly.taxonomy_id
        assembly_translations["core_db"] = assembly.core_db

        num_assembly_translations = len(assembly_translations)
        logger.info(
            f"retrieved {num_assembly_translations} canonical translations for {assembly.common_name}, {assembly.scientific_name}, {assembly.assembly_accession}"
        )

        canonical_translations_list.append(assembly_translations)

    canonical_translations = pd.concat(canonical_translations_list, ignore_index=True)

    dataset = dataset_cleanup(canonical_translations)

    # save dataset as a pickle file
    dataset_path = data_directory / "dataset.pickle"
    dataset.to_pickle(dataset_path)
    num_translations = len(dataset)
    logger.info(f"{num_translations} canonical translations saved at {dataset_path}")


def dataset_cleanup(dataset):
    """
    Merge symbol capitalization variants to the most frequent version.
    """
    logger.info("merging symbol capitalization variants...")
    # create temporary column with Xref symbols in lowercase
    dataset["Xref_symbol_lowercase"] = dataset["Xref_symbol"].str.lower()

    # create dictionary to map from the lowercase to the most frequent capitalization
    symbol_capitalization_mapping = (
        dataset.groupby(["Xref_symbol_lowercase"])["Xref_symbol"]
        .agg(lambda x: pd.Series.mode(x)[0])
        .to_dict()
    )

    # save the most frequent capitalization for each Xref symbol
    dataset["symbol"] = dataset["Xref_symbol_lowercase"].map(
        symbol_capitalization_mapping
    )

    # delete temporary lowercase symbols column
    dataset = dataset.drop(columns=["Xref_symbol_lowercase"])

    # reorder dataframe columns
    columns = [
        "gene.stable_id",
        "gene.version",
        "translation.stable_id",
        "translation.version",
        "Xref_symbol",
        "external_db.db_display_name",
        "symbol",
        "sequence",
        "assembly_accession",
        "scientific_name",
        "common_name",
        "taxonomy_id",
        "core_db",
    ]
    dataset = dataset[columns]

    num_original = dataset["Xref_symbol"].nunique()
    num_merged = dataset["symbol"].nunique()
    logger.info(
        f"{num_original} original symbol capitalization variants merged to {num_merged}"
    )

    return dataset


def generate_statistics():
    """
    Generate and log dataset statistics.
    """
    dataset = load_dataset()

    dataset_object_size = sys.getsizeof(dataset)
    logger.info("dataset object usage: {}".format(sizeof_fmt(dataset_object_size)))

    num_canonical_translations = len(dataset)
    logger.info(f"dataset contains {num_canonical_translations:,} canonical translations")

    # calculate unique symbols occurrence frequency
    symbol_counts = dataset["symbol"].value_counts()
    logger.info(symbol_counts)

    symbol_counts_mean = symbol_counts.mean()
    symbol_counts_median = symbol_counts.median()
    symbol_counts_standard_deviation = symbol_counts.std()
    logger.info(
        f"symbol counts mean: {symbol_counts_mean:.2f}, median: {symbol_counts_median:.2f}, standard deviation: {symbol_counts_standard_deviation:.2f}"
    )

    sequence_length_mean = dataset["sequence"].str.len().mean()
    sequence_length_median = dataset["sequence"].str.len().median()
    sequence_length_standard_deviation = dataset["sequence"].str.len().std()
    logger.info(
        f"sequence length mean: {sequence_length_mean:.2f}, median: {sequence_length_median:.2f}, standard deviation: {sequence_length_standard_deviation:.2f}"
    )


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--generate_dataset",
        action="store_true",
        help="generate dataset from genome assemblies in the latest Ensembl release",
    )
    argument_parser.add_argument(
        "--generate_statistics",
        action="store_true",
        help="generate and log dataset statistics",
    )
    argument_parser.add_argument(
        "--save_dev_datasets",
        action="store_true",
        help="save subsets of the full dataset for development",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)
    log_file_path = data_directory / "dataset_generation.log"
    logger.add(log_file_path, format=LOGURU_FORMAT)

    if args.generate_dataset:
        generate_dataset()
    elif args.generate_statistics:
        generate_statistics()
    elif args.save_dev_datasets:
        save_dev_datasets()
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
