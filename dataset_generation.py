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
Generate training dataset.

Download protein sequences for canonical translations from the assemblies in
the latest Ensembl release, create a Pandas dataframe of them including useful metadata,
and save it to a pickle file.
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
    sizeof_fmt,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

sequences_directory = data_directory / "protein_sequences"
sequences_directory.mkdir(exist_ok=True)

get_xref_symbols_for_canonical_gene_transcripts = """
-- Xref symbols for canonical translations
SELECT
  gene.stable_id AS gene_stable_id,
  gene.version AS gene_version,
  translation.stable_id AS translation_stable_id,
  translation.version AS translation_version,
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
-- EntrezGene symbols for translations with no Xref symbols
SELECT
  gene.stable_id AS gene_stable_id,
  gene.version AS gene_version,
  translation.stable_id AS translation_stable_id,
  translation.version AS translation_version,
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
  gene.stable_id AS gene_stable_id,
  gene.version AS gene_version,
  translation.stable_id AS translation_stable_id,
  translation.version AS translation_version,
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
    dataset["symbol"] = dataset["Xref_symbol_lowercase"].map(symbol_capitalization_mapping)

    # delete temporary lowercase symbols column
    dataset = dataset.drop(columns=["Xref_symbol_lowercase"])

    num_original = dataset["Xref_symbol"].nunique()
    num_merged = dataset["symbol"].nunique()
    logger.info(f"{num_original} original symbol capitalization variants merged to {num_merged}")

    return dataset


def save_all_datasets():
    """
    Save the examples for each num_symbols to a pickled dataframe and a FASTA file.
    """
    num_symbols_max_frequencies = [
        [3, 335],
        [101, 297],
        [1013, 252],
        [10059, 165],
        [20147, 70],
        [25028, 23],
        [26007, 17],
        [27137, 13],
        [28197, 10],
        [29041, 8],
        [30591, 5],
    ]

    data = load_dataset()

    for (num_symbols, max_frequency) in num_symbols_max_frequencies:
        print(f"saving {num_symbols} symbols dataset")
        save_dataset(data, num_symbols, max_frequency)


def save_dataset(data, num_symbols, max_frequency):
    """
    Save a training and testin
    """
    symbol_counts = data["symbol"].value_counts()

    # verify that max_frequency is the cutoff limit for the selected symbols
    if max_frequency is not None:
        assert all(
            symbol_counts[:num_symbols] == symbol_counts[symbol_counts >= max_frequency]
        )
        assert symbol_counts[num_symbols] < max_frequency

    most_frequent_n = data[data["symbol"].isin(symbol_counts[:num_symbols].index)]

    # save dataframe to a pickle file
    pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
    most_frequent_n.to_pickle(pickle_path)
    print(
        f"pickle file of the most {num_symbols} frequent symbol sequences saved at {pickle_path}"
    )

    # save sequences to a FASTA file
    fasta_path = data_directory / f"{num_symbols}_symbols.fasta"
    with open(fasta_path, "w+") as fasta_file:
        for entry in most_frequent_n.itertuples():
            entry_dict = entry._asdict()

            stable_id = entry_dict["stable_id"]
            symbol = entry_dict["symbol"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id};{symbol}\n{sequence}\n")

    print(
        f"FASTA file of the most {num_symbols} frequent symbol sequences saved at {fasta_path}"
    )


def save_all_sample_fasta_files():
    num_samples = 1000

    num_symbols_list = [
        3,
        101,
        1013,
        10059,
        20147,
        25028,
        26007,
        27137,
        28197,
        29041,
        30591,
    ]

    for num_symbols in num_symbols_list:
        print(f"saving {num_symbols} sample fasta file")
        save_sample_fasta(num_samples, num_symbols)


def save_sample_fasta(num_samples, num_symbols):
    data_pickle_path = data_directory / f"{num_symbols}_symbols.pickle"

    dataset = pd.read_pickle(data_pickle_path)

    # get num_samples random samples
    data = dataset.sample(num_samples)

    # only the sequences and the symbols are needed as features and labels
    data = data[["stable_id", "symbol", "sequence"]]

    # save sequences to a FASTA file
    fasta_path = data_directory / f"{num_symbols}_symbols-{num_samples}_samples.fasta"
    with open(fasta_path, "w+") as fasta_file:
        for entry in data.itertuples():
            entry_dict = entry._asdict()

            stable_id = entry_dict["stable_id"]
            symbol = entry_dict["symbol"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id} {symbol}\n{sequence}\n")


def generate_dataset_statistics():
    data = load_data()
    print()

    # data_memory_usage = sys.getsizeof(data)
    # print("data memory usage: {}".format(sizeof_fmt(data_memory_usage)))
    # 3.8GiB

    num_sequences = len(data)
    print(f"{num_sequences} sequences")
    print()
    # 3805809 sequences

    # symbols occurrence frequency
    symbol_counts = data["symbol"].value_counts()
    print(symbol_counts)
    print()
    # nxpe3            378
    # pla1a            339
    # tbc1d9           335
    # NXPH1            329
    # CPNE3            323
    #                 ...
    # TRXH_1             1
    # BnaA10g05860D      1
    # BnaA03g11140D      1
    # BnaA09g47630D      1
    # BnaC04g10230D      1
    # Name: symbol, Length: 229133, dtype: int64

    symbol_counts_mean = symbol_counts.mean()
    symbol_counts_median = symbol_counts.median()
    symbol_counts_standard_deviation = symbol_counts.std()
    print(
        f"symbol counts mean: {symbol_counts_mean:.2f}, median: {symbol_counts_median:.2f}, standard deviation: {symbol_counts_standard_deviation:.2f}"
    )
    print()
    # symbol counts mean: 16.61, median: 1.00, standard deviation: 50.23

    sequence_length_mean = data["sequence"].str.len().mean()
    sequence_length_median = data["sequence"].str.len().median()
    sequence_length_standard_deviation = data["sequence"].str.len().std()
    print(
        f"sequence length mean: {sequence_length_mean:.2f}, median: {sequence_length_median:.2f}, standard deviation: {sequence_length_standard_deviation:.2f}"
    )
    print()
    # sequence length mean: 576.49, median: 442.00, standard deviation: 511.25


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
        response = requests.get(species_data_url)
        with open(species_data_path, "wb+") as f:
            f.write(response.content)
        logger.info(f"downloaded {species_data_path}")

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
        "gene_stable_id",
        "gene_version",
        "translation_stable_id",
        "translation_version",
        "Xref_symbol",
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
    if df_row["translation_version"] is None:
        translation_stable_id_version = df_row["translation_stable_id"]
    else:
        translation_stable_id_version = "{}.{}".format(
            df_row["translation_stable_id"], df_row["translation_version"]
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
    for assembly in assemblies:
        fasta_path = download_protein_sequences_fasta(assembly, ensembl_release)
        assembly.fasta_path = str(fasta_path.resolve())
    logger.info("protein sequences FASTA files in place")

    metadata_path = data_directory / pathlib.Path("metadata.pickle")
    if metadata_path.exists():
        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)
        logger.info(f"loaded existing metadata file {metadata_path}")
    else:
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
            logger.info(f"retrieved metadata for {assembly.name}, {species}, {assembly.assembly_accession}")

        with open(metadata_path, "wb") as f:
            pickle.dump(metadata, f)

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


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--generate_dataset",
        action="store_true",
        help="generate training dataset from genome assemblies in the latest Ensembl release",
    )
    argument_parser.add_argument("--dataset_cleanup", action="store_true")
    argument_parser.add_argument("--generate_dataset_statistics", action="store_true")
    argument_parser.add_argument("--save_all_datasets", action="store_true")
    argument_parser.add_argument("--save_all_sample_fasta_files", action="store_true")

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)
    log_file_path = data_directory / "dataset_generation.log"
    logger.add(log_file_path, format=LOGURU_FORMAT)

    if args.generate_dataset:
        generate_dataset()
    elif args.dataset_cleanup:
        dataset = load_dataset()
        dataset = dataset_cleanup(dataset)
        dataset_path = data_directory / "dataset.pickle"
        dataset.to_pickle(dataset_path)
        logger.info(f"dataset saved at {dataset_path}")
    elif args.generate_dataset_statistics:
        generate_dataset_statistics()
    elif args.save_all_datasets:
        save_all_datasets()
    elif args.save_all_sample_fasta_files:
        save_all_sample_fasta_files()
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
