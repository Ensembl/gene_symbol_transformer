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
import sys
import time

# third party imports
import ensembl_rest
import pandas as pd

from loguru import logger

# project imports
from utils import (
    PrettySimpleNamespace,
    data_directory,
    dev_datasets_num_symbols,
    download_protein_sequences_fasta,
    fasta_to_dict,
    generate_assemblies_metadata,
    get_xref_canonical_translations,
    get_ensembl_release,
    get_taxonomy_id_clade,
    load_dataset,
    logging_format,
    save_dev_datasets,
    sequences_directory,
    sizeof_fmt,
)


def generate_dataset():
    """
    Download canonical translations of protein coding genes from all genome assemblies
    in the latest Ensembl release.
    """
    ensembl_release = get_ensembl_release()
    logger.info(f"Ensembl release {ensembl_release}")

    assemblies_df = generate_assemblies_metadata()

    logger.info(f"downloading protein sequences FASTA files")
    for index, values in assemblies_df.iterrows():
        assembly = PrettySimpleNamespace(**values.to_dict())
        _fasta_path = download_protein_sequences_fasta(assembly, ensembl_release)
    logger.info("protein sequences FASTA files in place")

    logger.info(
        f"retrieving protein coding gene canonical translation IDs and metadata from the public Ensembl MySQL server"
    )
    canonical_translations_list = []
    for index, values in assemblies_df.iterrows():
        assembly = PrettySimpleNamespace(**values.to_dict())
        # delay between SQL queries
        time.sleep(0.1)

        assembly_translations = get_xref_canonical_translations(assembly.core_db)
        num_assembly_translations = len(assembly_translations)

        logger.info(
            f"retrieved {num_assembly_translations} canonical translations for {assembly.common_name}, {assembly.scientific_name}, {assembly.assembly_accession}"
        )

        if num_assembly_translations == 0:
            continue

        sequences_fasta_path = sequences_directory / assembly.fasta_filename
        assembly_fasta_dict = fasta_to_dict(sequences_fasta_path)

        assembly_translations["sequence"] = assembly_translations.apply(
            lambda x: get_sequence_from_assembly_fasta_dict(x, assembly_fasta_dict),
            axis=1,
        )

        assembly_translations["assembly_accession"] = assembly.assembly_accession
        assembly_translations["scientific_name"] = assembly.scientific_name
        assembly_translations["common_name"] = assembly.common_name
        assembly_translations["taxonomy_id"] = assembly.taxonomy_id
        assembly_translations["clade"] = assembly.clade
        assembly_translations["core_db"] = assembly.core_db

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
        "transcript.stable_id",
        "transcript.version",
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
        "clade",
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
    logger.add(sys.stderr, format=logging_format)
    data_directory.mkdir(exist_ok=True)
    log_file_path = data_directory / "dataset_generation.log"
    logger.add(log_file_path, format=logging_format)

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
