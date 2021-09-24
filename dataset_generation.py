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
import json
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
    generate_canonical_protein_sequences_fasta,
    fasta_to_dict,
    get_assemblies_metadata,
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

    assemblies = get_assemblies_metadata()

    logger.info(f"downloading protein sequences FASTA files")
    for assembly in assemblies:
        _canonical_fasta_path = generate_canonical_protein_sequences_fasta(
            assembly, ensembl_release
        )
    logger.info("protein sequences FASTA files in place")

    logger.info(
        f"retrieving protein coding gene canonical translation IDs and metadata from the public Ensembl MySQL server"
    )
    canonical_translations_list = []
    for assembly in assemblies:
        # delay between SQL queries
        time.sleep(0.1)

        assembly_translations = get_xref_canonical_translations(assembly.core_db)
        num_assembly_translations = len(assembly_translations)

        logger.info(
            f"retrieved {num_assembly_translations} Xref canonical translations for {assembly.common_name}, {assembly.scientific_name}, {assembly.assembly_accession}"
        )

        if num_assembly_translations == 0:
            continue

        canonical_fasta_filename = assembly.fasta_filename.replace(
            "pep.all.fa", "pep.all_canonical.fa"
        )
        canonical_sequences_fasta_path = sequences_directory / canonical_fasta_filename
        assembly_fasta_dict = fasta_to_dict(canonical_sequences_fasta_path)

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

    save_symbols_metadata(dataset)

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
        "xref.description",
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


def save_symbols_metadata(dataset):
    """
    Generate a dictionary with symbols to symbol source and description and store as
    a JSON file.
    """
    logger.info("generating symbols metadata...")

    symbol_groups = dataset.groupby(["symbol"])

    symbol_sources_list = [
        "HGNC Symbol",
        "ZFIN",
        "MGI Symbol",
        "VGNC Symbol",
        "RGD Symbol",
        "Xenbase",
        "FlyBase gene name",
        "WormBase Locus",
        "WormBase Gene Sequence-name",
        "SGD gene name",
        "Clone-based (Ensembl) gene",
        "FlyBase annotation",
        "UniProtKB Gene Name",
        "NCBI gene (formerly Entrezgene)",
    ]

    symbols_metadata = {}
    for symbol, group in symbol_groups:
        symbol_sources = set(group["external_db.db_display_name"])
        for symbol_source in symbol_sources_list:
            if symbol_source in symbol_sources:
                symbols_metadata[symbol] = {"symbol_source": symbol_source}
                break

    scientific_names_priority_list = [
        "Homo sapiens",
        "Mus musculus",
        "Danio rerio",
    ]

    for symbol, group in symbol_groups:
        symbol_scientific_names = set(group["scientific_name"])

        scientific_name = None
        for priority_scientific_name in scientific_names_priority_list:
            for symbol_scientific_name in symbol_scientific_names:
                if priority_scientific_name in symbol_scientific_name:
                    scientific_name = priority_scientific_name
                    break
            if scientific_name:
                break

        if scientific_name:
            symbol_description = group.loc[
                group["scientific_name"].str.contains(scientific_name)
            ]["xref.description"].iloc[0]
        else:
            descriptions = set(group["xref.description"])
            for description_item in descriptions:
                if description_item not in {"", "None"}:
                    symbol_description = description_item
                    break
            else:
                symbol_description = None

        symbols_metadata[symbol]["description"] = symbol_description

    symbols_metadata_filename = "symbols_metadata.json"
    symbols_metadata_path = data_directory / symbols_metadata_filename

    with open(symbols_metadata_path, "w") as f:
        json.dump(symbols_metadata, f, sort_keys=True, indent=4)
    logger.info(f"symbols metadata saved at {symbols_metadata_path}")


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
    if df_row["translation.version"] == "None":
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
        "--save_symbols_metadata",
        action="store_true",
        help="save symbols source and description to a JSON file",
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
    elif args.save_symbols_metadata:
        dataset = load_dataset()
        save_symbols_metadata(dataset)
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
