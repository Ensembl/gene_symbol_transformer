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
Generate training dataset of protein sequences for canonical translations from assemblies
in the latest Ensembl release and useful metadata, development datasets containing
a subset of the full dataset for faster prototyping, and dataset statistics.
"""


# standard library imports
import argparse
import json
import sys
import time

# third party imports
import pandas as pd

# project imports
from utils import (
    add_log_file_handler,
    data_directory,
    dev_datasets_num_symbols,
    fasta_to_dict,
    generate_canonical_protein_sequences_fasta,
    get_assemblies_metadata,
    get_ensembl_release,
    get_xref_canonical_translations,
    load_dataset,
    logger,
    main_release_sequences_directory,
    sizeof_fmt,
)


def generate_datasets():
    """
    Download canonical translations of protein coding genes from all genome assemblies
    in the latest Ensembl release, generate pandas dataframes with the full and dev
    transcripts dataset, and save them as pickle files.
    """
    ensembl_release = get_ensembl_release()
    logger.info(f"Ensembl release {ensembl_release}")

    assemblies = get_assemblies_metadata()

    logger.info("downloading protein sequences FASTA files")
    for assembly in assemblies:
        _canonical_fasta_path = generate_canonical_protein_sequences_fasta(
            assembly, ensembl_release
        )
    logger.info("protein sequences FASTA files in place")

    logger.info(
        "retrieving protein coding gene canonical translation IDs and metadata from the public Ensembl MySQL server"
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
        canonical_sequences_fasta_path = (
            main_release_sequences_directory / canonical_fasta_filename
        )
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

    generate_dataset_statistics(dataset)

    # save dataset as a pickle file
    dataset_path = data_directory / "dataset.pickle"
    dataset.to_pickle(dataset_path)
    logger.info(f"dataset saved at {dataset_path}")

    generate_dev_datasets(dataset)


def generate_dev_datasets(dataset):
    """
    Generate and save subsets of the full dataset for faster loading during development,
    and the datasets as FASTA files.

    Args:
        dataset (pandas DataFrame): full dataset dataframe
    """
    symbol_counts = dataset["symbol"].value_counts()

    for num_symbols in dev_datasets_num_symbols:
        logger.info(f"generating {num_symbols} most frequent symbols dev dataset ...")

        dev_dataset = dataset[dataset["symbol"].isin(symbol_counts[:num_symbols].index)]

        generate_dataset_statistics(dev_dataset)

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
        "VGNC Symbol",
        "MGI Symbol",
        "ZFIN",
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

    # get source and corresponding description with priority of the order in symbol_sources_list
    symbols_metadata = {}
    for symbol, group in symbol_groups:
        # sort the group dataframe by symbol_sources_list
        group["external_db.db_display_name"] = pd.Categorical(
            group["external_db.db_display_name"], categories=symbol_sources_list
        )
        group = group.sort_values("external_db.db_display_name")

        description = group["xref.description"].iloc[0]
        symbol_source = group["external_db.db_display_name"].iloc[0]
        symbols_metadata[symbol] = {
            "description": description,
            "source": symbol_source,
        }

    symbols_metadata_filename = "symbols_metadata.json"
    symbols_metadata_path = data_directory / symbols_metadata_filename

    with open(symbols_metadata_path, "w") as file:
        json.dump(symbols_metadata, file, sort_keys=True, indent=4)
    logger.info(f"symbols metadata saved at {symbols_metadata_path}")


def generate_dataset_statistics(dataset):
    """
    Generate and log dataset statistics.
    """
    dataset_object_size = sys.getsizeof(dataset)
    logger.info("dataset object memory usage: {}".format(sizeof_fmt(dataset_object_size)))

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


def dataframe_to_fasta(dataframe, fasta_path):
    """
    Save a dataframe containing entries of sequences and metadata to a FASTA file.
    """
    with open(fasta_path, "w+") as fasta_file:
        for _index, values in dataframe.iterrows():
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


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--save_symbols_metadata",
        action="store_true",
        help="save symbols source and description to a JSON file",
    )

    args = argument_parser.parse_args()

    log_file_path = data_directory / "dataset_generation.log"
    add_log_file_handler(logger, log_file_path)

    if args.save_symbols_metadata:
        dataset = load_dataset()
        save_symbols_metadata(dataset)
    else:
        generate_datasets()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
