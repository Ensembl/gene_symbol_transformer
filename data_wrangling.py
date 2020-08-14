#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Data wrangling.
"""


# standard library imports
import argparse
import pathlib
import sys

# third party imports
import pandas as pd

from Bio import SeqIO

# project imports


# DEBUG = True
DEBUG = False

data_directory = pathlib.Path("data")


def fasta_to_dataframe(fasta_path):
    """
    Generate a pandas dataframe from a FASTA file.

    dataframe columns: description, sequence
    """
    records = []
    with open(fasta_path) as fasta_file:
        for fasta_record in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            records.append(
                {"description": fasta_record[0], "sequence": fasta_record[1]}
            )

    records_dataframe = pd.DataFrame(records)

    return records_dataframe


def merge_metadata_sequences():
    """
    Merge the metadata CSV file and the sequences FASTA file in a single CSV file.
    """
    all_data_pickle_path = data_directory / "all_species_metadata_sequences.pickle"

    # exit function if the output file exists
    if all_data_pickle_path.is_file():
        print(f"{all_data_pickle_path} file already exists")
        return

    metadata_csv_path = data_directory / "original" / "all_species.csv"
    sequences_fasta_path = data_directory / "original" / "all_species.fa"

    # read the metadata csv file to a pandas dataframe
    metadata = pd.read_csv(metadata_csv_path, sep="\t")
    if DEBUG:
        print(metadata.head())
        print()
        metadata.info()
        print()
        print(metadata.describe())
        print()
    assert metadata["stable_id"].nunique() == len(metadata)

    # generate a pandas dataframe from the sequences FASTA file
    sequences = fasta_to_dataframe(sequences_fasta_path)
    if DEBUG:
        print(sequences.head())
        print()
        sequences.info()
        print()
        print(sequences.describe())
        print()
    assert sequences["description"].nunique() == len(sequences)

    # merge the two dataframes in a single one
    merged_data = pd.merge(
        left=metadata, right=sequences, left_on="stable_id", right_on="description"
    )
    if DEBUG:
        print(merged_data.head())
        print()
        merged_data.info()
        print()
    assert (
        merged_data["stable_id"].nunique()
        == merged_data["description"].nunique()
        == len(merged_data)
    )

    # remove duplicate description column
    merged_data.drop(columns=["description"], inplace=True)
    if DEBUG:
        merged_data.info()
        print()

    # save the merged dataframe to a pickle file
    merged_data.to_pickle(all_data_pickle_path)


def data_wrangling():
    """
    - simplify a couple of column names
    - ignore capitalization of symbol names
    """
    # load the original data file
    all_data_pickle_path = data_directory / "all_species_metadata_sequences.pickle"
    data = pd.read_pickle(all_data_pickle_path)

    # simplify a couple of column names
    if "display_xref.display_id" in data:
        print("renaming dataframe columns...")
        data = data.rename(
            columns={
                "display_xref.display_id": "symbol_original",
                "display_xref.db_display_name": "db_display_name",
            }
        )

    # ignore capitalization of symbol names
    if "symbol" not in data:
        print("generating lower case symbol column...")
        data.insert(loc=3, column="symbol", value=data["symbol_original"].str.lower())

    # save the dataframe to a new data file
    data_pickle_path = data_directory / "data.pickle"
    data.to_pickle(data_pickle_path)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--merge_metadata_sequences", action="store_true")

    args = argument_parser.parse_args()

    if args.merge_metadata_sequences:
        merge_metadata_sequences()
    else:
        data_wrangling()


if __name__ == "__main__":
    main()
