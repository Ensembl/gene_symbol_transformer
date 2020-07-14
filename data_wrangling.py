#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Gene symbol classifier data exploration code.
"""


# standard library imports
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
            records.append({"description": fasta_record[0], "sequence": fasta_record[1]})

    records_dataframe = pd.DataFrame(records)

    return records_dataframe


def merge_metadata_sequences():
    """
    Merge the metadata CSV file and the sequences FASTA file in a single CSV file.
    """
    merged_data_path = data_directory / "all_species_metadata_sequences.csv"

    # exit the function if the merged file has already been generated
    if merged_data_path.is_file():
        print("the merged file has already been generated, exiting")
        return

    metadata_csv_path = data_directory / "all_species.csv"
    sequences_fasta_path = data_directory / "all_species.fa"

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
    merged_data = pd.merge(left=metadata, right=sequences, left_on="stable_id", right_on="description")
    if DEBUG:
        print(merged_data.head())
        print()
        merged_data.info()
        print()
    assert merged_data["stable_id"].nunique() == merged_data["description"].nunique() == len(merged_data)

    # remove duplicate description column
    merged_data.drop(columns=["description"], inplace=True)
    if DEBUG:
        merged_data.info()
        print()

    # save merged dataframe as CSV file
    merged_data.to_csv(merged_data_path, sep="\t", index=False)


def data_wrangling():
    """
    - simplify a couple of column names
    - ignore capitalization of symbol names
    """
    csv_path = data_directory / "all_species_metadata_sequences.csv"
    data = pd.read_csv(csv_path, sep="\t")

    metadata_sequences_path = data_directory / "metadata_sequences.csv"

    # simplify a couple of column names
    if "display_xref.display_id" in data:
        data = data.rename(columns={"display_xref.display_id": "symbol", "display_xref.db_display_name": "db_display_name"})

    # ignore capitalization of symbol names
    if "symbol_lower" not in data:
        data.insert(loc=3, column="symbol_lower", value=data["symbol"].str.lower())

    data.to_csv(metadata_sequences_path, sep="\t", index=False)


def main():
    """
    main function
    """
    # merge_metadata_sequences()

    data_wrangling()


if __name__ == "__main__":
    main()
