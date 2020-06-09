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


def merge_sequences_and_metadata():
    """
    Merge the sequences FASTA file and the metadata CSV file to a single CSV file.
    """
    data_directory = pathlib.Path("data")

    metadata_csv_filename = "all_species.csv"
    sequences_fasta_filename = "all_species.fa"

    metadata_csv_path = data_directory / metadata_csv_filename
    sequences_fasta_path = data_directory / sequences_fasta_filename

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

    # merge the two dataframes to a single one
    combined_data = pd.merge(left=metadata, right=sequences, left_on="stable_id", right_on="description")
    if DEBUG:
        print(combined_data.head())
        print()
        combined_data.info()
        print()
    assert combined_data["stable_id"].nunique() == combined_data["description"].nunique() == len(combined_data)

    # remove duplicate description column
    combined_data.drop(columns=["description"], inplace=True)
    if DEBUG:
        combined_data.info()
        print()

    # save merged dataframe as CSV file
    combined_data_filename = "all_species_metadata_sequences.csv"
    combined_data_path = data_directory / combined_data_filename
    # if not combined_data_path.is_file():
    combined_data.to_csv(combined_data_path, sep="\t", index=False)


def main():
    """
    main function
    """
    merge_sequences_and_metadata()


if __name__ == "__main__":
    main()
