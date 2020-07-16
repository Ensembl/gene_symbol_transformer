#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Feature generation.
"""


# standard library imports
import pathlib
import sys

# third party imports
import pandas as pd

# project imports


data_directory = pathlib.Path("data")


def dataframe_to_fasta(dataframe, fasta_path):
    """
    Generate a FASTA file from a sequences dataframe.
    """
    print("generating FASTA file from dataframe...")
    with open(fasta_path, "w+") as fasta_file:
        for entry in dataframe.itertuples():
            symbol = entry[3]
            stable_id = entry[1]
            sequence = entry[9]
            fasta_file.write(f">{symbol};{stable_id}\n{sequence}\n")
    print(f"FASTA file saved at {fasta_path}")


def select_to_fasta():
    """
    Save a subset of the sequences to a FASTA file.
    """
    data_csv_path = data_directory / "data.csv"
    print("loading data CSV...")
    data = pd.read_csv(data_csv_path, sep="\t")
    print("data CSV loaded")

    data_csv_path = data_directory / "data.csv"

    symbol_counts = data["symbol"].value_counts()

    most_frequent_100 = data[data["symbol"].isin(symbol_counts[:100].index)]

    fasta_path = data_directory / "most_frequent_100.fasta"
    dataframe_to_fasta(most_frequent_100, fasta_path)


def main():
    """
    main function
    """
    select_to_fasta()


if __name__ == "__main__":
    main()
