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
    with open(fasta_path, "w+") as fasta_file:
        for entry in dataframe.itertuples():
            symbol = entry[3]
            stable_id = entry[1]
            sequence = entry[9]
            fasta_file.write(f">{symbol};{stable_id}\n{sequence}")


def select_to_fasta():
    """
    Save a subset of the sequences to a FASTA file.
    """
    data_csv_path = data_directory / "data.csv"
    print("loading data CSV...")
    data = pd.read_csv(data_csv_path, sep="\t")
    print("data CSV loaded")

    data_csv_path = data_directory / "data.csv"

    fasta_path = "output.fasta"
    dataframe_to_fasta(data, fasta_path)


def main():
    """
    main function
    """
    select_to_fasta()


if __name__ == "__main__":
    main()
