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


def main():
    """
    main function
    """
    data_directory = pathlib.Path("data")
    fasta_filename = "all_species.fa"

    fasta_path = data_directory / fasta_filename

    df = fasta_to_dataframe(fasta_path)

    df.info()
    print()
    print(df.describe())


if __name__ == "__main__":
    main()
