#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Feature generation.
"""


# standard library imports
import argparse
import csv
import io
import pathlib
import shelve
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
            symbol = entry[4]
            stable_id = entry[1]
            sequence = entry[9]
            fasta_file.write(f">{stable_id};{symbol}\n{sequence}\n")
    print(f"FASTA file saved at {fasta_path}")


def select_to_fasta():
    """
    Save a subset of the sequences to a FASTA file.
    """
    data_pickle_path = data_directory / "data.pickle"
    print("loading data file...")
    data = pd.read_pickle(data_pickle_path)
    print("data file loaded")

    symbol_counts = data["symbol"].value_counts()

    most_frequent_100 = data[data["symbol"].isin(symbol_counts[:100].index)]

    fasta_path = data_directory / "most_frequent_100.fasta"
    dataframe_to_fasta(most_frequent_100, fasta_path)


def parse_blast_results(blast_results):
    """
    """
    # open blast_results string as a file for easy consumption by csv.reader
    with io.StringIO(blast_results) as file_object:
        fieldnames = [
                "query_id", "subject_id", "percent_identity", "alignment_length", "mismatches", "gap_opens", "query_start", "query_end", "subject_start", "subject_end", "evalue", "bit_score"
                ]
        for row in csv.DictReader(file_object, fieldnames=fieldnames, delimiter="\t"):
            print(row)
            break


def split_fasta_sequence(fasta_sequence):
    parts = fasta_sequence.split()
    description = parts[0].replace(">", "")
    sequence = parts[1]

    return description, sequence


def generate_blast_features():
    """
    Parse raw BLAST results and generate a dictionary with important values.
    """
    # shelve_db_path = data_directory / "most_frequent_100-blast_results.db"
    shelve_db_path = data_directory / "blast_results_sample.db"

    with shelve.open(str(shelve_db_path)) as blast_results:
        print("loading blast_results database...")
        print()

        for fasta_sequence, blast_output in blast_results.items():
            description, sequence = split_fasta_sequence(fasta_sequence)
            print(description)
            print()
            blast_features = parse_blast_results(blast_output)
            break


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--select_to_fasta", action="store_true")

    args = argument_parser.parse_args()

    if args.select_to_fasta:
        select_to_fasta()
    else:
        generate_blast_features()


if __name__ == "__main__":
    main()
