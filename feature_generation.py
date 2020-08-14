#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Feature generation.
"""


# standard library imports
import argparse
import io
import pathlib
import shelve
import sys

# third party imports
import pandas as pd

# project imports
import dataset_generation


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
    data = dataset_generation.load_data()

    symbol_counts = data["symbol"].value_counts()

    most_frequent_100 = data[data["symbol"].isin(symbol_counts[:100].index)]

    fasta_path = data_directory / "most_frequent_100.fasta"
    dataframe_to_fasta(most_frequent_100, fasta_path)


def parse_blast_results(blast_results):
    """
    """
    # open blast_results string as a file
    with io.StringIO(blast_results) as file_object:
        column_names = [
            "query_id",
            "subject_id",
            "percent_identity",
            "alignment_length",
            "mismatches",
            "gap_opens",
            "query_start",
            "query_end",
            "subject_start",
            "subject_end",
            "evalue",
            "bit_score",
        ]
        df = pd.read_csv(file_object, delimiter="\t", names=column_names)

    # verify existence of the sequence itself and remove it from the BLAST results
    assert df.loc[0]["query_id"] == df.loc[0]["subject_id"]
    df.drop(labels=0, inplace=True)

    df[["stable_id", "label"]] = df["subject_id"].str.split(";", expand=True)

    print(df.head())

    return df


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

        columns = ["fasta_sequence", "blast_output"]
        blast_results_dataframe = pd.DataFrame(blast_results.items(), columns=columns)

    blast_results = blast_results_dataframe

    get_description = lambda x: split_fasta_sequence(x)[0]
    get_sequence = lambda x: split_fasta_sequence(x)[1]

    blast_results["description"] = blast_results["fasta_sequence"].apply(
        get_description
    )
    blast_results["sequence"] = blast_results["fasta_sequence"].apply(get_sequence)

    blast_results.drop(columns=["fasta_sequence"], inplace=True)

    blast_results[["stable_id", "symbol"]] = blast_results["description"].str.split(
        ";", expand=True
    )

    columns = ["description", "stable_id", "symbol", "sequence", "blast_output"]
    blast_results = blast_results.reindex(columns=columns)

    blast_features = parse_blast_results(blast_results.loc[0]["blast_output"])

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    # print(blast_features)


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
