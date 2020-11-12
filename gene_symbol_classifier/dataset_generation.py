#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Copyright 2020 EMBL-European Bioinformatics Institute
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
Merge original data files, normalize, cleanup, and filter examples to a single pandas dataframe saved as a pickle file.
"""


# standard library imports
import argparse
import pathlib
import sys

# third party imports
import Bio
import pandas as pd

from Bio import SeqIO

# project imports


# DEBUG = True
DEBUG = False

data_directory = pathlib.Path("data")


def fasta_to_dataframe(fasta_path):
    """
    Generate a pandas dataframe with columns "description" and "sequence" from
    a FASTA file.
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
    - simplify some column names
    - use symbol names in lower case
    267536 unique original symbol names
    233824 unique lower case symbol names
    12.6% reduction

    - filter out "Clone-based (Ensembl) gene" examples
      No standard name exists for them. 4691 examples filtered out.
    """
    # load the original data file
    all_data_pickle_path = data_directory / "all_species_metadata_sequences.pickle"
    print(f"loading {all_data_pickle_path} file...")
    data = pd.read_pickle(all_data_pickle_path)
    print(f"{all_data_pickle_path} file loaded")

    # simplify some column names
    print("renaming dataframe columns...")
    data = data.rename(
        columns={
            "display_xref.display_id": "symbol_original",
            "display_xref.db_display_name": "db_display_name",
        }
    )

    # use symbol names in lower case
    print("generating lower case symbol column...")
    data["symbol"] = data["symbol_original"].str.lower()

    # filter out "Clone-based (Ensembl) gene" examples
    print(
        'creating "include" column and filtering out "Clone-based (Ensembl) gene" examples'
    )
    data["include"] = data["db_display_name"] != "Clone-based (Ensembl) gene"

    # save the dataframe to a new data file
    data_pickle_path = data_directory / "data.pickle"
    data.to_pickle(data_pickle_path)


def save_most_frequent_n(n, max_frequency=None):
    """
    Save the examples of the n most frequent symbols to a pickled dataframe and
    a FASTA file.

    Specify the max_frequency of the n symbols to run an extra validation check.
    """
    data = load_data()

    symbol_counts = data["symbol"].value_counts()

    # verify that max_frequency is the cutoff limit for selected symbols
    if max_frequency is not None:
        assert all(symbol_counts[:n] == symbol_counts[symbol_counts >= max_frequency])
        assert symbol_counts[n] < max_frequency

    most_frequent_n = data[data["symbol"].isin(symbol_counts[:n].index)]

    # save dataframe to a pickle file
    pickle_path = data_directory / f"most_frequent_{n}.pickle"
    most_frequent_n.to_pickle(pickle_path)
    print(f"pickle file of the most {n} frequent symbol sequences saved at {pickle_path}")

    # save sequences to a FASTA file
    fasta_path = data_directory / f"most_frequent_{n}.fasta"
    with open(fasta_path, "w+") as fasta_file:
        for entry in most_frequent_n.itertuples():
            entry_dict = entry._asdict()

            stable_id = entry_dict["stable_id"]
            symbol = entry_dict["symbol"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id};{symbol}\n{sequence}\n")

    print(f"FASTA file of the most {n} frequent symbol sequences saved at {fasta_path}")


def load_data():
    """
    Load data dataframe, excluding filtered out examples.
    """
    data_pickle_path = data_directory / "data.pickle"
    print("loading data...")
    data = pd.read_pickle(data_pickle_path)
    print("data loaded")

    # exclude filtered out examples
    data = data[data["include"] == True]

    return data


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--merge_metadata_sequences", action="store_true")
    argument_parser.add_argument("--data_wrangling", action="store_true")
    argument_parser.add_argument("--save_most_frequent_3", action="store_true")
    argument_parser.add_argument("--save_most_frequent_101", action="store_true")
    argument_parser.add_argument("--save_most_frequent_1013", action="store_true")
    argument_parser.add_argument("--save_most_frequent_10059", action="store_true")
    argument_parser.add_argument("--save_most_frequent_20147", action="store_true")
    argument_parser.add_argument("--save_most_frequent_25028", action="store_true")
    argument_parser.add_argument("--save_most_frequent_30591", action="store_true")

    args = argument_parser.parse_args()

    if args.merge_metadata_sequences:
        merge_metadata_sequences()
    elif args.data_wrangling:
        data_wrangling()
    elif args.save_most_frequent_3:
        save_most_frequent_n(n=3, max_frequency=335)
    elif args.save_most_frequent_101:
        save_most_frequent_n(n=101, max_frequency=297)
    elif args.save_most_frequent_1013:
        save_most_frequent_n(n=1013, max_frequency=252)
    elif args.save_most_frequent_10059:
        save_most_frequent_n(n=10059, max_frequency=165)
    elif args.save_most_frequent_20147:
        save_most_frequent_n(n=20147, max_frequency=70)
    elif args.save_most_frequent_25028:
        save_most_frequent_n(n=25028, max_frequency=23)
    elif args.save_most_frequent_30591:
        save_most_frequent_n(n=30591, max_frequency=5)
    else:
        print("nothing to do")


if __name__ == "__main__":
    main()
