#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Generate BLAST and raw sequences features.
"""


# standard library imports
import argparse
import io
import pathlib
import pickle
import shelve
import sys

# third party imports
import Bio
import pandas as pd

# project imports
import dataset_generation


USE_CACHE = True

data_directory = pathlib.Path("data")


def split_fasta_sequence(fasta_sequence):
    parts = fasta_sequence.split()
    description = parts[0].replace(">", "")
    sequence = parts[1]

    return description, sequence


def get_protein_letters():
    """
    Generate and return a list of protein letters that occur in the dataset and
    those that can potentially be used.
    """
    extended_IUPAC_protein_letters = Bio.Alphabet.IUPAC.ExtendedIUPACProtein.letters

    # cache the following operation, as it's very expensive in time and space
    if USE_CACHE:
        extra_letters = ["*"]
    else:
        data = dataset_generation.load_data()

        # generate a list of all protein letters that occur in the dataset
        dataset_letters = set(data["sequence"].str.cat())

        extra_letters = [
            letter
            for letter in dataset_letters
            if letter not in extended_IUPAC_protein_letters
        ]

    protein_letters = list(extended_IUPAC_protein_letters) + extra_letters
    assert len(protein_letters) == 27, protein_letters

    return protein_letters


def dataframe_to_fasta(dataframe, fasta_path):
    """
    Generate a FASTA file from a genes dataframe.
    """
    print("generating FASTA file from dataframe...")
    with open(fasta_path, "w+") as fasta_file:
        for entry in dataframe.itertuples():
            entry_dict = entry._asdict()

            symbol = entry_dict["symbol"]
            stable_id = entry_dict["stable_id"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id};{symbol}\n{sequence}\n")
    print(f"FASTA file saved at {fasta_path}")


def save_most_frequent_n(n, max_frequency=None):
    """
    Save the examples of the n most frequent symbols to a pickled dataframe and
    a FASTA file.

    Specify the max_frequency of the n symbols to run an extra validation check.
    """
    data = dataset_generation.load_data()

    symbol_counts = data["symbol"].value_counts()

    if max_frequency is not None:
        assert all(symbol_counts[:n] == symbol_counts[symbol_counts >= max_frequency])

    most_frequent_n = data[data["symbol"].isin(symbol_counts[:n].index)]

    # save dataframe to a pickle file
    pickle_path = data_directory / f"most_frequent_{n}.pickle"
    most_frequent_n.to_pickle(pickle_path)

    # save sequences to a FASTA file
    fasta_path = data_directory / f"most_frequent_{n}.fasta"
    dataframe_to_fasta(most_frequent_n, fasta_path)


def parse_blast_output(query_id, blast_output_raw, sequence):
    """
    """
    # open blast_output_raw string as a file
    with io.StringIO(blast_output_raw) as file_object:
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

    # A potential reason why the BLAST results doesn't contain a match of
    # the sequence itself, is that the sequence contains too many ambiguous
    # amino acids, i.e. "X".
    # From the BLAST help page:
    # "The degenerate nucleotide codes [...] are treated as mismatches
    # in nucleotide alignment. Too many such degenerate codes within an input
    # nucleotide query will cause the BLAST webpage to reject the input."
    # https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
    assert query_id in df["subject_id"].values, f"{query_id=}, {sequence=}"

    # Remove the match(es) of the sequence with itself from the BLAST output.
    # Counterintuitively, a sequence may have more than 1 matches with itself,
    # potentially due to containing multiple ambiguous amino acids. One such case
    # is the sequence with stable_id "ENSPVAT00000000880" and symbol "pla1a",
    # that has two matches, with itself, both of 100 percent identity.
    # Also counterintuitively, it is possible for a sequence to have a non 100 percent
    # identity match with itself if it contains too many ambiguous amino acids, i.e. "X".
    df_index = df[df["subject_id"] == query_id].index
    df.drop(labels=df_index, inplace=True)

    # Handle the case where the match(es) with itself are the only matches for
    # the sequence and at this point they have been removed.
    if df.empty:
        df["subject_stable_id"] = None
        df["subject_symbol"] = None
    else:
        df[["subject_stable_id", "subject_symbol"]] = df["subject_id"].str.split(
            ";", expand=True
        )

    return df


def generate_blast_features_most_frequent_n(n):
    """
    Parse the raw BLAST outputs and generate a dataframe with training features.
    """
    data_pickle_path = data_directory / f"most_frequent_{n}.pickle"
    data = pd.read_pickle(data_pickle_path)

    # generate a categorical data type from the list of unique labels (symbols)
    # to use for one-hot encoding
    labels = data["symbol"].unique().tolist()
    labels.sort()
    symbol_categorical_datatype = pd.CategoricalDtype(categories=labels, ordered=True)

    blast_features = {}
    shelve_db_path = data_directory / f"most_frequent_{n}-blast_results.db"
    with shelve.open(str(shelve_db_path), flag="r") as blast_results_database:
        print("loading blast results database...")

        for stable_id, symbol in zip(data["stable_id"], data["symbol"]):
            query_id = f"{stable_id};{symbol}"
            blast_output_raw = blast_results_database[query_id]
            sequence = data[data["stable_id"] == stable_id]["sequence"].item()

            blast_output = parse_blast_output(query_id, blast_output_raw, sequence)

            # generate an one-hot encoding of the subject_symbol
            subject_symbol_categorical = blast_output["subject_symbol"].astype(
                symbol_categorical_datatype
            )
            one_hot_subject_symbol = pd.get_dummies(
                subject_symbol_categorical, prefix="subject_symbol"
            )

            # merge the dataframes
            blast_values = pd.concat([blast_output, one_hot_subject_symbol], axis=1)

            # remove data not going to be used as training features
            columns = [
                "query_id",
                "subject_id",
                "subject_stable_id",
                "subject_symbol",
            ]
            blast_values.drop(columns=columns, inplace=True)

            # generate an one-hot encoding of the label (symbol)
            symbol_categorical = pd.Series(symbol, dtype=symbol_categorical_datatype)
            one_hot_symbol = pd.get_dummies(symbol_categorical, prefix="symbol")

            blast_features[stable_id] = {
                "symbol": symbol,
                "one_hot_symbol": one_hot_symbol,
                "blast_values": blast_values,
            }

    # save blast_features dictionary to a pickle file
    blast_features_pickle_path = (
        data_directory / f"most_frequent_{n}-blast_features.pickle"
    )
    with open(blast_features_pickle_path, "wb") as f:
        pickle.dump(blast_features, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"BLAST features file saved at {blast_features_pickle_path}")


def main():
    """
    main function
    """
    # DEBUG
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--save_most_frequent_101", action="store_true")
    argument_parser.add_argument("--save_most_frequent_3", action="store_true")
    argument_parser.add_argument(
        "--generate_blast_features_most_frequent_101", action="store_true"
    )
    argument_parser.add_argument(
        "--generate_blast_features_most_frequent_3", action="store_true"
    )

    args = argument_parser.parse_args()

    if args.save_most_frequent_101:
        save_most_frequent_n(n=101, max_frequency=297)
    elif args.save_most_frequent_3:
        save_most_frequent_n(n=3, max_frequency=335)
    elif args.generate_blast_features_most_frequent_101:
        generate_blast_features_most_frequent_n(n=101)
    elif args.generate_blast_features_most_frequent_3:
        generate_blast_features_most_frequent_n(n=3)
    else:
        print("nothing to do")


if __name__ == "__main__":
    main()
