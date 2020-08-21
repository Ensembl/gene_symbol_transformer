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


def save_n_most_frequent(n, max_frequency=None):
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

    # Remove the match(es) of the sequence with itself from the BLAST output.
    # Counterintuitively, a sequence may have more than 1 matches with itself,
    # potentially due to containing multiple ambiguous amino acids. One such case
    # is the sequence with stable_id "ENSPVAT00000000880", that has two matches,
    # with itself, both of 100 percent identity.
    if query_id in df["subject_id"].values:
        df_index = df[df["subject_id"] == query_id].index
        assert all(df.loc[df_index]["percent_identity"] == 100), query_id
        df.drop(labels=df_index, inplace=True)
    else:
        # A potential reason why the BLAST results doesn't contain a match of
        # the sequence itself, is that the sequence contains too many ambiguous
        # amino acids, i.e. "X". One such case is the sequence with stable id
        # "ENSTBET00000001003".
        # From the BLAST help page:
        # "The degenerate nucleotide codes [...] are treated as mismatches
        # in nucleotide alignment. Too many such degenerate codes within an input
        # nucleotide query will cause the BLAST webpage to reject the input."
        # https://blast.ncbi.nlm.nih.gov/Blast.cgi?CMD=Web&PAGE_TYPE=BlastDocs&DOC_TYPE=BlastHelp
        assert "X" * 1000 in sequence, sequence

    df[["subject_stable_id", "subject_symbol"]] = df["subject_id"].str.split(";", expand=True)

    return df


def generate_blast_features():
    """
    Parse the raw BLAST outputs and generate a dataframe with training features.
    """
    # n = 101
    n = 3

    data_pickle_path = data_directory / f"most_frequent_{n}.pickle"
    data = pd.read_pickle(data_pickle_path)

    blast_features = {}
    shelve_db_path = data_directory / f"most_frequent_{n}-blast_results.db"
    with shelve.open(str(shelve_db_path)) as blast_results_database:
        print("loading blast results database...")

        for stable_id, symbol in zip(data["stable_id"], data["symbol"]):
            query_id = f"{stable_id};{symbol}"
            blast_output_raw = blast_results_database[query_id]
            sequence = data[data["stable_id"] == stable_id]["sequence"].item()

            blast_output = parse_blast_output(query_id, blast_output_raw, sequence)

            # remove data not going to be used as training features
            columns = [
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
                "subject_symbol",
            ]
            blast_values = blast_output[columns]

            blast_features[stable_id] = {
                "symbol": symbol,
                "blast_values": blast_values,
            }

    # save dataframe to a pickle file
    blast_features_pickle_path = data_directory / f"blast_features-most_frequent_{n}.pickle"
    with open(blast_features_pickle_path, 'wb') as f:
        pickle.dump(blast_features, f, protocol=pickle.HIGHEST_PROTOCOL)


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

    args = argument_parser.parse_args()

    if args.save_most_frequent_101:
        save_n_most_frequent(n=101, max_frequency=297)
    elif args.save_most_frequent_3:
        save_n_most_frequent(n=3, max_frequency=335)
    else:
        generate_blast_features()


if __name__ == "__main__":
    main()
