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
Pipeline to BLAST specified sequences and save the raw results to a shelve database file.

Generate BLAST features.
"""


# standard library imports
import argparse
import io
import pathlib
import pickle
import shelve
import subprocess
import sys

# third party imports
import pandas as pd

from Bio import SeqIO

# project imports


data_directory = pathlib.Path("data")


def blast_sequence(fasta_sequence, db, evalue=None, word_size=None, outfmt="6"):
    """
    BLAST a FASTA sequence using the specified BLAST database db.

    Documentation of BLAST arguments used:

    -db <String>
      BLAST database name
       * Incompatible with:  subject, subject_loc

    -evalue <Real>
      Expectation value (E) threshold for saving hits
      Default = `10'

    -word_size <Integer, >=2>
      Word size for wordfinder algorithm

    *** Formatting options
    -outfmt <String>
      alignment view options:
        0 = pairwise,
        1 = query-anchored showing identities,
        2 = query-anchored no identities,
        3 = flat query-anchored, show identities,
        4 = flat query-anchored, no identities,
        5 = XML Blast output,
        6 = tabular,
        7 = tabular with comment lines,
        8 = Text ASN.1,
        9 = Binary ASN.1,
       10 = Comma-separated values,
       11 = BLAST archive format (ASN.1)
       12 = JSON Seqalign output
    """
    arguments = [
        "blastp",
        "-db",
        db,
        "-outfmt",
        outfmt,
    ]
    if evalue is not None:
        arguments.extend(["-evalue", evalue])
    if word_size is not None:
        arguments.extend(["-word_size", word_size])

    completed_process = subprocess.run(
        arguments, input=fasta_sequence, capture_output=True, text=True
    )
    output = completed_process.stdout

    return output


def generate_blast_results(db, fasta_path, shelve_db_path, total):
    """
    Generate a BLAST results database.
    """
    with open(fasta_path) as fasta_file, shelve.open(
        str(shelve_db_path)
    ) as blast_results:
        for counter, fasta_record in enumerate(
            SeqIO.FastaIO.SimpleFastaParser(fasta_file), start=1
        ):
            description = fasta_record[0]
            sequence = fasta_record[1]
            fasta_sequence = f">{description}\n{sequence}\n"

            blast_output = blast_sequence(fasta_sequence, db=db)

            blast_results[description] = blast_output
            print(f"{description} : {counter} out of {total}")


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
    argument_parser.add_argument(
        "--generate_most_frequent_101_blast_results", action="store_true"
    )
    argument_parser.add_argument(
        "--generate_most_frequent_3_blast_results", action="store_true"
    )
    argument_parser.add_argument(
        "--generate_blast_features_most_frequent_101", action="store_true"
    )
    argument_parser.add_argument(
        "--generate_blast_features_most_frequent_3", action="store_true"
    )

    args = argument_parser.parse_args()

    if args.generate_most_frequent_101_blast_results:
        n = 101
        db = data_directory / f"blast_databases/most_frequent_{n}/most_frequent_{n}"
        fasta_path = data_directory / f"most_frequent_{n}.fasta"
        shelve_db_path = data_directory / f"most_frequent_{n}-blast_results.db"
        total = 31204
        generate_blast_results(
            db=db, fasta_path=fasta_path, shelve_db_path=shelve_db_path, total=total
        )
    elif args.generate_most_frequent_3_blast_results:
        n = 3
        db = data_directory / f"blast_databases/most_frequent_{n}/most_frequent_{n}"
        fasta_path = data_directory / f"most_frequent_{n}.fasta"
        shelve_db_path = data_directory / f"most_frequent_{n}-blast_results.db"
        total = 1052
        generate_blast_results(
            db=db, fasta_path=fasta_path, shelve_db_path=shelve_db_path, total=total
        )
    elif args.generate_blast_features_most_frequent_101:
        generate_blast_features_most_frequent_n(n=101)
    elif args.generate_blast_features_most_frequent_3:
        generate_blast_features_most_frequent_n(n=3)
    else:
        print("nothing to do")


if __name__ == "__main__":
    main()
