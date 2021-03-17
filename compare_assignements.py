#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
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
Compare the classifier assigned symbols to the Xref assigned ones.
"""


# standard library imports
import argparse
import csv
import itertools
import pathlib
import sys

# third party imports
import pandas as pd

from Bio import SeqIO
from loguru import logger

# project imports


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"


def read_fasta_in_chunks(fasta_file_path, num_entries_in_chunk=1024):
    """
    Read a FASTA file in chunks, returning a TODO of tuples of two strings,
    the FASTA title line without the leading ">" character, and
    the sequence with any whitespace removed.

    num_entries_in_chunk: number of entries in each chunk
    """
    # Count the number of entries in the FASTA file up to the maximum of
    # the num_entries_in_chunk chunk size. If the FASTA file has fewer entries
    # than num_entries_in_chunk, re-assign the latter to that smaller value.
    with open(fasta_file_path) as fasta_file:
        num_entries_counter = 0
        for _ in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            num_entries_counter += 1
            if num_entries_counter == num_entries_in_chunk:
                break
        else:
            num_entries_in_chunk = num_entries_counter

    # read the FASTA file in chunks
    with open(fasta_file_path) as fasta_file:
        fasta_generator = SeqIO.FastaIO.SimpleFastaParser(fasta_file)
        args = [fasta_generator] * num_entries_in_chunk
        fasta_chunks_iterator = itertools.zip_longest(*args)

        for fasta_entries in fasta_chunks_iterator:
            if fasta_entries[-1] is None:
                fasta_entries = [entry for entry in fasta_entries if entry is not None]
            yield fasta_entries


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--sequences_fasta",
        help="protein sequences FASTA file path",
    )
    argument_parser.add_argument(
        "--predictions_csv",
        help="predictions CSV file path",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)

    if args.sequences_fasta is None or args.predictions_csv is None:
        argument_parser.print_help()
        sys.exit()

    sequences_fasta_path = pathlib.Path(args.sequences_fasta)
    predictions_csv_path = pathlib.Path(args.predictions_csv)

    assignments = []
    with open(predictions_csv_path, "r") as predictions_file:
        csv_reader = csv.reader(predictions_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            for fasta_entry, csv_row in zip(fasta_entries, csv_reader):
                fasta_title_parts = fasta_entry[0].split()
                fasta_stable_id = fasta_title_parts[0]
                for title_part in fasta_title_parts:
                    if "gene_symbol" in title_part:
                        # 12 = len("gene_symbol") + 1
                        xref_symbol = title_part[12:]

                csv_stable_id = csv_row[0]
                classifier_symbol = csv_row[1]

                assert fasta_stable_id == csv_stable_id, f"{fasta_stable_id=}, {csv_stable_id=}"

                assignments.append(
                    (fasta_stable_id, classifier_symbol, xref_symbol)
                )

    dataframe_columns = ["stable_id", "classifier_symbol","xref_symbol"]
    assignments_df = pd.DataFrame(assignments, columns=dataframe_columns)

    assignments_csv_path = pathlib.Path(
        f"{predictions_csv_path.parent}/{predictions_csv_path.stem}_assignments.csv"
    )
    assignments_df.to_csv(assignments_csv_path, sep="\t", index=False)
    logger.info(f"assignments CSV saved at {assignments_csv_path}")

    num_assignments = len(assignments_df)

    num_equal_assignments = assignments_df["classifier_symbol"].eq(assignments_df["xref_symbol"]).sum()

    matching_percentage = (num_equal_assignments / num_assignments) * 100
    logger.info(f"{num_equal_assignments} matching out of {num_assignments} assignments ({matching_percentage}%)")


if __name__ == "__main__":
    main()
