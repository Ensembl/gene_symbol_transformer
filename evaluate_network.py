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
Evaluate trained network by comparing the classifier assigned symbols to
the Xref assigned ones.
"""


# standard library imports
import argparse
import csv
import gzip
import pathlib
import pprint
import sys

# third party imports
import pandas as pd
import pymysql
import requests
import yaml

from loguru import logger

# project imports
from fully_connected_pipeline import FullyConnectedNetwork
from pipeline_abstractions import PrettySimpleNamespace, load_checkpoint, read_fasta_in_chunks


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

sequences_directory = pathlib.Path("protein_sequences")
sequences_directory.mkdir(exist_ok=True)


def parse_fasta_description(fasta_description):
    """
    Parse a FASTA entry description string, generating an object with a number
    of attributes if they exist.
    """
    fields = ["gene", "transcript", "gene_biotype", "transcript_biotype", "gene_symbol"]
    description = PrettySimpleNamespace()

    description_parts = fasta_description.split()

    description.translation_stable_id = description_parts[0]

    for description_part in description_parts:
        if ":" in description_part:
            elements = description_part.split(":")
            key = elements[0]
            value = elements[1]
            if key in fields:
                setattr(description, key, value)

    return description


def compare_with_fasta(assignments_csv, sequences_fasta):
    """
    Compare classifier assignments with the gene symbols in the FASTA file.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)
    sequences_fasta_path = pathlib.Path(sequences_fasta)

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            for fasta_entry, csv_row in zip(fasta_entries, csv_reader):
                description = parse_fasta_description(fasta_entry[0])

                if not hasattr(description, "gene_symbol"):
                    continue

                if not hasattr(description, "gene_biotype"):
                    continue

                if description.gene_biotype != "protein_coding":
                    continue

                translation_stable_id = description.translation_stable_id
                transcript_stable_id = description.transcript
                xref_symbol = description.gene_symbol

                csv_stable_id = csv_row[0]
                classifier_symbol = csv_row[1]

                assert (
                    translation_stable_id == csv_stable_id
                ), f"{translation_stable_id=}, {csv_stable_id=}"

                comparisons.append(
                    (
                        translation_stable_id,
                        transcript_stable_id,
                        classifier_symbol,
                        xref_symbol,
                    )
                )

    dataframe_columns = [
        "translation_stable_id",
        "transcript_stable_id",
        "classifier_symbol",
        "xref_symbol",
    ]
    comparisons_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare_fasta.csv"
    )
    comparisons_df.to_csv(comparisons_csv_path, sep="\t", index=False)
    logger.info(f"comparisons CSV saved at {comparisons_csv_path}")

    num_assignments = len(comparisons_df)

    comparisons_df["classifier_symbol_lowercase"] = comparisons_df[
        "classifier_symbol"
    ].str.lower()
    comparisons_df["xref_symbol_lowercase"] = comparisons_df["xref_symbol"].str.lower()

    num_equal_assignments = (
        comparisons_df["classifier_symbol_lowercase"]
        .eq(comparisons_df["xref_symbol_lowercase"])
        .sum()
    )

    matching_percentage = (num_equal_assignments / num_assignments) * 100
    logger.info(
        f"{num_equal_assignments} matching out of {num_assignments} assignments ({matching_percentage:.2f}%)"
    )


def compare_with_database(assignments_csv, ensembldb_species_database):
    """
    Compare classifier assignments with the gene symbols in the species database on
    the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    host = "ensembldb.ensembl.org"
    user = "anonymous"
    connection = pymysql.connect(
        host=host,
        user=user,
        database=ensembldb_species_database,
        # cursorclass=pymysql.cursors.DictCursor,
    )

    # read SQL query from sql file
    sql_query_filepath = "sql_query.sql"
    with open(sql_query_filepath) as f:
        sql_query = f.read()

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(sql_query)
            db_response = cursor.fetchall()

    db_response_dict = dict(db_response)

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]

            translation_stable_id = csv_stable_id[:-2]

            if translation_stable_id in db_response_dict:
                xref_symbol = db_response_dict[translation_stable_id]
                comparisons.append((csv_stable_id, classifier_symbol, xref_symbol))

    dataframe_columns = [
        "csv_stable_id",
        "classifier_symbol",
        "xref_symbol",
    ]
    comparisons_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare_database.csv"
    )
    comparisons_df.to_csv(comparisons_csv_path, sep="\t", index=False)
    logger.info(f"comparisons CSV saved at {comparisons_csv_path}")

    num_assignments = len(comparisons_df)

    comparisons_df["classifier_symbol_lowercase"] = comparisons_df[
        "classifier_symbol"
    ].str.lower()
    comparisons_df["xref_symbol_lowercase"] = comparisons_df["xref_symbol"].str.lower()

    num_equal_assignments = (
        comparisons_df["classifier_symbol_lowercase"]
        .eq(comparisons_df["xref_symbol_lowercase"])
        .sum()
    )

    matching_percentage = (num_equal_assignments / num_assignments) * 100
    logger.info(
        f"{num_equal_assignments} matching out of {num_assignments} assignments ({matching_percentage:.2f}%)"
    )


def evaluate_network(checkpoint_path, species_data_path):
    """
    """
    with open(species_data_path) as f:
        species_data_list = yaml.safe_load(f)

    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]
    training_session = checkpoint["training_session"]

    for species_data_dict in species_data_list:
        species_data = PrettySimpleNamespace(**species_data_dict)

        # download archived protein sequences FASTA file
        archived_fasta_filename = species_data.protein_sequences.split("/")[-1]
        archived_fasta_path = sequences_directory / archived_fasta_filename
        if not archived_fasta_path.exists():
            response = requests.get(species_data.protein_sequences)
            with open(archived_fasta_path, "wb+") as f:
                f.write(response.content)
            logger.info(f"downloaded {archived_fasta_filename}")

        # extract archived protein sequences FASTA file
        fasta_path = archived_fasta_path.with_suffix("")
        if not fasta_path.exists():
            with gzip.open(archived_fasta_path, "rb") as f:
                file_content = f.read()
            with open(fasta_path, "wb+") as f:
                f.write(file_content)
            logger.info(f"extracted {fasta_path}")

        # assign symbols
        assignments_csv_path = pathlib.Path(f"{fasta_path.parent}/{fasta_path.stem}_symbols.csv")
        if not assignments_csv_path.exists():
            assign_symbols(network, fasta_path)

        compare_with_database(assignments_csv_path, species_data.ensembldb_database)


def assign_symbols(network, sequences_fasta):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    fasta_path = pathlib.Path(sequences_fasta)
    assignments_csv_path = pathlib.Path(f"{fasta_path.parent}/{fasta_path.stem}_symbols.csv")

    # read the FASTA file in chunks and assign symbols
    with open(assignments_csv_path, "w+") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol"]
        csv_writer = csv.writer(csv_file, delimiter="\t")
        csv_writer.writerow(field_names)

        for fasta_entries in read_fasta_in_chunks(fasta_path):
            if fasta_entries[-1] is None:
                fasta_entries = [
                    fasta_entry
                    for fasta_entry in fasta_entries if fasta_entry is not None
                ]

            stable_ids = [
                fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries
            ]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]

            assignments = network.predict(sequences)

            # save assignments to the CSV file
            csv_writer.writerows(zip(stable_ids, assignments))
    logger.info(f"symbol assignments saved at {assignments_csv_path}")


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--assignments_csv", help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--sequences_fasta",
        help="protein sequences FASTA file path that includes gene symbols to compare the classifier assignments to",
    )
    argument_parser.add_argument(
        "--ensembldb_species_database",
        help="species database name on the public Ensembl MySQL server",
    )
    argument_parser.add_argument("--checkpoint", help="training session checkpoint path")
    argument_parser.add_argument("--species_data", help="species data YAML file path")

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)
    if args.assignments_csv:
        assignments_csv_path = pathlib.Path(args.assignments_csv)
        log_file_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
        )
        logger.add(log_file_path, format=LOGURU_FORMAT)
    elif args.species_data:
        species_data_path = pathlib.Path(args.species_data)
        log_file_path = pathlib.Path(
            f"{species_data_path.parent}/{species_data_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=LOGURU_FORMAT)
    else:
        print("Error: one of --assignments_csv or --species_data arguments is required:\n")
        argument_parser.print_help()
        sys.exit()

    if args.sequences_fasta:
        compare_with_fasta(args.assignments_csv, args.sequences_fasta)
    elif args.ensembldb_species_database:
        compare_with_database(args.assignments_csv, args.ensembldb_species_database)
    elif args.checkpoint and args.species_data:
        evaluate_network(args.checkpoint, args.species_data)
    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    main()
