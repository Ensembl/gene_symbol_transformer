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
Evaluate a trained network or get statistics for existing symbol assignments.

Pass the --assignments_csv and --ensembldb_database arguments to compare
the assignments in the `assignments_csv` CSV file with the ones in the `ensembldb_database`
Ensembl database,
or
pass the --checkpoint argument to evaluate a trained network by assigning symbols
to the canonical translations of protein sequences of annotations in the latest
Ensembl release and comparing them to the existing symbol assignments.

A trained network can be evaluated by assigning gene symbols to the assembly annotations on the main Ensembl website and comparing them with the existing gene symbols.
"""


# standard library imports
import argparse
import csv
import pathlib
import sys

# third party imports
import pandas as pd

from icecream import ic
from loguru import logger

# project imports
from dataset_generation import (
    download_protein_sequences_fasta,
    get_canonical_translations,
    get_genomes_metadata,
)
from fully_connected_pipeline import FullyConnectedNetwork
from pipeline_abstractions import load_checkpoint, read_fasta_in_chunks


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"


def evaluate_network(checkpoint_path, complete=False):
    """
    Evaluate a trained network by assigning gene symbols to the protein sequences
    of genomes in the latest Ensembl release, and comparing them to the existing
    Xref assignments.

    Args:
        checkpoint_path (Path): path to the training checkpoint
        complete (bool): Whether or not to run the evaluation for all genomes.
            Defaults to False, which runs the evaluation only for a selection of
            the most important species genomes.
    """
    selected_species_genomes = {
        "ailuropoda_melanoleuca": "giant panda",
        "aquila_chrysaetos_chrysaetos": "golden eagle",
        "balaenoptera_musculus": "blue whale",
        "bos_taurus": "cow",
        "caenorhabditis_elegans": "caenorhabditis elegans",
        "canis_lupus_familiaris": "dog",
        "cyprinus_carpio": "common carp",
        "danio_rerio": "zebrafish",
        "drosophila_melanogaster": "drosophila melanogaster",
        "felis_catus": "cat",
        "gallus_gallus": "chicken",
        "homo_sapiens": "human",
        "loxodonta_africana": "elephant",
        "mus_musculus": "mouse",
        "oryctolagus_cuniculus": "rabbit",
        "ovis_aries": "sheep",
        "pan_troglodytes": "chimpanzee",
        "panthera_leo": "lion",
        "saccharomyces_cerevisiae": "saccharomyces cerevisiae",
        "sus_scrofa": "pig",
        "tursiops_truncatus": "dolphin",
        "varanus_komodoensis": "komodo dragon",
    }

    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]
    training_session = checkpoint["training_session"]

    genomes = get_genomes_metadata()
    for genome in genomes:
        if not complete and genome.species not in selected_species_genomes:
            continue

        download_protein_sequences_fasta(genome)

        # assign symbols
        assignments_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {fasta_path}")
            assign_symbols(network, checkpoint_path, fasta_path)

        comparisons_csv_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
        )
        if not comparisons_csv_path.exists():
            compare_with_database(
                assignments_csv_path,
                genome.core_db,
                genome.species,
            )


def assign_symbols(network, checkpoint_path, sequences_fasta):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    fasta_path = pathlib.Path(sequences_fasta)
    assignments_csv_path = pathlib.Path(
        f"{checkpoint_path.parent}/{fasta_path.stem}_symbols.csv"
    )

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
                    for fasta_entry in fasta_entries
                    if fasta_entry is not None
                ]

            stable_ids = [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]

            assignments = network.predict(sequences)

            # save assignments to the CSV file
            csv_writer.writerows(zip(stable_ids, assignments))
    logger.info(f"symbol assignments saved at {assignments_csv_path}")


def are_strict_subsets(symbol_a, symbol_b):
    symbol_a = symbol_a.lower()
    symbol_b = symbol_b.lower()

    if symbol_a == symbol_b:
        return False

    if (symbol_a in symbol_b) or (symbol_b in symbol_a):
        return True
    else:
        return False


def compare_with_database(
    assignments_csv,
    ensembldb_database,
    scientific_name=None,
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Compare classifier assignments with the gene symbols in the genome database on
    the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    canonical_translations = get_canonical_translations(
        ensembldb_database, EntrezGene, Uniprot_gn
    )

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]

            translation_stable_id = csv_stable_id[:-2]

            if translation_stable_id in canonical_translations.index:
                xref_symbol = canonical_translations.loc[translation_stable_id]["Xref_symbol"]
                comparisons.append((csv_stable_id, classifier_symbol, xref_symbol))

    dataframe_columns = [
        "csv_stable_id",
        "classifier_symbol",
        "xref_symbol",
    ]
    compare_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    num_assignments = len(compare_df)

    num_exact_matches = (
        compare_df["classifier_symbol"]
        .str.lower()
        .eq(compare_df["xref_symbol"].str.lower())
        .sum()
    )

    compare_df["strict_subsets"] = compare_df.apply(
        lambda x: are_strict_subsets(x["classifier_symbol"], x["xref_symbol"]),
        axis=1,
        result_type="reduce",
    )

    fuzzy_matches = compare_df.loc[
        compare_df["strict_subsets"] == True, ["classifier_symbol", "xref_symbol"]
    ]
    fuzzy_matches_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_fuzzy_matches.csv"
    )
    fuzzy_matches.to_csv(fuzzy_matches_csv_path, sep="\t", index=False)

    num_fuzzy_matches = compare_df["strict_subsets"].sum()

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    compare_df.to_csv(comparisons_csv_path, sep="\t", index=False)

    matching_percentage = (num_exact_matches / num_assignments) * 100
    fuzzy_percentage = (num_fuzzy_matches / num_assignments) * 100
    total_matches_percentage = (
        (num_exact_matches + num_fuzzy_matches) / num_assignments
    ) * 100
    if scientific_name is not None:
        message = f"{scientific_name}: "
    else:
        message = ""
    message += "{} assignments, {} exact matches ({:.2f}%), {} fuzzy matches ({:.2f}%), {} total matches ({:.2f}%)".format(
        num_assignments,
        num_exact_matches,
        matching_percentage,
        num_fuzzy_matches,
        fuzzy_percentage,
        num_exact_matches + num_fuzzy_matches,
        total_matches_percentage,
    )
    logger.info(message)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--assignments_csv",
        help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--ensembldb_database",
        help="genome database name on the public Ensembl MySQL server",
    )
    argument_parser.add_argument("--checkpoint", help="training session checkpoint path")
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run the evaluation for all genomes in the Ensembl release",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)

    if args.assignments_csv and args.ensembldb_database:
        assignments_csv_path = pathlib.Path(args.assignments_csv)
        log_file_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
        )

        logger.add(log_file_path, format=LOGURU_FORMAT)
        compare_with_database(args.assignments_csv, args.ensembldb_database)
    elif args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        log_file_path = pathlib.Path(
            f"{checkpoint_path.parent}/{checkpoint_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=LOGURU_FORMAT)

        evaluate_network(checkpoint_path, args.complete)
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
