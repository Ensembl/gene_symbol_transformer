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

Pass the --assignments_csv and --ensembl_database arguments to compare
the assignments in the `assignments_csv` CSV file with the ones in the `ensembl_database`
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

from loguru import logger

# project imports
from dataset_generation import (
    download_protein_sequences_fasta,
    get_assemblies_metadata,
    get_canonical_translations,
    get_ensembl_release,
)
from gene_symbol_classifier import (
    EarlyStopping,
    FullyConnectedNetwork,
    TrainingSession,
    assign_symbols,
)
from utils import get_clade, load_checkpoint


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

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
    "varanus_komodoensis": "komodo dragon",
}


def evaluate_network(checkpoint_path, complete=False):
    """
    Evaluate a trained network by assigning gene symbols to the protein sequences
    of genome assemblies in the latest Ensembl release, and comparing them to the existing
    Xref assignments.

    Args:
        checkpoint_path (Path): path to the training checkpoint
        complete (bool): Whether or not to run the evaluation for all genome assemblies.
            Defaults to False, which runs the evaluation only for a selection of
            the most important species genome assemblies.
    """
    network, _training_session = load_checkpoint(checkpoint_path)

    ensembl_release = get_ensembl_release()
    logger.info(f"Ensembl release {ensembl_release}")

    assemblies = get_assemblies_metadata()
    for assembly in assemblies:
        if not complete and assembly.species not in selected_species_genomes:
            continue

        fasta_path = download_protein_sequences_fasta(assembly, ensembl_release)

        # get the Genebuild defined clade for the species
        clade = get_clade(assembly.taxonomy_id)

        # assign symbols
        assignments_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {fasta_path}")
            assign_symbols(network, fasta_path, clade, checkpoint_path.parent)

        comparisons_csv_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
        )
        if not comparisons_csv_path.exists():
            compare_with_database(
                assignments_csv_path,
                assembly.core_db,
                assembly.species,
            )


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
    ensembl_database,
    scientific_name=None,
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Compare classifier assignments with the gene symbols in the genome assembly
    ensembl_database core database on the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    canonical_translations = get_canonical_translations(
        ensembl_database, EntrezGene, Uniprot_gn
    )

    comparisons = []
    with open(assignments_csv_path, "r") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        _csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]

            translation_stable_id = csv_stable_id.split(".")[0]

            if (
                translation_stable_id
                in canonical_translations["translation.stable_id"].values
            ):
                xref_symbol = canonical_translations.loc[
                    canonical_translations["translation.stable_id"]
                    == translation_stable_id,
                    "Xref_symbol",
                ].values[0]
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
        "--ensembl_database",
        help="genome assembly core database name on the public Ensembl MySQL server",
    )
    argument_parser.add_argument("--checkpoint", help="training session checkpoint path")
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run the evaluation for all genome assemblies in the Ensembl release",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)

    if args.assignments_csv and args.ensembl_database:
        assignments_csv_path = pathlib.Path(args.assignments_csv)
        log_file_path = pathlib.Path(
            f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
        )

        logger.add(log_file_path, format=LOGURU_FORMAT)
        compare_with_database(args.assignments_csv, args.ensembl_database)
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
