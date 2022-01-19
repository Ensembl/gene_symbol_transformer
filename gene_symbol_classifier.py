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
Train, test, evaluate, and use a gene symbol classifier to assign gene symbols
to protein sequences.

Evaluate a trained network
A trained network, specified with the `--checkpoint` argument with its path,
is evaluated by assigning symbols to the canonical translations of protein sequences
of annotations in the latest Ensembl release and comparing them to the existing
symbol assignments.

Get statistics for existing symbol assignments
Gene symbol assignments from a classifier can be compared against the existing
assignments in the Ensembl database, by specifying the path to the assignments CSV file
with `--assignments_csv` and the Ensembl database name with `--ensembl_database`.
"""


# standard library imports
import argparse
import csv
import datetime as dt
import json
import logging
import pathlib
import random
import sys
import warnings

# third party imports
import pandas as pd
import pytorch_lightning as pl
import torch
import yaml

from torch.utils.data import DataLoader, random_split

# project imports
from utils import (
    AttributeDict,
    GeneSymbolClassifier,
    SequenceDataset,
    add_log_file_handler,
    data_directory,
    get_assemblies_metadata,
    get_species_taxonomy_id,
    get_taxonomy_id_clade,
    get_xref_canonical_translations,
    load_checkpoint,
    log_pytorch_cuda_info,
    logger,
    logging_formatter_time_message,
    read_fasta_in_chunks,
    sequences_directory,
)


selected_genome_assemblies = {
    "GCA_002007445.2": ("Ailuropoda melanoleuca", "Giant panda"),
    "GCA_900496995.2": ("Aquila chrysaetos chrysaetos", "Golden eagle"),
    "GCA_009873245.2": ("Balaenoptera musculus", "Blue whale"),
    "GCA_002263795.2": ("Bos taurus", "Cow"),
    "GCA_000002285.2": ("Canis lupus familiaris", "Dog"),
    "GCA_000951615.2": ("Cyprinus carpio", "Common carp"),
    "GCA_000002035.4": ("Danio rerio", "Zebrafish"),
    "GCA_000001215.4": ("Drosophila melanogaster", "Drosophila melanogaster"),
    "GCA_000181335.4": ("Felis catus", "Cat"),
    "GCA_000002315.5": ("Gallus gallus", "Chicken"),
    "GCA_000001405.28": ("Homo sapiens", "Human"),
    "GCA_000001905.1": ("Loxodonta africana", "Elephant"),
    "GCA_000001635.9": ("Mus musculus", "Mouse"),
    "GCA_000003625.1": ("Oryctolagus cuniculus", "Rabbit"),
    "GCA_002742125.1": ("Ovis aries", "Sheep"),
    "GCA_000001515.5": ("Pan troglodytes", "Chimpanzee"),
    "GCA_008795835.1": ("Panthera leo", "Lion"),
    "GCA_000146045.2": ("Saccharomyces cerevisiae", "Saccharomyces cerevisiae"),
    "GCA_000003025.6": ("Sus scrofa", "Pig"),
}


def generate_dataloaders(experiment):
    """
    Generate training, validation, and test dataloaders from the dataset files.

    Args:
        experiment (Experiment): Experiment object containing metadata
    Returns:
        tuple containing the training, validation, and test dataloaders
    """
    dataset = SequenceDataset(
        num_symbols=experiment.num_symbols,
        sequence_length=experiment.sequence_length,
        padding_side=experiment.padding_side,
        excluded_genera=experiment.excluded_genera,
    )

    experiment.symbol_mapper = dataset.symbol_mapper
    experiment.protein_sequence_mapper = dataset.protein_sequence_mapper
    experiment.clade_mapper = dataset.clade_mapper

    experiment.num_protein_letters = (
        experiment.protein_sequence_mapper.num_protein_letters
    )
    experiment.num_clades = experiment.clade_mapper.num_categories

    logger.info(
        "gene symbols:\n{}".format(pd.Series(experiment.symbol_mapper.categories))
    )

    # calculate the training, validation, and test set size
    dataset_size = len(dataset)
    experiment.validation_size = int(experiment.validation_ratio * dataset_size)
    experiment.test_size = int(experiment.test_ratio * dataset_size)
    experiment.training_size = (
        dataset_size - experiment.validation_size - experiment.test_size
    )

    # split dataset into training, validation, and test datasets
    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        lengths=(
            experiment.training_size,
            experiment.validation_size,
            experiment.test_size,
        ),
        generator=torch.Generator().manual_seed(experiment.random_seed),
    )

    logger.info(
        f"dataset split to training ({experiment.training_size}), validation ({experiment.validation_size}), and test ({experiment.test_size}) datasets"
    )

    # set the batch size equal to the size of the smallest dataset if larger than that
    experiment.batch_size = min(
        experiment.batch_size,
        experiment.training_size,
        experiment.validation_size,
        experiment.test_size,
    )

    training_loader = DataLoader(
        training_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=experiment.num_workers,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=experiment.batch_size,
        num_workers=experiment.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment.batch_size,
        num_workers=experiment.num_workers,
    )

    return (training_loader, validation_loader, test_loader)


def assign_symbols(
    network,
    symbols_metadata,
    sequences_fasta,
    scientific_name=None,
    taxonomy_id=None,
    output_directory=None,
):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    sequences_fasta_path = pathlib.Path(sequences_fasta)

    if scientific_name is not None:
        taxonomy_id = get_species_taxonomy_id(scientific_name)
    clade = get_taxonomy_id_clade(taxonomy_id)
    # logger.info(f"got clade {clade} for {scientific_name}")

    if output_directory is None:
        output_directory = sequences_fasta_path.parent
    assignments_csv_path = pathlib.Path(
        f"{output_directory}/{sequences_fasta_path.stem}_symbols.csv"
    )

    # read the FASTA file in chunks and assign symbols
    with open(assignments_csv_path, "w+", newline="") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol", "probability", "description", "source"]
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(field_names)

        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            if fasta_entries[-1] is None:
                fasta_entries = [
                    fasta_entry
                    for fasta_entry in fasta_entries
                    if fasta_entry is not None
                ]

            identifiers = [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]
            clades = [clade for _ in range(len(fasta_entries))]

            assignments_probabilities = network.predict_probabilities(sequences, clades)
            # save assignments and probabilities to the CSV file
            for identifier, (assignment, probability) in zip(
                identifiers, assignments_probabilities
            ):
                symbol_description = symbols_metadata[assignment]["description"]
                symbol_source = symbols_metadata[assignment]["source"]
                csv_writer.writerow(
                    [
                        identifier,
                        assignment,
                        probability,
                        symbol_description,
                        symbol_source,
                    ]
                )

    logger.info(f"symbol assignments saved at {assignments_csv_path}")


def save_network_from_checkpoint(checkpoint_path):
    """
    Save the network in a checkpoint file as a separate file.
    """
    _experiment, network, _optimizer, _symbols_metadata = load_checkpoint(checkpoint_path)

    path = checkpoint_path
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")

    torch.save(network, network_path)

    return network_path


def evaluate_network(checkpoint_path, complete=False):
    """
    Evaluate a trained network by assigning gene symbols to the protein sequences
    of genome assemblies in the latest Ensembl release, and comparing them to the existing
    Xref assignments.

    Args:
        checkpoint_path (Path): path to the experiment checkpoint
        complete (bool): Whether or not to run the evaluation for all genome assemblies.
            Defaults to False, which runs the evaluation only for a selection of
            the most important species genome assemblies.
    """
    evaluation_directory_path = (
        checkpoint_path.parent / f"{checkpoint_path.stem}_evaluation"
    )

    experiment, network, _optimizer, symbols_metadata = load_checkpoint(checkpoint_path)
    symbols_set = set(symbol.lower() for symbol in experiment.symbol_mapper.categories)

    assemblies = get_assemblies_metadata()
    comparison_statistics_list = []
    for assembly in assemblies:
        if not complete and assembly.assembly_accession not in selected_genome_assemblies:
            continue

        canonical_fasta_filename = assembly.fasta_filename.replace(
            "pep.all.fa", "pep.all_canonical.fa"
        )
        canonical_fasta_path = sequences_directory / canonical_fasta_filename

        # assign symbols
        assignments_csv_path = (
            evaluation_directory_path / f"{canonical_fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {canonical_fasta_path}")
            assign_symbols(
                network,
                symbols_metadata,
                canonical_fasta_path,
                scientific_name=assembly.scientific_name,
                output_directory=evaluation_directory_path,
            )

        comparisons_csv_path = (
            evaluation_directory_path / f"{assignments_csv_path.stem}_compare.csv"
        )
        if not comparisons_csv_path.exists():
            comparison_successful = compare_with_database(
                assignments_csv_path,
                assembly.core_db,
                assembly.scientific_name,
                symbols_set,
            )
            if not comparison_successful:
                continue

        comparison_statistics = get_comparison_statistics(comparisons_csv_path)
        comparison_statistics["scientific_name"] = assembly.scientific_name
        comparison_statistics["taxonomy_id"] = assembly.taxonomy_id
        comparison_statistics["clade"] = assembly.clade

        comparison_statistics_list.append(comparison_statistics)

        message = "{}: {} assignments, {} exact matches ({:.2f}%), {} fuzzy matches ({:.2f}%), {} total matches ({:.2f}%)".format(
            comparison_statistics["scientific_name"],
            comparison_statistics["num_assignments"],
            comparison_statistics["num_exact_matches"],
            comparison_statistics["matching_percentage"],
            comparison_statistics["num_fuzzy_matches"],
            comparison_statistics["fuzzy_percentage"],
            comparison_statistics["num_total_matches"],
            comparison_statistics["total_matches_percentage"],
        )
        logger.info(message)

    dataframe_columns = [
        "clade",
        "scientific_name",
        "num_assignments",
        "num_exact_matches",
        "matching_percentage",
        "num_fuzzy_matches",
        "fuzzy_percentage",
        "num_total_matches",
        "total_matches_percentage",
    ]
    comparison_statistics = pd.DataFrame(
        comparison_statistics_list,
        columns=dataframe_columns,
    )

    clade_groups = comparison_statistics.groupby(["clade"])
    clade_groups_statistics = []
    aggregated_statistics = []
    for clade, group in clade_groups:
        with pd.option_context("display.float_format", "{:.2f}".format):
            group_string = group.to_string(index=False)

        num_assignments_sum = group["num_assignments"].sum()
        num_exact_matches_sum = group["num_exact_matches"].sum()
        num_fuzzy_matches_sum = group["num_fuzzy_matches"].sum()
        num_total_matches_sum = num_exact_matches_sum + num_fuzzy_matches_sum

        matching_percentage_weighted_average = (
            num_exact_matches_sum / num_assignments_sum
        ) * 100
        fuzzy_percentage_weighted_average = (
            num_fuzzy_matches_sum / num_assignments_sum
        ) * 100
        total_percentage_weighted_average = (
            num_total_matches_sum / num_assignments_sum
        ) * 100

        averages_message = "{} weighted averages: {:.2f}% exact matches, {:.2f}% fuzzy matches, {:.2f}% total matches".format(
            clade,
            matching_percentage_weighted_average,
            fuzzy_percentage_weighted_average,
            total_percentage_weighted_average,
        )

        aggregated_statistics.append(
            {
                "clade": clade,
                "exact matches": matching_percentage_weighted_average,
                "fuzzy matches": fuzzy_percentage_weighted_average,
                "total matches": total_percentage_weighted_average,
            }
        )

        clade_statistics = f"{group_string}\n{averages_message}"

        clade_groups_statistics.append(clade_statistics)

    comparison_statistics_string = "comparison statistics:\n"
    comparison_statistics_string += "\n\n".join(
        clade_statistics for clade_statistics in clade_groups_statistics
    )
    logger.info(comparison_statistics_string)

    aggregated_statistics = pd.DataFrame(aggregated_statistics)
    logger.info(f"\n\n{aggregated_statistics.to_string(index=False)}")


def is_exact_match(symbol_a, symbol_b):
    symbol_a = symbol_a.lower()
    symbol_b = symbol_b.lower()

    if symbol_a == symbol_b:
        return "exact_match"
    else:
        return "no_exact_match"


def is_fuzzy_match(symbol_a, symbol_b):
    symbol_a = symbol_a.lower()
    symbol_b = symbol_b.lower()

    if symbol_a == symbol_b:
        return "no_fuzzy_match"

    if (symbol_a in symbol_b) or (symbol_b in symbol_a):
        return "fuzzy_match"
    else:
        return "no_fuzzy_match"


def is_known_symbol(symbol, symbols_set):
    symbol = symbol.lower()

    if symbol in symbols_set:
        return "known"
    else:
        return "unknown"


def compare_with_database(
    assignments_csv,
    ensembl_database,
    scientific_name=None,
    symbols_set=None,
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Compare classifier assignments with the gene symbols in the genome assembly
    ensembl_database core database on the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    canonical_translations = get_xref_canonical_translations(
        ensembl_database, EntrezGene=EntrezGene, Uniprot_gn=Uniprot_gn
    )

    if len(canonical_translations) == 0:
        if scientific_name is None:
            logger.info("0 canonical translations retrieved, nothing to compare")
        else:
            logger.info(
                f"{scientific_name}: 0 canonical translations retrieved, nothing to compare"
            )
        return False

    comparisons = []
    with open(assignments_csv_path, "r", newline="") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        _csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]
            probability = csv_row[2]

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
                comparisons.append(
                    (csv_stable_id, xref_symbol, classifier_symbol, probability)
                )

    dataframe_columns = [
        "csv_stable_id",
        "xref_symbol",
        "classifier_symbol",
        "probability",
    ]
    compare_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    compare_df["exact_match"] = compare_df.apply(
        lambda x: is_exact_match(x["classifier_symbol"], x["xref_symbol"]),
        axis=1,
        result_type="reduce",
    )

    compare_df["fuzzy_match"] = compare_df.apply(
        lambda x: is_fuzzy_match(x["classifier_symbol"], x["xref_symbol"]),
        axis=1,
        result_type="reduce",
    )

    if symbols_set:
        compare_df["known_symbol"] = compare_df.apply(
            lambda x: is_known_symbol(x["xref_symbol"], symbols_set),
            axis=1,
            result_type="reduce",
        )

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    compare_df.to_csv(comparisons_csv_path, sep="\t", index=False)

    return True


def get_comparison_statistics(comparisons_csv_path):
    compare_df = pd.read_csv(comparisons_csv_path, sep="\t", index_col=False)

    num_assignments = len(compare_df)

    if num_assignments > 0:
        num_exact_matches = len(compare_df[compare_df["exact_match"] == "exact_match"])
        num_fuzzy_matches = len(compare_df[compare_df["fuzzy_match"] == "fuzzy_match"])

        matching_percentage = (num_exact_matches / num_assignments) * 100
        fuzzy_percentage = (num_fuzzy_matches / num_assignments) * 100
        num_total_matches = num_exact_matches + num_fuzzy_matches
        total_matches_percentage = (num_total_matches / num_assignments) * 100

        comparison_statistics = {
            "num_assignments": num_assignments,
            "num_exact_matches": num_exact_matches,
            "matching_percentage": matching_percentage,
            "num_fuzzy_matches": num_fuzzy_matches,
            "fuzzy_percentage": fuzzy_percentage,
            "num_total_matches": num_total_matches,
            "total_matches_percentage": total_matches_percentage,
        }
    else:
        comparison_statistics = {
            "num_assignments": 0,
            "num_exact_matches": 0,
            "matching_percentage": 0,
            "num_fuzzy_matches": 0,
            "fuzzy_percentage": 0,
            "num_total_matches": 0,
            "total_matches_percentage": 0,
        }

    return comparison_statistics


def compare_assignments(
    assignments_csv, ensembl_database, scientific_name, checkpoint=None
):
    """Compare assignments with the ones on the latest Ensembl release."""
    assignments_csv_path = pathlib.Path(assignments_csv)

    log_file_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
    )
    add_log_file_handler(logger, log_file_path)

    if checkpoint is None:
        symbols_set = None
    else:
        experiment, _network, _optimizer, _symbols_metadata = load_checkpoint(checkpoint)
        symbols_set = set(
            symbol.lower() for symbol in experiment.symbol_mapper.categories
        )

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    if not comparisons_csv_path.exists():
        compare_with_database(
            assignments_csv_path, ensembl_database, scientific_name, symbols_set
        )

    comparison_statistics = get_comparison_statistics(comparisons_csv_path)

    taxonomy_id = get_species_taxonomy_id(scientific_name)
    clade = get_taxonomy_id_clade(taxonomy_id)

    comparison_statistics["scientific_name"] = scientific_name
    comparison_statistics["taxonomy_id"] = taxonomy_id
    comparison_statistics["clade"] = clade

    message = "{} assignments, {} exact matches ({:.2f}%), {} fuzzy matches ({:.2f}%), {} total matches ({:.2f}%)".format(
        comparison_statistics["num_assignments"],
        comparison_statistics["num_exact_matches"],
        comparison_statistics["matching_percentage"],
        comparison_statistics["num_fuzzy_matches"],
        comparison_statistics["fuzzy_percentage"],
        comparison_statistics["num_total_matches"],
        comparison_statistics["total_matches_percentage"],
    )
    logger.info(message)

    dataframe_columns = [
        "clade",
        "scientific_name",
        "num_assignments",
        "num_exact_matches",
        "matching_percentage",
        "num_fuzzy_matches",
        "fuzzy_percentage",
        "num_total_matches",
        "total_matches_percentage",
    ]
    comparison_statistics = pd.DataFrame(
        [comparison_statistics],
        columns=dataframe_columns,
    )
    with pd.option_context("display.float_format", "{:.2f}".format):
        logger.info(
            f"comparison statistics:\n{comparison_statistics.to_string(index=False)}"
        )


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--datetime",
        help="datetime string; if set this will be used instead of generating a new one",
    )
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration file",
    )
    argument_parser.add_argument(
        "--checkpoint",
        help="experiment checkpoint path",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")
    argument_parser.add_argument(
        "--sequences_fasta",
        help="path of FASTA file with protein sequences to assign symbols to",
    )
    argument_parser.add_argument(
        "--scientific_name",
        help="scientific name of the species the protein sequences belong to",
    )
    argument_parser.add_argument(
        "--save_network",
        action="store_true",
        help="save the network in a checkpoint file as a separate file",
    )
    argument_parser.add_argument(
        "--evaluate", action="store_true", help="evaluate a classifier"
    )
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run the evaluation for all genome assemblies in the Ensembl release",
    )
    argument_parser.add_argument(
        "--assignments_csv",
        help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--ensembl_database",
        help="genome assembly core database name on the public Ensembl MySQL server",
    )

    args = argument_parser.parse_args()

    # filter warning about number of dataloader workers
    warnings.filterwarnings(
        "ignore",
        ".*does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument.*",
    )

    # train a new classifier
    if args.train and args.configuration:
        # read the experiment configuration YAML file to a dictionary
        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)

        configuration = AttributeDict(configuration)

        configuration.datetime = configuration.get(
            "datetime", dt.datetime.now().isoformat(sep="_", timespec="seconds")
        )

        configuration.logging_version = f"{configuration.experiment_prefix}_ns{configuration.num_symbols}_{configuration.datetime}"

        # generate random seed if it doesn't exist
        # Using the range [1_000_000, 1_001_000] for the random seed. This range contains
        # numbers that have a good balance of 0 and 1 bits, as recommended by the PyTorch docs.
        # https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed
        configuration.random_seed = configuration.get(
            "random_seed", random.randint(1_000_000, 1_001_000)
        )

        configuration.feature_encoding = "label"

        configuration.experiment_directory = (
            f"{configuration.save_directory}/{configuration.logging_version}"
        )
        log_directory_path = pathlib.Path(configuration.experiment_directory)
        log_directory_path.mkdir(parents=True, exist_ok=True)

        log_file_path = log_directory_path / "experiment.log"
        add_log_file_handler(logger, log_file_path)

        log_pytorch_cuda_info()

        # get training, validation, and test dataloaders
        (
            training_dataloader,
            validation_dataloader,
            test_dataloader,
        ) = generate_dataloaders(configuration)

        # load symbols metadata
        symbols_metadata_filename = "symbols_metadata.json"
        symbols_metadata_path = data_directory / symbols_metadata_filename
        with open(symbols_metadata_path) as file:
            symbols_metadata = json.load(file)

        # instantiate neural network
        network = GeneSymbolClassifier(**configuration)

        # don't use a per-experiment subdirectory
        logging_name = ""

        tensorboard_logger = pl.loggers.TensorBoardLogger(
            save_dir=configuration.save_directory,
            name=logging_name,
            version=configuration.logging_version,
            default_hp_metric=False,
        )

        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="validation_loss",
            min_delta=configuration.loss_delta,
            patience=configuration.patience,
            verbose=True,
        )

        trainer = pl.Trainer(
            gpus=configuration.gpus,
            logger=tensorboard_logger,
            max_epochs=configuration.max_epochs,
            log_every_n_steps=1,
            callbacks=[early_stopping_callback],
            profiler=configuration.profiler,
        )

        trainer.fit(
            model=network,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )

        if args.test:
            trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # resume training and/or test a classifier
    elif (args.train or args.test) and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = f"{checkpoint_path.parent}/experiment.log"
        add_log_file_handler(logger, log_file_path)

        # resume training classifier
        if args.train:
            logger.info("resume training classifier")
            experiment, network, optimizer, symbols_metadata = load_checkpoint(
                checkpoint_path
            )

            logger.info(f"experiment:\n{experiment}")
            logger.info(f"network:\n{network}")

            # get training, validation, and test dataloaders
            training_loader, validation_loader, test_loader = generate_dataloaders(
                experiment
            )

            train_network(
                network,
                optimizer,
                experiment,
                symbols_metadata,
                training_loader,
                validation_loader,
            )

        # test classifier
        if args.test:
            test_network(checkpoint_path, print_sample_assignments=True)

    # evaluate classifier
    elif args.evaluate and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        evaluation_directory_path = (
            checkpoint_path.parent / f"{checkpoint_path.stem}_evaluation"
        )
        evaluation_directory_path.mkdir()
        log_file_path = evaluation_directory_path / f"{checkpoint_path.stem}_evaluate.log"
        add_log_file_handler(logger, log_file_path)

        evaluate_network(checkpoint_path, args.complete)

    # assign symbols to sequences
    elif args.sequences_fasta and args.scientific_name and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = f"{checkpoint_path.parent}/experiment.log"
        add_log_file_handler(logger, log_file_path)

        _experiment, network, _optimizer, symbols_metadata = load_checkpoint(
            checkpoint_path
        )

        logger.info("assigning symbols...")
        assign_symbols(
            network,
            symbols_metadata,
            args.sequences_fasta,
            scientific_name=args.scientific_name,
        )

    # compare assignments with the ones on the latest Ensembl release
    elif args.assignments_csv and args.ensembl_database and args.scientific_name:
        compare_assignments(
            args.assignments_csv,
            args.ensembl_database,
            args.scientific_name,
            args.checkpoint,
        )

    # save a network in a checkpoint as a separate file
    elif args.save_network and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        network_path = save_network_from_checkpoint(checkpoint_path)
        logger.info(f'saved network at "{network_path}"')

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
