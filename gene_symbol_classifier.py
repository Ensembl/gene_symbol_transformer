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
import math
import pathlib
import pprint
import random
import sys
import time

# third party imports
import numpy as np
import pandas as pd
import torch
import torchmetrics
import yaml

from loguru import logger
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from utils import (
    GeneSymbolClassifier,
    SequenceDataset,
    data_directory,
    get_assemblies_metadata,
    get_species_taxonomy_id,
    get_taxonomy_id_clade,
    get_xref_canonical_translations,
    load_checkpoint,
    logging_format,
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


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    """
    Stop training if validation loss doesn't improve during a specified patience period.
    """

    def __init__(self, patience=7, loss_delta=0):
        """
        Args:
            checkpoint_path (path-like object): Path to save the checkpoint.
            patience (int): Number of calls to continue training if validation loss is not improving. Defaults to 7.
            loss_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
        """
        self.patience = patience
        self.loss_delta = loss_delta

        self.no_progress = 0
        self.min_validation_loss = np.Inf

    def __call__(
        self,
        network,
        optimizer,
        experiment,
        symbols_metadata,
        validation_loss,
        checkpoint_path,
    ):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            logger.info("saving first network checkpoint...")
            checkpoint = {
                "experiment": experiment,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "symbols_metadata": symbols_metadata,
            }
            torch.save(checkpoint, checkpoint_path)

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            logger.info(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )

            self.min_validation_loss = validation_loss
            self.no_progress = 0
            checkpoint = {
                "experiment": experiment,
                "network_state_dict": network.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "symbols_metadata": symbols_metadata,
            }
            torch.save(checkpoint, checkpoint_path)

        else:
            self.no_progress += 1

            if self.no_progress == self.patience:
                logger.info(
                    f"{self.no_progress} epochs with no validation loss improvement, stopping training"
                )
                return True

        return False


class Experiment:
    """
    Object containing settings values and status of an experiment.
    """

    def __init__(self, experiment_settings, datetime):
        for attribute, value in experiment_settings.items():
            setattr(self, attribute, value)

        # experiment parameters
        self.datetime = datetime

        # set a seed for the PyTorch random number generator if not present
        if not hasattr(self, "random_seed"):
            self.random_seed = random.randint(1, 100)

        if self.included_genera is not None and self.excluded_genera is not None:
            raise ValueError(
                '"included_genera" and "excluded_genera" are mutually exclusive experiment settings parameters, specify values to at most one of them'
            )

        # early stopping
        loss_delta = 0.001
        self.stop_early = EarlyStopping(self.patience, loss_delta)

        # loss function
        self.criterion = nn.NLLLoss()

        self.num_complete_epochs = 0

        self.filename = f"{self.filename_prefix}_ns{self.num_symbols}_{self.datetime}"

        # self.padding_side = "left"
        self.padding_side = "right"

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


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
        included_genera=experiment.included_genera,
        excluded_genera=experiment.excluded_genera,
    )

    experiment.symbol_mapper = dataset.symbol_mapper
    experiment.protein_sequence_mapper = dataset.protein_sequence_mapper
    experiment.clade_mapper = dataset.clade_mapper

    experiment.num_protein_letters = len(
        experiment.protein_sequence_mapper.protein_letters
    )
    experiment.num_clades = len(experiment.clade_mapper.categories)

    pandas_symbols_categories = experiment.symbol_mapper.categorical_datatype.categories
    logger.info(
        "gene symbols:\n{}".format(
            pandas_symbols_categories.to_series(
                index=range(len(pandas_symbols_categories)), name="gene symbols"
            )
        )
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
        shuffle=True,
        num_workers=experiment.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=experiment.batch_size,
        shuffle=True,
        num_workers=experiment.num_workers,
    )

    return (training_loader, validation_loader, test_loader)


def train_network(
    network,
    optimizer,
    experiment,
    symbols_metadata,
    training_loader,
    validation_loader,
):
    tensorboard_log_dir = f"runs/{experiment.num_symbols}/{experiment.datetime}"
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    max_epochs = experiment.max_epochs
    criterion = experiment.criterion

    checkpoint_path = f"{experiment.experiment_directory}/{experiment.filename}.pth"
    logger.info(f"start training, experiment checkpoints saved at {checkpoint_path}")

    path = pathlib.Path(checkpoint_path)
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")
    torch.save(network, network_path)
    logger.info(f"initial raw network saved at {network_path}")

    max_epochs_length = len(str(max_epochs))

    num_train_batches = math.ceil(experiment.training_size / experiment.batch_size)
    num_batches_length = len(str(num_train_batches))

    if not hasattr(experiment, "average_training_losses"):
        experiment.average_training_losses = []

    if not hasattr(experiment, "average_validation_losses"):
        experiment.average_validation_losses = []

    experiment.epoch = experiment.num_complete_epochs + 1
    epoch_times = []
    for epoch in range(experiment.epoch, max_epochs + 1):
        epoch_start_time = time.time()

        experiment.epoch = epoch

        # training
        ########################################################################
        training_losses = []
        # https://torchmetrics.readthedocs.io/en/latest/pages/overview.html#metrics-and-devices
        train_accuracy = torchmetrics.Accuracy().to(DEVICE)

        # set the network in training mode
        network.train()

        batch_execution_times = []
        batch_loading_times = []
        pre_batch_loading_time = time.time()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
            batch_start_time = time.time()
            batch_loading_time = batch_start_time - pre_batch_loading_time
            if batch_number < num_train_batches:
                batch_loading_times.append(batch_loading_time)

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # zero accumulated gradients
            network.zero_grad()

            # forward pass
            output = network(inputs)

            # get predicted labels from output
            predictions = network.get_predictions(output)

            with torch.no_grad():
                # get class indexes from the one-hot encoded labels
                labels = torch.argmax(labels, dim=1)

            # compute training loss
            training_loss = criterion(output, labels)
            training_losses.append(training_loss.item())
            summary_writer.add_scalar("loss/training", training_loss, epoch)

            # perform back propagation
            training_loss.backward()

            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(network.parameters(), experiment.clip_max_norm)

            # perform an optimization step
            optimizer.step()

            batch_train_accuracy = train_accuracy(predictions, labels)
            average_training_loss = np.average(training_losses)

            batch_finish_time = time.time()
            pre_batch_loading_time = batch_finish_time
            batch_execution_time = batch_finish_time - batch_start_time
            if batch_number < num_train_batches:
                batch_execution_times.append(batch_execution_time)

            train_progress = f"epoch {epoch:{max_epochs_length}} batch {batch_number:{num_batches_length}} of {num_train_batches} | average loss: {average_training_loss:.4f} | accuracy: {batch_train_accuracy:.4f} | execution: {batch_execution_time:.2f}s | loading: {batch_loading_time:.2f}s"
            logger.info(train_progress)

        experiment.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        experiment.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        num_validation_batches = math.ceil(
            experiment.validation_size / experiment.batch_size
        )
        num_batches_length = len(str(num_validation_batches))

        validation_losses = []
        validation_accuracy = torchmetrics.Accuracy().to(DEVICE)

        # disable gradient calculation
        with torch.no_grad():
            # set the network in evaluation mode
            network.eval()
            for batch_number, (inputs, labels) in enumerate(validation_loader, start=1):
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # forward pass
                output = network(inputs)

                # get predicted labels from output
                predictions = network.get_predictions(output)

                # get class indexes from the one-hot encoded labels
                labels = torch.argmax(labels, dim=1)

                # compute validation loss
                validation_loss = criterion(output, labels)
                validation_losses.append(validation_loss.item())
                summary_writer.add_scalar("loss/validation", validation_loss, epoch)

                batch_validation_accuracy = validation_accuracy(predictions, labels)
                average_validation_loss = np.average(validation_losses)

                validation_progress = f"epoch {epoch:{max_epochs_length}} validation batch {batch_number:{num_batches_length}} of {num_validation_batches} | average loss: {average_validation_loss:.4f} | accuracy: {batch_validation_accuracy:.4f}"
                logger.info(validation_progress)

        average_validation_loss = np.average(validation_losses)
        experiment.average_validation_losses.append(average_validation_loss)

        total_validation_accuracy = validation_accuracy.compute()

        average_batch_execution_time = sum(batch_execution_times) / len(
            batch_execution_times
        )
        average_batch_loading_time = sum(batch_loading_times) / len(batch_loading_times)

        epoch_finish_time = time.time()
        epoch_time = epoch_finish_time - epoch_start_time
        epoch_times.append(epoch_time)

        train_progress = f"epoch {epoch:{max_epochs_length}} complete | validation loss: {average_validation_loss:.4f} | validation accuracy: {total_validation_accuracy:.4f} | time: {epoch_time:.2f}s"
        logger.info(train_progress)
        logger.info(
            f"training batch average execution time: {average_batch_execution_time:.2f}s | average loading time: {average_batch_loading_time:.2f}s ({num_train_batches - 1} complete batches)"
        )

        if experiment.stop_early(
            network,
            optimizer,
            experiment,
            symbols_metadata,
            average_validation_loss,
            checkpoint_path,
        ):
            summary_writer.flush()
            summary_writer.close()
            break

    training_time = sum(epoch_times)
    average_epoch_time = training_time / len(epoch_times)
    logger.info(
        f"total training time: {training_time:.2f}s | epoch average training time: {average_epoch_time:.2f}s ({epoch} epochs)"
    )

    return checkpoint_path


def test_network(checkpoint_path, print_sample_assignments=False):
    """
    Calculate test loss and generate metrics.
    """
    experiment, network, _optimizer, _symbols_metadata = load_checkpoint(checkpoint_path)

    logger.info("start testing classifier")
    logger.info(f"experiment:\n{experiment}")
    logger.info(f"network:\n{network}")

    # get test dataloader
    _, _, test_loader = generate_dataloaders(experiment)

    criterion = experiment.criterion

    num_test_batches = math.ceil(experiment.test_size / experiment.batch_size)
    num_batches_length = len(str(num_test_batches))

    test_losses = []
    test_accuracy = torchmetrics.Accuracy().to(DEVICE)
    test_precision = torchmetrics.Precision(
        num_classes=experiment.num_symbols, average="macro"
    ).to(DEVICE)
    test_recall = torchmetrics.Recall(
        num_classes=experiment.num_symbols, average="macro"
    ).to(DEVICE)

    with torch.no_grad():
        network.eval()

        for batch_number, (inputs, labels) in enumerate(test_loader, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # forward pass
            output = network(inputs)

            # get predicted labels from output
            predictions = network.get_predictions(output)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

            # calculate test loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            batch_accuracy = test_accuracy(predictions, labels)
            test_precision(predictions, labels)
            test_recall(predictions, labels)

            logger.info(
                f"test batch {batch_number:{num_batches_length}} of {num_test_batches} | accuracy: {batch_accuracy:.4f}"
            )

    # log statistics
    average_test_loss = np.mean(test_losses)
    total_test_accuracy = test_accuracy.compute()
    precision = test_precision.compute()
    recall = test_recall.compute()
    logger.info(
        f"testing complete | average loss: {average_test_loss:.4f} | accuracy: {total_test_accuracy:.4f}"
    )
    logger.info(f"precision: {precision:.4f} | recall: {recall:.4f}")

    if print_sample_assignments:
        num_sample_assignments = 10
        # num_sample_assignments = 20
        # num_sample_assignments = 100

        with torch.no_grad():
            network.eval()

            inputs, labels = next(iter(test_loader))
            # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            inputs = inputs.to(DEVICE)

            with torch.random.fork_rng():
                torch.manual_seed(time.time() * 1000)
                permutation = torch.randperm(len(inputs))

            inputs = inputs[permutation[0:num_sample_assignments]]
            labels = labels[permutation[0:num_sample_assignments]]

            # forward pass
            output = network(inputs)

            # get predicted labels from output
            predictions = network.get_predictions(output)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

        # reset logger, add raw messages format
        logger.remove()
        logger.add(sys.stderr, format="{message}")
        log_file_path = pathlib.Path(checkpoint_path).with_suffix(".log")
        logger.add(log_file_path, format="{message}")

        assignments = network.symbol_mapper.one_hot_to_label(predictions.cpu())
        labels = network.symbol_mapper.one_hot_to_label(labels)

        logger.info("\nsample assignments")
        logger.info("assignment | true label")
        logger.info("-----------------------")
        for assignment, label in zip(assignments, labels):
            if assignment == label:
                logger.info(f"{assignment:>10} | {label:>10}")
            else:
                logger.info(f"{assignment:>10} | {label:>10}  !!!")


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


def log_pytorch_cuda_info():
    """
    Log PyTorch and CUDA info and device to be used.
    """
    logger.debug(f"{torch.__version__=}")
    logger.debug(f"{DEVICE=}")
    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")

    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")


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
        assignments_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{canonical_fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {canonical_fasta_path}")
            assign_symbols(
                network,
                symbols_metadata,
                canonical_fasta_path,
                scientific_name=assembly.scientific_name,
                output_directory=checkpoint_path.parent,
            )

        comparisons_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{assignments_csv_path.stem}_compare.csv"
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

        clade_statistics = f"{group_string}\n{averages_message}"

        clade_groups_statistics.append(clade_statistics)

    comparison_statistics_string = "comparison statistics:\n"
    comparison_statistics_string += "\n\n".join(
        clade_statistics for clade_statistics in clade_groups_statistics
    )
    logger.info(comparison_statistics_string)


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
    logger.add(log_file_path, format=logging_format)

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
        "-ex",
        "--experiment_settings",
        help="path to the experiment settings configuration file",
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

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=logging_format)

    # train a new classifier
    if args.train and args.experiment_settings:
        # read the experiment settings YAML file to a dictionary
        with open(args.experiment_settings) as file:
            experiment_settings = yaml.safe_load(file)

        if args.datetime is None:
            datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        else:
            datetime = args.datetime

        # generate new experiment
        experiment = Experiment(experiment_settings, datetime)

        pathlib.Path(experiment.experiment_directory).mkdir(exist_ok=True)
        log_file_path = f"{experiment.experiment_directory}/{experiment.filename}.log"
        logger.add(log_file_path, format=logging_format)

        log_pytorch_cuda_info()

        torch.manual_seed(experiment.random_seed)

        # get training, validation, and test dataloaders
        training_loader, validation_loader, _test_loader = generate_dataloaders(
            experiment
        )

        # load symbols metadata
        symbols_metadata_filename = "symbols_metadata.json"
        symbols_metadata_path = data_directory / symbols_metadata_filename
        with open(symbols_metadata_path) as file:
            symbols_metadata = json.load(file)

        # instantiate neural network
        network = GeneSymbolClassifier(
            experiment.sequence_length,
            experiment.padding_side,
            experiment.num_protein_letters,
            experiment.num_clades,
            experiment.num_symbols,
            experiment.num_connections,
            experiment.dropout_probability,
            experiment.symbol_mapper,
            experiment.protein_sequence_mapper,
            experiment.clade_mapper,
        )
        network.to(DEVICE)

        # optimization function
        optimizer = torch.optim.Adam(network.parameters(), lr=experiment.learning_rate)

        logger.info("start training new classifier")
        logger.info(f"experiment:\n{experiment}")
        logger.info(f"network:\n{network}")

        checkpoint_path = train_network(
            network,
            optimizer,
            experiment,
            symbols_metadata,
            training_loader,
            validation_loader,
        )

        if args.test:
            test_network(checkpoint_path, print_sample_assignments=True)

    # resume training and/or test a classifier
    elif (args.train or args.test) and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = checkpoint_path.with_suffix(".log")
        logger.add(log_file_path, format=logging_format)

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
        log_file_path = pathlib.Path(
            f"{checkpoint_path.parent}/{checkpoint_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=logging_format)

        evaluate_network(checkpoint_path, args.complete)

    # assign symbols to sequences
    elif args.sequences_fasta and args.scientific_name and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = checkpoint_path.with_suffix(".log")
        logger.add(log_file_path, format=logging_format)

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
