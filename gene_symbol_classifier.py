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
Gene symbol classifier.
"""


# standard library imports
import argparse
import csv
import datetime as dt
import math
import pathlib
import pprint
import sys
import time

# third party imports
import ensembl_rest
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml

from icecream import ic
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from utils import (
    CladesMapper,
    PrettySimpleNamespace,
    ProteinSequencesMapper,
    SequenceDataset,
    experiments_directory,
    get_clade,
    load_checkpoint,
    dev_datasets_symbol_frequency,
    read_fasta_in_chunks,
    specify_device,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

DEVICE = specify_device()


class FullyConnectedNetwork(nn.Module):
    """
    A fully connected neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(
        self,
        sequence_length,
        num_protein_letters,
        num_clades,
        num_symbols,
        num_connections,
        dropout_probability,
        gene_symbols_mapper,
        protein_sequences_mapper,
        clades_mapper,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.dropout_probability = dropout_probability
        self.gene_symbols_mapper = gene_symbols_mapper
        self.protein_sequences_mapper = protein_sequences_mapper
        self.clades_mapper = clades_mapper

        input_size = (self.sequence_length * num_protein_letters) + num_clades
        output_size = num_symbols

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        if self.dropout_probability > 0:
            self.dropout = nn.Dropout(self.dropout_probability)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )

        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        x = self.input_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def predict(self, sequences, clades):
        """
        Get assignments of symbols for a list of protein sequences.
        """
        features_tensor = self.generate_features_tensor(sequences, clades)

        features_tensor = features_tensor.to(DEVICE)

        # run inference
        with torch.no_grad():
            self.eval()
            output = self.forward(features_tensor)

        # get predicted labels from output
        predictions = self.get_predictions(output)

        predictions = self.gene_symbols_mapper.one_hot_to_symbol(predictions)
        predictions = predictions.tolist()

        return predictions

    @staticmethod
    def get_predictions(output):
        """
        Get predicted labels from network's forward pass output.
        """
        predicted_probabilities = torch.exp(output)
        # get class indexes from the one-hot encoded labels
        predictions = torch.argmax(predicted_probabilities, dim=1)
        return predictions

    def generate_features_tensor(self, sequences, clades):
        """
        Convert lists of protein sequences and species clades to an one-hot
        encoded features tensor.
        """
        one_hot_features_list = []
        for sequence, clade in zip(sequences, clades):
            # pad or truncate sequence to be exactly `self.sequence_length` letters long
            string_length = len(sequence)
            if string_length <= self.sequence_length:
                sequence = " " * (self.sequence_length - string_length) + sequence
            else:
                sequence = sequence[: self.sequence_length]

            one_hot_sequence = self.protein_sequences_mapper.protein_letters_to_one_hot(
                sequence
            )
            one_hot_clade = self.clades_mapper.clade_to_one_hot(clade)

            # convert the dataframes to NumPy arrays
            one_hot_sequence = one_hot_sequence.to_numpy(dtype=np.float32)
            one_hot_clade = one_hot_clade.to_numpy(dtype=np.float32)

            # flatten sequence matrix to a vector
            flat_one_hot_sequence = one_hot_sequence.flatten()

            # remove extra dimension
            one_hot_clade = np.squeeze(one_hot_clade)

            one_hot_features_vector = np.concatenate(
                [flat_one_hot_sequence, one_hot_clade], axis=0
            )

            one_hot_features_list.append(one_hot_features_vector)

        one_hot_features = np.stack(one_hot_features_list)

        features_tensor = torch.from_numpy(one_hot_features)

        return features_tensor


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

    def __call__(self, network, training_session, validation_loss, checkpoint_path):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            logger.info("saving initial network checkpoint...")
            checkpoint = {
                "network": network,
                "training_session": training_session,
            }
            torch.save(checkpoint, checkpoint_path)
            return False

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            logger.info(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )
            checkpoint = {
                "network": network,
                "training_session": training_session,
            }
            torch.save(checkpoint, checkpoint_path)
            self.min_validation_loss = validation_loss
            self.no_progress = 0
            return False

        else:
            self.no_progress += 1

            if self.no_progress == self.patience:
                logger.info(
                    f"{self.no_progress} calls with no validation loss improvement. Stopping training."
                )
                return True


def train_network(
    network,
    training_session,
    training_loader,
    validation_loader,
):
    tensorboard_log_dir = (
        f"runs/{training_session.num_symbols}/{training_session.datetime}"
    )
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    num_epochs = training_session.num_epochs
    criterion = training_session.criterion

    # optimization function
    training_session.optimizer = torch.optim.Adam(
        network.parameters(), lr=training_session.learning_rate
    )

    clip_max_norm = 5

    checkpoint_path = experiments_directory / training_session.checkpoint_filename
    logger.info(f"training started, session checkpoints saved at {checkpoint_path}")

    num_epochs_length = len(str(num_epochs))

    if training_session.drop_last:
        num_train_batches = int(
            training_session.training_size / training_session.batch_size
        )
    else:
        num_train_batches = math.ceil(
            training_session.training_size / training_session.batch_size
        )
    num_batches_length = len(str(num_train_batches))

    if not hasattr(training_session, "average_training_losses"):
        training_session.average_training_losses = []

    if not hasattr(training_session, "average_validation_losses"):
        training_session.average_validation_losses = []

    training_session.epoch = training_session.num_complete_epochs + 1
    for epoch in range(training_session.epoch, num_epochs + 1):
        training_session.epoch = epoch

        # training
        ########################################################################
        training_losses = []
        train_accuracy = torchmetrics.Accuracy()

        # set the network in training mode
        network.train()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
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
            nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)

            # perform an optimization step
            training_session.optimizer.step()

            batch_train_accuracy = train_accuracy(predictions, labels)
            average_training_loss = np.average(training_losses)

            train_progress = f"epoch {epoch:{num_epochs_length}}, batch {batch_number:{num_batches_length}} of {num_train_batches} | average loss: {average_training_loss:.4f} | accuracy: {batch_train_accuracy:.4f}"
            logger.info(train_progress)

        training_session.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        training_session.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        if training_session.drop_last:
            num_validation_batches = int(
                training_session.validation_size / training_session.batch_size
            )
        else:
            num_validation_batches = math.ceil(
                training_session.validation_size / training_session.batch_size
            )
        num_batches_length = len(str(num_validation_batches))

        validation_losses = []
        validation_accuracy = torchmetrics.Accuracy()

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

                validation_progress = f"epoch {epoch:{num_epochs_length}}, validation batch {batch_number:{num_batches_length}} of {num_validation_batches} | average loss: {average_validation_loss:.4f} | accuracy: {batch_validation_accuracy:.4f}"
                logger.info(validation_progress)

        average_validation_loss = np.average(validation_losses)
        training_session.average_validation_losses.append(average_validation_loss)

        total_validation_accuracy = validation_accuracy.compute()

        train_progress = f"epoch {epoch:{num_epochs_length}} complete | validation loss: {average_validation_loss:.4f} | validation accuracy: {total_validation_accuracy:.4f}"
        logger.info(train_progress)

        if training_session.stop_early(
            network, training_session, average_validation_loss, checkpoint_path
        ):
            summary_writer.flush()
            summary_writer.close()
            break


def test_network(
    network, training_session, test_loader, log_file_path, print_sample_assignments=False
):
    """
    Calculate test loss and generate metrics.
    """
    logger.info("testing started")

    criterion = training_session.criterion

    if training_session.drop_last:
        num_test_batches = int(training_session.test_size / training_session.batch_size)
    else:
        num_test_batches = math.ceil(
            training_session.test_size / training_session.batch_size
        )
    num_batches_length = len(str(num_test_batches))

    test_losses = []
    test_accuracy = torchmetrics.Accuracy()
    test_precision = torchmetrics.Precision(
        num_classes=training_session.num_symbols, average="macro"
    )
    test_recall = torchmetrics.Recall(
        num_classes=training_session.num_symbols, average="macro"
    )

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
        logger.add(log_file_path, format="{message}")

        assignments = network.gene_symbols_mapper.one_hot_to_symbol(predictions)
        labels = network.gene_symbols_mapper.one_hot_to_symbol(labels)

        logger.info("\nsample assignments")
        logger.info("assignment | true label")
        logger.info("-----------------------")
        for assignment, label in zip(assignments, labels):
            if assignment == label:
                logger.info(f"{assignment:>10} | {label:>10}")
            else:
                logger.info(f"{assignment:>10} | {label:>10}  !!!")


def save_network_from_checkpoint(checkpoint_path):
    """
    Save a network residing inside a checkpoint file as a separate file.
    """
    network, training_session = load_checkpoint(checkpoint_path)

    path = checkpoint_path
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")

    torch.save(network, network_path)

    return network_path


class TrainingSession:
    def __init__(self, experiment_settings, datetime):
        # dataset
        self.num_symbols = experiment_settings.num_symbols

        # experiment parameters
        self.datetime = datetime
        self.random_state = experiment_settings.random_state

        # test and validation sets
        if self.num_symbols in dev_datasets_symbol_frequency:
            self.test_ratio = 0.2
            self.validation_ratio = 0.2
        else:
            self.test_ratio = 0.05
            self.validation_ratio = 0.05

        # samples and batches
        self.sequence_length = experiment_settings.sequence_length
        self.batch_size = experiment_settings.batch_size

        # network
        self.num_connections = experiment_settings.num_connections
        self.dropout_probability = experiment_settings.dropout_probability
        self.learning_rate = experiment_settings.learning_rate

        # training length and early stopping
        self.num_epochs = experiment_settings.num_epochs
        self.num_complete_epochs = 0

        # early stopping
        # larger patience for short epochs and smaller patience for longer epochs
        if self.num_symbols in dev_datasets_symbol_frequency:
            patience = 11
        else:
            patience = 7
        loss_delta = 0.001
        self.stop_early = EarlyStopping(patience, loss_delta)

        self.checkpoint_filename = f"n={self.num_symbols}_{self.datetime}.pth"

        # loss function
        self.criterion = nn.NLLLoss()

    def __repr__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


def assign_symbols(network, sequences_fasta, clade, output_directory):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    sequences_fasta_path = pathlib.Path(sequences_fasta)
    assignments_csv_path = pathlib.Path(
        f"{output_directory}/{sequences_fasta_path.stem}_symbols.csv"
    )

    # read the FASTA file in chunks and assign symbols
    with open(assignments_csv_path, "w+") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol"]
        csv_writer = csv.writer(csv_file, delimiter="\t")
        csv_writer.writerow(field_names)

        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            if fasta_entries[-1] is None:
                fasta_entries = [
                    fasta_entry
                    for fasta_entry in fasta_entries
                    if fasta_entry is not None
                ]

            stable_ids = [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]
            clades = [clade for _ in range(len(fasta_entries))]

            assignments = network.predict(sequences, clades)

            # save assignments to the CSV file
            csv_writer.writerows(zip(stable_ids, assignments))

    logger.info(f"symbol assignments saved at {assignments_csv_path}")


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
        help="training session checkpoint path",
    )
    argument_parser.add_argument("--train", action="store_true", help="train a network")
    argument_parser.add_argument("--test", action="store_true", help="test a network")
    argument_parser.add_argument(
        "--sequences_fasta",
        help="path of FASTA file with protein sequences to generate symbol assignments for",
    )
    argument_parser.add_argument(
        "--scientific_name",
        help="scientific name of the species the protein sequences belong to",
    )
    argument_parser.add_argument(
        "--save_network",
        action="store_true",
        help="extract the network from a checkpoint file",
    )

    args = argument_parser.parse_args()

    # load experiment settings and generate the log file path
    if args.experiment_settings:
        with open(args.experiment_settings) as f:
            experiment_settings = PrettySimpleNamespace(**yaml.safe_load(f))

        if args.datetime is None:
            datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        else:
            datetime = args.datetime

        log_file_path = (
            experiments_directory / f"n={experiment_settings.num_symbols}_{datetime}.log"
        )
    elif args.checkpoint:
        log_file_path = pathlib.Path(args.checkpoint).with_suffix(".log")
    else:
        argument_parser.print_help()
        sys.exit()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)
    logger.add(log_file_path, format=LOGURU_FORMAT)

    # save network
    if args.save_network:
        checkpoint_path = pathlib.Path(args.checkpoint)
        logger.info(f'loading checkpoint "{checkpoint_path}" ...')
        network_path = save_network_from_checkpoint(checkpoint_path)
        logger.info(f'saved network at "{network_path}"')
        return

    if args.sequences_fasta and args.scientific_name:
        checkpoint_path = pathlib.Path(args.checkpoint)
        network, _training_session = load_checkpoint(checkpoint_path)

        response = ensembl_rest.taxonomy_name(args.scientific_name)
        assert len(response) == 1

        taxonomy_id = ensembl_rest.taxonomy_name(args.scientific_name)[0]["id"]

        rest_scientific_name = ensembl_rest.taxonomy_id(taxonomy_id)["scientific_name"]
        assert rest_scientific_name.lower() == args.scientific_name.lower()

        clade = get_clade(taxonomy_id)
        logger.info(f"got clade {clade} for {args.scientific_name}")

        logger.info("assigning symbols...")
        assign_symbols(network, args.sequences_fasta, clade, checkpoint_path.parent)

        sys.exit()

    # log PyTorch version information
    logger.info(f"{torch.__version__=}")
    # specify GPU devices visible to CUDA applications
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logger.info(f"{DEVICE=}")

    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")
    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")
        # logger.debug(f"{torch.cuda.memory_summary(DEVICE)}")

    # load training checkpoint or generate new training session
    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        network, training_session = load_checkpoint(checkpoint_path)
    else:
        training_session = TrainingSession(experiment_settings, datetime)

    torch.manual_seed(training_session.random_state)

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        training_session.num_symbols, training_session.sequence_length
    )

    if not args.checkpoint:
        # neural network instantiation
        ############################################################################
        training_session.num_protein_letters = len(
            dataset.protein_sequences_mapper.protein_letters
        )
        training_session.num_clades = len(dataset.clades_mapper.clades)

        network = FullyConnectedNetwork(
            training_session.sequence_length,
            training_session.num_protein_letters,
            training_session.num_clades,
            training_session.num_symbols,
            training_session.num_connections,
            training_session.dropout_probability,
            dataset.gene_symbols_mapper,
            dataset.protein_sequences_mapper,
            dataset.clades_mapper,
        )
        ############################################################################
        training_session.device = DEVICE

        network.to(DEVICE)

    pandas_symbols_categories = (
        dataset.gene_symbols_mapper.symbol_categorical_datatype.categories
    )
    logger.info(
        "gene symbols:\n{}".format(
            pandas_symbols_categories.to_series(
                index=range(len(pandas_symbols_categories)), name="gene symbols"
            )
        )
    )

    # split dataset into train, validation, and test datasets
    validation_size = int(training_session.validation_ratio * len(dataset))
    test_size = int(training_session.test_ratio * len(dataset))
    training_session.training_size = len(dataset) - validation_size - test_size
    training_session.validation_size = validation_size
    training_session.test_size = test_size

    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        lengths=(
            training_session.training_size,
            training_session.validation_size,
            training_session.test_size,
        ),
    )

    logger.info(
        f"dataset split to train ({training_session.training_size}), validation ({training_session.validation_size}), and test ({training_session.test_size}) datasets"
    )

    # set the batch size to the size of the smallest dataset if larger than that
    min_dataset_size = min(
        training_session.training_size,
        training_session.validation_size,
        training_session.test_size,
    )
    if training_session.batch_size > min_dataset_size:
        training_session.batch_size = min_dataset_size

    if training_session.num_symbols in dev_datasets_symbol_frequency:
        training_session.drop_last = False
    else:
        training_session.drop_last = True
    training_loader = DataLoader(
        training_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    ############################################################################

    logger.info(f"network:\n{network}")
    logger.info(f"training_session:\n{training_session}")

    # train network
    if args.train:
        train_network(
            network,
            training_session,
            training_loader,
            validation_loader,
        )

    # test trained network
    if args.test:
        if args.train:
            checkpoint_path = experiments_directory / training_session.checkpoint_filename
            network, training_session = load_checkpoint(checkpoint_path)
        test_network(
            network,
            training_session,
            test_loader,
            log_file_path,
            print_sample_assignments=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
