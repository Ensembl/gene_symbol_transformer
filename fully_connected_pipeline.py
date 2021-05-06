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
Fully connected neural network pipeline.
"""


# standard library imports
import argparse
import csv
import datetime as dt
import math
import pathlib
import sys
import time

# third party imports
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import yaml

from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from pipeline_abstractions import (
    EarlyStopping,
    PrettySimpleNamespace,
    SequenceDataset,
    TrainingSession,
    experiments_directory,
    load_checkpoint,
    read_fasta_in_chunks,
    specify_device,
    transform_sequences,
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
        num_symbols,
        num_connections,
        dropout_probability,
        gene_symbols,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.gene_symbols = gene_symbols

        input_size = self.sequence_length * num_protein_letters
        output_size = num_symbols

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )
        self.dropout = nn.Dropout(dropout_probability)

        self.relu = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        # flatten sample values to one dimension
        # original x.shape: torch.Size([batch_size, sequence_length, num_protein_letters])
        # e.g. torch.Size([512, 1000, 27])
        # flattened x.shape: torch.Size([batch_size, sequence_length * num_protein_letters])
        # e.g. torch.Size([512, 27000])
        x = torch.flatten(x, start_dim=1)

        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def predict(self, sequences):
        """
        Get assignments of symbols for a list of protein sequences.
        """
        tensor_sequences = transform_sequences(sequences, self.sequence_length)
        tensor_sequences = tensor_sequences.to(DEVICE)

        # run inference
        with torch.no_grad():
            self.eval()
            output = self.forward(tensor_sequences)

        # get predicted labels from output
        predictions = self.get_predictions(output)

        predictions = self.gene_symbols.one_hot_encoding_to_symbol(predictions)
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
    stop_early = EarlyStopping(
        checkpoint_path, training_session.patience, training_session.loss_delta
    )
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
            # inputs.shape: torch.Size([batch_size, sequence_length, num_protein_letters])
            # e.g. torch.Size([512, 1000, 27])
            # inputs[i].shape: torch.Size([sequence_length, num_protein_letters])
            # e.g. torch.Size([1000, 27])
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

        if stop_early(network, training_session, average_validation_loss):
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

        assignments = network.gene_symbols.one_hot_encoding_to_symbol(predictions)
        labels = network.gene_symbols.one_hot_encoding_to_symbol(labels)

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
    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]

    path = checkpoint_path
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")

    torch.save(network, network_path)

    return network_path


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
        "--save_network",
        action="store_true",
        help="extract the network from a checkpoint file"
    )

    args = argument_parser.parse_args()

    # load experiment settings and generate the log file path
    if args.experiment_settings:
        with open(args.experiment_settings) as f:
            experiment = PrettySimpleNamespace(**yaml.safe_load(f))

        if args.datetime is None:
            datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        else:
            datetime = args.datetime

        log_file_path = (
            experiments_directory / f"n={experiment.num_symbols}_{datetime}.log"
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

    if args.sequences_fasta:
        fasta_path = pathlib.Path(args.sequences_fasta)

        checkpoint_path = pathlib.Path(args.checkpoint)
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]

        logger.info("assigning symbols...")

        assignments_csv_path = pathlib.Path(
            f"{fasta_path.parent}/{fasta_path.stem}_symbols.csv"
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

                stable_ids = [
                    fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries
                ]
                sequences = [fasta_entry[1] for fasta_entry in fasta_entries]

                start = time.time()
                assignments = network.predict(sequences)
                end = time.time()
                inference_duration = end - start
                logger.debug(f"inference call took {inference_duration:.3f} seconds")

                # save assignments to the CSV file
                csv_writer.writerows(zip(stable_ids, assignments))
        logger.debug(f"symbol assignments saved at {assignments_csv_path}")
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
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]
    else:
        if experiment.num_symbols == 3:
            test_ratio = 0.2
            validation_ratio = 0.2
        elif experiment.num_symbols in {101, 1013}:
            test_ratio = 0.1
            validation_ratio = 0.1
        else:
            test_ratio = 0.05
            validation_ratio = 0.05

        # larger patience for short epochs and smaller patience for longer epochs
        if experiment.num_symbols in {3, 101, 1013}:
            patience = 11
        else:
            patience = 7

        training_session = TrainingSession(
            experiment.num_symbols,
            datetime,
            experiment.random_state,
            test_ratio,
            validation_ratio,
            experiment.sequence_length,
            experiment.batch_size,
            experiment.num_connections,
            experiment.dropout_probability,
            experiment.learning_rate,
            experiment.num_epochs,
            patience,
        )

        # loss function
        training_session.criterion = nn.NLLLoss()

    torch.manual_seed(training_session.random_state)

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        training_session.num_symbols, training_session.sequence_length
    )

    if not args.checkpoint:
        # neural network instantiation
        ############################################################################
        # num_protein_letters = len(dataset.protein_letters)
        num_protein_letters = 27

        network = FullyConnectedNetwork(
            training_session.sequence_length,
            num_protein_letters,
            training_session.num_symbols,
            training_session.num_connections,
            training_session.dropout_probability,
            dataset.gene_symbols,
        )
        ############################################################################
        training_session.device = DEVICE

        network.to(DEVICE)

    pandas_symbols_categories = (
        dataset.gene_symbols.symbol_categorical_datatype.categories
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

    if training_session.num_symbols in {3, 101}:
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
            checkpoint = load_checkpoint(checkpoint_path)
            network = checkpoint["network"]
            training_session = checkpoint["training_session"]
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
