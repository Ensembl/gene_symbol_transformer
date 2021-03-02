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
Fully connected neural network pipeline.
"""


# standard library imports
import argparse
import math
import pathlib
import sys
import time

# third party imports
import numpy as np
import torch
import torch.nn as nn

from colorama import Fore
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from pipeline_abstractions import (
    load_checkpoint,
    networks_directory,
    EarlyStopping,
    SequenceDataset,
    TrainingSession,
)


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FullyConnectedNetwork(nn.Module):
    """
    A fully connected neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(
        self,
        sequence_length,
        num_protein_letters,
        num_most_frequent_symbols,
        dropout_probability,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        input_size = sequence_length * num_protein_letters
        # num_connections = 256
        num_connections = 512
        # num_connections = 1024
        # num_connections = 2048
        output_size = num_most_frequent_symbols

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        # self.hidden_layer = nn.Linear(in_features=None, out_features=None)
        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )
        self.dropout = nn.Dropout(dropout_probability)

        self.relu = nn.ReLU()
        # self.final_activation = nn.LogSoftmax(dim=2)
        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        num_samples = len(x)
        x = x.view(num_samples, -1)

        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        # x = self.hidden_layer(x)
        # x = self.relu(x)

        x = self.output_layer(x)
        x = self.dropout(x)
        x = self.final_activation(x)

        return x


def train_network(
    network,
    training_session,
    training_loader,
    validation_loader,
    verbose=False,
):
    tensorboard_log_dir = (
        f"runs/{training_session.num_most_frequent_symbols}/{training_session.datetime}"
    )
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    num_epochs = training_session.num_epochs
    criterion = training_session.criterion

    # optimization function
    training_session.optimizer = torch.optim.Adam(
        network.parameters(), lr=training_session.learning_rate
    )

    clip_max_norm = 5

    checkpoint_path = networks_directory / training_session.checkpoint_filename
    stop_early = EarlyStopping(
        checkpoint_path, training_session.patience, training_session.loss_delta
    )
    logger.info(f"checkpoints of the network being trained saved to {checkpoint_path}")
    # print()

    num_epochs_length = len(str(num_epochs))

    if training_session.drop_last:
        num_batches = int(training_session.training_size / training_session.batch_size)
    else:
        num_batches = math.ceil(
            training_session.training_size / training_session.batch_size
        )
    num_batches_length = len(str(num_batches))

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

        # set the network in training mode
        network.train()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
            epoch_end = batch_number == num_batches

            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # zero accumulated gradients
            network.zero_grad()

            # get network output and hidden state
            output = network(inputs)

            with torch.no_grad():
                # get class indexes from the one-hot encoded labels
                labels = torch.argmax(labels, dim=1)

            # calculate the training loss
            training_loss = criterion(output, labels)
            training_losses.append(training_loss.item())
            summary_writer.add_scalar("loss/training", training_loss, epoch)

            # perform back propagation
            training_loss.backward()

            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)

            # perform an optimization step
            training_session.optimizer.step()

            if verbose and not epoch_end:
                average_training_loss = np.average(training_losses)

                training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, batch {batch_number:{num_batches_length}} of {num_batches} | average training loss: {average_training_loss:.4f}"
                logger.info(training_progress)

        training_session.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        training_session.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        validation_losses = []

        # disable gradient calculation
        with torch.no_grad():
            # set the network in evaluation mode
            network.eval()

            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                output = network(inputs)
                labels = torch.argmax(labels, dim=1)
                validation_loss = criterion(output, labels)
                validation_losses.append(validation_loss.item())
                summary_writer.add_scalar("loss/validation", validation_loss, epoch)

        average_validation_loss = np.average(validation_losses)
        training_session.average_validation_losses.append(average_validation_loss)

        training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, "
        if verbose:
            training_progress += (
                f"batch {batch_number:{num_batches_length}} of {num_batches} "
            )
        training_progress += f"| average training loss: {average_training_loss:.4f}, average validation loss: {average_validation_loss:.4f}"
        logger.info(training_progress)

        if stop_early(network, training_session, average_validation_loss):
            summary_writer.flush()
            summary_writer.close()

            break


def test_network(network, training_session, test_loader, print_sample_predictions=False):
    """
    Calculate test loss and generate metrics.
    """
    criterion = training_session.criterion

    if training_session.drop_last:
        num_batches = int(training_session.test_size / training_session.batch_size)
    else:
        num_batches = math.ceil(training_session.test_size / training_session.batch_size)
    num_batches_length = len(str(num_batches))

    test_losses = []
    num_correct_predictions = 0

    with torch.no_grad():
        network.eval()

        num_samples = 0
        for batch_number, (inputs, labels) in enumerate(test_loader, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # get output values
            output = network(inputs)

            # get predicted labels from output
            predicted_probabilities = torch.exp(output)
            predictions = torch.argmax(predicted_probabilities, dim=1)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

            # calculate test loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            # predictions to ground truth comparison
            predictions_correctness = predictions.eq(labels)
            num_correct_predictions += torch.sum(predictions_correctness).item()

            num_samples += len(predictions)
            running_test_accuracy = num_correct_predictions / num_samples

            logger.info(
                f"batch {batch_number:{num_batches_length}} of {num_batches} | running test accuracy: {running_test_accuracy:.4f}"
            )
    # print()

    # print statistics
    print()
    print("average test loss: {:.4f}".format(np.mean(test_losses)))

    # test predictions accuracy
    test_accuracy = num_correct_predictions / num_samples
    print("test accuracy: {:.3f}".format(test_accuracy))

    if print_sample_predictions:
        # num_sample_predictions = 10
        # num_sample_predictions = 20
        num_sample_predictions = 100

        with torch.no_grad():
            network.eval()

            inputs, labels = next(iter(test_loader))
            # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            with torch.random.fork_rng():
                torch.manual_seed(time.time() * 1000)
                permutation = torch.randperm(len(inputs))

            inputs = inputs[permutation[0:num_sample_predictions]]
            labels = labels[permutation[0:num_sample_predictions]]

            # get output values
            output = network(inputs)

            # get predicted labels from output
            predicted_probabilities = torch.exp(output)
            predictions = torch.argmax(predicted_probabilities, dim=1)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

        predictions = training_session.gene_symbols.one_hot_encoding_to_symbol(
            predictions
        )
        labels = training_session.gene_symbols.one_hot_encoding_to_symbol(labels)

        print()
        print("sample predictions")
        print("prediction | true label")
        print("-----------------------")
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                # unicode "CHECK MARK"
                correctness = "\u2713"
                print(
                    f"{prediction:10} | {label:10}    "
                    + Fore.GREEN
                    + f"{correctness}"
                    + Fore.RESET
                )
            else:
                # unicode "BALLOT X"
                correctness = "\u2717"
                print(
                    Fore.RED
                    + f"{prediction:10}"
                    + Fore.RESET
                    + " | "
                    + Fore.RED
                    + f"{label:10}    {correctness}"
                    + Fore.RESET
                )


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
    argument_parser.add_argument("--train", action="store_true")
    argument_parser.add_argument("--test", action="store_true")
    argument_parser.add_argument("--load")
    argument_parser.add_argument("--save_network")
    argument_parser.add_argument("--datetime")
    argument_parser.add_argument("--random_state", type=int)
    argument_parser.add_argument("--num_most_frequent_symbols", type=int)
    argument_parser.add_argument(
        "--sequence_length",
        type=int,
        help="constant length to pad or truncate all sequences",
    )
    argument_parser.add_argument(
        "--batch_size", type=int, help="training and test batch size"
    )
    argument_parser.add_argument("--learning_rate", type=float, help="learning rate")
    argument_parser.add_argument(
        "--num_epochs", type=int, help="number of epochs to train the network"
    )

    args = argument_parser.parse_args()

    if args.save_network:
        checkpoint_path = pathlib.Path(args.save_network)
        logger.info(f'Loading checkpoint "{checkpoint_path}" ...')
        network_path = save_network_from_checkpoint(checkpoint_path)
        logger.info(f'Saved network at "{network_path}"')
        return

    # print PyTorch version information
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
        # print(f"{torch.cuda.memory_summary(DEVICE)}")

    # load training checkpoint or generate new training session
    if args.load:
        checkpoint_path = pathlib.Path(args.load)
        logger.info(f'Loading training checkpoint "{checkpoint_path}"...')
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]
        logger.info(f'"{checkpoint_path}" training checkpoint loaded')
    else:
        if args.num_most_frequent_symbols == 3:
            test_ratio = 0.2
            validation_ratio = 0.2
        elif args.num_most_frequent_symbols in {101, 1013}:
            test_ratio = 0.1
            validation_ratio = 0.1
        else:
            test_ratio = 0.05
            validation_ratio = 0.05

        # larger patience for short epochs and smaller patience for longer epochs
        if args.num_most_frequent_symbols in {3, 101, 1013}:
            patience = 11
        else:
            patience = 7

        training_session = TrainingSession(
            args.datetime,
            args.random_state,
            args.num_most_frequent_symbols,
            test_ratio,
            validation_ratio,
            args.sequence_length,
            args.batch_size,
            args.learning_rate,
            args.num_epochs,
            patience,
        )

        # training_session.hidden_size = 128
        # training_session.hidden_size = 256
        # training_session.hidden_size = 512
        # training_session.hidden_size = 1024

        training_session.dropout_probability = 0
        # training_session.dropout_probability = 0.05
        # training_session.dropout_probability = 0.1
        # training_session.dropout_probability = 0.2

        # loss function
        training_session.criterion = nn.NLLLoss()

        # neural network instantiation
        ############################################################################
        # num_protein_letters = len(dataset.protein_letters)
        num_protein_letters = 27

        network = FullyConnectedNetwork(
            training_session.sequence_length,
            num_protein_letters,
            training_session.num_most_frequent_symbols,
            training_session.dropout_probability,
        )
        ############################################################################
        training_session.device = DEVICE

        network.to(DEVICE)

    torch.manual_seed(training_session.random_state)

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        training_session.num_most_frequent_symbols, training_session.sequence_length
    )

    training_session.gene_symbols = dataset.gene_symbols
    training_session.protein_sequences = dataset.protein_sequences

    pandas_symbols_categories = (
        dataset.gene_symbols.symbol_categorical_datatype.categories
    )
    logger.info(
        "gene symbols:\n",
        pandas_symbols_categories.to_series(
            index=range(len(pandas_symbols_categories)), name="gene symbols"
        ),
    )
    # print()

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
    # print()

    # set the batch size to the size of the smallest dataset if larger than that
    min_dataset_size = min(
        training_session.training_size,
        training_session.validation_size,
        training_session.test_size,
    )
    if training_session.batch_size > min_dataset_size:
        training_session.batch_size = min_dataset_size

    if training_session.num_most_frequent_symbols in {3, 101}:
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

    logger.info("network:\n", network)
    # print()
    logger.info("training_session:\n", training_session)
    # print()

    # train network
    if args.train:
        logger.info(f"training neural network")
        # print()

        verbose = True

        train_network(
            network,
            training_session,
            training_loader,
            validation_loader,
            verbose,
        )

    # test trained network
    if args.test:
        if args.train:
            checkpoint_path = networks_directory / training_session.checkpoint_filename
            checkpoint = load_checkpoint(checkpoint_path)
            network = checkpoint["network"]
            training_session = checkpoint["training_session"]
        test_network(
            network, training_session, test_loader, print_sample_predictions=True
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
