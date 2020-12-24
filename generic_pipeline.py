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
Generic training and testing pipeline functions and classes.
"""


# standard library imports
import pathlib
import sys

import pprint

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter

# project imports
import dataset_generation


USE_CACHE = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_directory = pathlib.Path("data")
networks_directory = pathlib.Path("networks")


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.
    """

    def __init__(self, num_most_frequent_symbols, sequence_length):
        print(
            f"Loading {num_most_frequent_symbols} most frequent symbols sequences dataset...",
            end="",
        )
        data_pickle_path = (
            data_directory / f"most_frequent_{num_most_frequent_symbols}.pickle"
        )
        data = pd.read_pickle(data_pickle_path)
        print(" Done.")
        print()

        # only the sequences and the symbols are needed as features and labels
        self.data = data[["sequence", "symbol"]]

        # sequence length statistics
        # print(self.data["sequence"].str.len().describe())

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            # self.data["sequence"] = self.data["sequence"].map(lambda x: pad_or_truncate_sequence(x, sequence_length))
            self.data["sequence"] = self.data["sequence"].str.pad(
                width=sequence_length, side="left", fillchar=" "
            )
            self.data["sequence"] = self.data["sequence"].str.slice(stop=sequence_length)

        # generate a categorical data type for symbols
        labels = self.data["symbol"].unique().tolist()
        labels.sort()
        self.symbol_categorical_datatype = pd.CategoricalDtype(
            categories=labels, ordered=True
        )

        # generate a categorical data type for protein letters
        self.protein_letters = get_protein_letters()
        self.protein_letters.sort()
        self.protein_letters_categorical_datatype = pd.CategoricalDtype(
            categories=self.protein_letters, ordered=True
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data.iloc[index]["sequence"]
        symbol = self.data.iloc[index]["symbol"]

        # generate one-hot encoding of the sequence
        protein_letters_categorical = pd.Series(
            list(sequence), dtype=self.protein_letters_categorical_datatype
        )
        one_hot_sequence = pd.get_dummies(
            protein_letters_categorical, prefix="protein_letter"
        )

        # generate one-hot encoding of the label (symbol)
        symbol_categorical = pd.Series(symbol, dtype=self.symbol_categorical_datatype)
        one_hot_symbol = pd.get_dummies(symbol_categorical, prefix="symbol")

        # convert features and labels to NumPy arrays
        one_hot_sequence = one_hot_sequence.to_numpy()
        one_hot_symbol = one_hot_symbol.to_numpy()

        # cast the arrays to `np.float32` data type, so that the PyTorch tensors
        # will be generated with type `torch.FloatTensor`.
        one_hot_sequence = one_hot_sequence.astype(np.float32)
        one_hot_symbol = one_hot_symbol.astype(np.float32)

        # remove extra dimension for a single example
        one_hot_symbol = np.squeeze(one_hot_symbol)

        item = one_hot_sequence, one_hot_symbol

        return item


def get_protein_letters():
    """
    Generate and return a list of protein letters that occur in the dataset and
    those that can potentially be used.
    """
    extended_IUPAC_protein_letters = Bio.Alphabet.IUPAC.ExtendedIUPACProtein.letters

    # cache the following operation, as it's very expensive in time and space
    if USE_CACHE:
        extra_letters = ["*"]
    else:
        data = dataset_generation.load_data()

        # generate a list of all protein letters that occur in the dataset
        dataset_letters = set(data["sequence"].str.cat())

        extra_letters = [
            letter
            for letter in dataset_letters
            if letter not in extended_IUPAC_protein_letters
        ]

    protein_letters = list(extended_IUPAC_protein_letters) + extra_letters
    assert len(protein_letters) == 27, protein_letters

    return protein_letters


def pad_or_truncate_sequence(sequence, normalized_length):
    """
    Pad or truncate `sequence` to be exactly `normalized_length` letters long.

    NOTE
    Maybe use this inside a DataLoader.
    """
    sequence_length = len(sequence)

    if sequence_length <= normalized_length:
        normalized_sequence = " " * (normalized_length - sequence_length) + sequence
    else:
        normalized_sequence = sequence[:normalized_length]

    return normalized_sequence


class SuppressSettingWithCopyWarning:
    """
    Suppress SettingWithCopyWarning warning.

    https://stackoverflow.com/a/53954986
    """

    def __init__(self):
        pass

    def __enter__(self):
        self.original_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.original_setting


def train_network(
    network,
    training_session,
    training_loader,
    validation_loader,
    num_training,
    verbose=False,
):
    """
    """
    tensorboard_log_dir = f'runs/{training_session.num_most_frequent_symbols}/{training_session.datetime}'
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    num_epochs = training_session.num_epochs

    criterion = training_session.criterion

    # optimization function
    optimizer = torch.optim.Adam(network.parameters(), lr=training_session.learning_rate)
    training_session.optimizer = optimizer

    clip_max_norm = 5

    checkpoint_filename = f'n={training_session.num_most_frequent_symbols}_{training_session.datetime}.net'
    checkpoint_path = networks_directory / checkpoint_filename
    patience = 11
    loss_delta = 0.001
    stop_early = EarlyStopping(checkpoint_path, patience, loss_delta)
    print(f"checkpoints of the network being trained saved to {checkpoint_path}")
    print()

    num_epochs_length = len(str(num_epochs))

    num_batches = int(num_training / training_session.batch_size)
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
        h = network.init_hidden(training_session.batch_size)

        # set the network in training mode
        network.train()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            epoch_end = batch_number == num_batches

            # generate new variables for the hidden state
            h = tuple(tensor.data for tensor in h)

            # zero accumulated gradients
            network.zero_grad()

            # get network output and hidden state
            output, h = network(inputs, h)

            with torch.no_grad():
                # get class indexes from the labels one hot encoding
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
            optimizer.step()

            if verbose and not epoch_end:
                average_training_loss = np.average(training_losses)

                training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, batch {batch_number:{num_batches_length}} of {num_batches} | average training loss: {average_training_loss:.4f}"
                print(training_progress)

        training_session.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        training_session.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        validation_losses = []
        h = network.init_hidden(training_session.batch_size)

        # disable gradient calculation
        with torch.no_grad():
            # set the network in evaluation mode
            network.eval()

            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # generate new variables for the hidden state
                h = tuple(tensor.data for tensor in h)

                output, h = network(inputs, h)
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
        print(training_progress)

        if stop_early(
            network, training_session, average_validation_loss
        ):
            summary_writer.flush()
            summary_writer.close()

            break


def load_checkpoint(checkpoint_path):
    """
    Load saved training checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    return checkpoint


def test_network(network, training_session, test_loader):
    """
    Calculate test loss and generate metrics.
    """
    criterion = training_session.criterion

    # initialize hidden state
    h = network.init_hidden(training_session.batch_size)

    test_losses = []
    num_correct_predictions = 0

    with torch.no_grad():
        network.eval()

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # create new variables for the hidden state
            h = tuple(tensor.data for tensor in h)

            # get output values
            output, h = network(inputs, h)

            # get predicted labels from output
            predicted_probabilities = torch.exp(output)
            predictions = torch.argmax(predicted_probabilities, dim=1)

            # get class indexes from one hot labels
            labels = torch.argmax(labels, dim=1)

            # calculate test loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            # predictions to ground truth comparison
            predictions_correctness = predictions.eq(labels)
            num_correct_predictions += torch.sum(predictions_correctness).item()

    # print statistics
    print("average test loss: {:.4f}".format(np.mean(test_losses)))

    # test predictions accuracy
    test_accuracy = num_correct_predictions / len(test_loader.dataset)
    print("test accuracy: {:.3f}".format(test_accuracy))


def save_training_checkpoint(
    network, training_session, checkpoint_path
):
    """
    """
    checkpoint = {
        "network": network,
        "training_session": training_session,
    }
    torch.save(checkpoint, checkpoint_path)


class EarlyStopping:
    """
    Stop training if validation loss doesn't improve during a specified patience period.
    """

    def __init__(self, checkpoint_path, patience=7, loss_delta=0):
        """
        Arguments:
            patience (int): Number of calls to continue training if validation loss is not improving. Defaults to 7.
            loss_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
            checkpoint_path (path-like object): Path to save the checkpoint.
        """
        self.patience = patience
        self.loss_delta = loss_delta
        self.checkpoint_path = checkpoint_path

        self.no_progress = 0
        self.min_validation_loss = np.Inf

    def __call__(self, network, training_session, validation_loss):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            print("saving initial network checkpoint...")
            print()
            save_training_checkpoint(
                network, training_session, self.checkpoint_path
            )
            return False

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert validation_loss_decrease > 0, f"{validation_loss_decrease=}, should be a positive number"
            print(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )
            print()
            save_training_checkpoint(
                network, training_session, self.checkpoint_path
            )
            self.min_validation_loss = validation_loss
            self.no_progress = 0
            return False

        else:
            self.no_progress += 1
            print()

            if self.no_progress == self.patience:
                print(
                    f"{self.no_progress} calls with no validation loss improvement. Stopping training."
                )
                return True


class TrainingSession:
    """
    """
    def __init__(self, args):
        self.datetime = args.datetime

        # self.random_state = None
        # self.random_state = 5
        # self.random_state = 7
        # self.random_state = 11
        self.random_state = args.random_state

        # self.num_most_frequent_symbols = 3
        # self.num_most_frequent_symbols = 101
        # self.num_most_frequent_symbols = 1013
        # self.num_most_frequent_symbols = 10059
        # self.num_most_frequent_symbols = 20147
        # self.num_most_frequent_symbols = 25028
        # self.num_most_frequent_symbols = 30591
        self.num_most_frequent_symbols = args.num_most_frequent_symbols

        # padding or truncating length
        self.sequence_length = 1000
        # self.sequence_length = 2000

        if self.num_most_frequent_symbols == 3:
            self.test_ratio = 0.2
            self.validation_ratio = 0.2
        elif self.num_most_frequent_symbols in {101, 1013}:
            self.test_ratio = 0.1
            self.validation_ratio = 0.1
        else:
            self.test_ratio = 0.05
            self.validation_ratio = 0.05

        # self.batch_size = 32
        self.batch_size = 64
        # self.batch_size = 128
        # self.batch_size = 200
        # self.batch_size = 256
        # self.batch_size = 512

        self.learning_rate = 0.001
        # self.learning_rate = 0.01

        # self.num_epochs = 10
        self.num_epochs = 100
        # self.num_epochs = 1000

        self.num_complete_epochs = 0

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


if __name__ == "__main__":
    print("library file, import to use")
