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
Raw sequences neural network pipeline.
"""


# standard library imports
import argparse
import pathlib
import pickle
import sys

from pprint import pprint

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
import dataset_generation


summary_writer = SummaryWriter()

USE_CACHE = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_directory = pathlib.Path("../data")


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.
    """

    def __init__(self, num_most_frequent_symbols, sequence_length):
        print(
            f"Loading dataset of the {num_most_frequent_symbols} most frequent symbols sequences...",
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


class Sequence_LSTM(nn.Module):
    """
    An LSTM neural network for gene classification using the protein letters of
    the sequence as features.
    """

    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers,
        lstm_dropout_probability,
        final_dropout_probability,
        batch_first=True,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
            dropout=lstm_dropout_probability,
        )

        self.final_dropout = nn.Dropout(final_dropout_probability)

        self.linear = nn.Linear(self.hidden_size, output_size)

        # final activation function
        self.activation = nn.LogSoftmax(dim=2)

    def forward(self, x, hidden_state):
        """
        Perform a forward pass of our network on some input and hidden state.
        """
        output, hidden_state = self.lstm(x, hidden_state)

        output = self.final_dropout(output)
        output = self.linear(output)
        output = self.activation(output)

        # get the last set of output values
        output = output[:, -1]

        # return last output and hidden state
        return output, hidden_state

    def init_hidden(self, batch_size):
        """
        Initializes hidden state

        Creates two new tensors with sizes num_layers x batch_size x hidden_size,
        initialized to zero, for the hidden state and cell state of the LSTM
        """
        hidden = tuple(
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
            for _count in range(2)
        )

        hidden = tuple(tensor.to(DEVICE) for tensor in hidden)

        return hidden


def train_network(
    network,
    hyperparameters,
    training_parameters,
    training_loader,
    validation_loader,
    num_training,
    verbose=False,
):
    """
    """
    num_epochs = hyperparameters["num_epochs"]

    criterion = training_parameters["criterion"]

    # optimization function
    optimizer = torch.optim.Adam(network.parameters(), lr=hyperparameters["lr"])
    training_parameters["optimizer"] = optimizer

    clip_max_norm = 5

    checkpoint_filename = f'n={hyperparameters["num_most_frequent_symbols"]}_{training_parameters["datetime"]}.net'
    checkpoint_path = data_directory / checkpoint_filename
    patience = 11
    loss_delta = 0.001
    stop_early = EarlyStopping(checkpoint_path, patience, loss_delta)
    print(
        f"checkpoints of the training neural network will be saved at {checkpoint_path}"
    )

    num_epochs_length = len(str(num_epochs))

    num_batches = int(num_training / hyperparameters["batch_size"])
    num_batches_length = len(str(num_batches))

    training_parameters["average_training_losses"] = training_parameters.get(
        "average_training_losses", []
    )
    training_parameters["average_validation_losses"] = training_parameters.get(
        "average_validation_losses", []
    )

    training_parameters["epoch"] = training_parameters.get("epoch", 1)
    for epoch in range(training_parameters["epoch"], num_epochs + 1):
        training_parameters["epoch"] = epoch

        # training
        ########################################################################
        training_losses = []
        h = network.init_hidden(hyperparameters["batch_size"])

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

        average_training_loss = np.average(training_losses)
        training_parameters["average_training_losses"].append(average_training_loss)

        # validation
        ########################################################################
        validation_losses = []
        h = network.init_hidden(hyperparameters["batch_size"])

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
        training_parameters["average_validation_losses"].append(average_validation_loss)

        training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, "
        if verbose:
            training_progress += (
                f"batch {batch_number:{num_batches_length}} of {num_batches} "
            )
        training_progress += f"| average training loss: {average_training_loss:.4f}, average validation loss: {average_validation_loss:.4f}"
        print(training_progress)

        if stop_early(
            network, hyperparameters, training_parameters, average_validation_loss
        ):
            break


def load_checkpoint(checkpoint_path):
    """
    Load saved training checkpoint.
    """
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    return checkpoint


def test_network(network, hyperparameters, training_parameters, test_loader):
    """
    Calculate test loss and generate metrics.
    """
    criterion = training_parameters["criterion"]

    # initialize hidden state
    h = network.init_hidden(hyperparameters["batch_size"])

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
    network, hyperparameters, training_parameters, checkpoint_path
):
    """
    """
    checkpoint = {
        "network": network,
        "hyperparameters": hyperparameters,
        "training_parameters": training_parameters,
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

    def __call__(self, network, hyperparameters, training_parameters, validation_loss):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            print("saving initial network checkpoint...")
            print()
            save_training_checkpoint(
                network, hyperparameters, training_parameters, self.checkpoint_path
            )
            return False

        elif validation_loss <= self.min_validation_loss + self.loss_delta:
            validation_loss_improvement = self.min_validation_loss - validation_loss
            print(
                f"validation loss decreased by {validation_loss_improvement:.4f}, saving network checkpoint..."
            )
            print()
            save_training_checkpoint(
                network, hyperparameters, training_parameters, self.checkpoint_path
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


def generate_training_session(args):
    """
    """
    # hyperparameters dictionary
    hyperparameters = {}

    # hyperparameters["random_state"] = None
    # hyperparameters["random_state"] = 5
    # hyperparameters["random_state"] = 7
    # hyperparameters["random_state"] = 11
    hyperparameters["random_state"] = args.random_state

    # hyperparameters["num_most_frequent_symbols"] = 3
    # hyperparameters["num_most_frequent_symbols"] = 101
    # hyperparameters["num_most_frequent_symbols"] = 1013
    # hyperparameters["num_most_frequent_symbols"] = 10059
    # hyperparameters["num_most_frequent_symbols"] = 20147
    # hyperparameters["num_most_frequent_symbols"] = 25028
    # hyperparameters["num_most_frequent_symbols"] = 30591
    hyperparameters["num_most_frequent_symbols"] = args.num_most_frequent_symbols

    # padding or truncating length
    hyperparameters["sequence_length"] = 1000
    # hyperparameters["sequence_length"] = 2000

    hyperparameters["test_ratio"] = 0.05
    # hyperparameters["test_ratio"] = 0.1
    # hyperparameters["test_ratio"] = 0.2

    hyperparameters["validation_ratio"] = 0.05
    # hyperparameters["validation_ratio"] = 0.1
    # hyperparameters["validation_ratio"] = 0.2

    # hyperparameters["batch_size"] = 1
    # hyperparameters["batch_size"] = 4
    # hyperparameters["batch_size"] = 32
    # hyperparameters["batch_size"] = 64
    hyperparameters["batch_size"] = 128
    # hyperparameters["batch_size"] = 200
    # hyperparameters["batch_size"] = 256
    # hyperparameters["batch_size"] = 512

    hyperparameters["lr"] = 0.001
    # hyperparameters["lr"] = 0.01

    # hyperparameters["num_epochs"] = 10
    hyperparameters["num_epochs"] = 100
    # hyperparameters["num_epochs"] = 1000

    # hyperparameters["hidden_size"] = 128
    hyperparameters["hidden_size"] = 256
    # hyperparameters["hidden_size"] = 512
    # hyperparameters["hidden_size"] = 1024

    hyperparameters["num_layers"] = 1
    # hyperparameters["num_layers"] = 2

    if hyperparameters["num_layers"] == 1:
        hyperparameters["lstm_dropout_probability"] = 0
    else:
        hyperparameters["lstm_dropout_probability"] = 1 / 3
        # hyperparameters["lstm_dropout_probability"] = 1 / 4

    hyperparameters["final_dropout_probability"] = 1 / 4
    # hyperparameters["final_dropout_probability"] = 1 / 5

    # training parameters dictionary
    training_parameters = {}

    training_parameters["datetime"] = args.datetime

    # loss function
    training_parameters["criterion"] = nn.NLLLoss()

    return hyperparameters, training_parameters


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--random_state", type=int)
    argument_parser.add_argument("--num_most_frequent_symbols", type=int)
    argument_parser.add_argument("--train", action="store_true")
    argument_parser.add_argument("--test", action="store_true")
    argument_parser.add_argument("--load")
    argument_parser.add_argument("--datetime")

    args = argument_parser.parse_args()

    # DEBUG
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    # print version and environment information
    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(f"{torch.backends.cudnn.enabled=}")
    print(f"{torch.cuda.is_available()=}")
    print(f"{DEVICE=}")
    if torch.cuda.is_available():
        print(f"{torch.cuda.get_device_properties(DEVICE)}")
        # print(f"{torch.cuda.memory_summary(DEVICE)}")
    print()

    # load training checkpoint or generate new training session
    if args.load:
        checkpoint_path = pathlib.Path(args.load)
        print(f'loading training checkpoint "{checkpoint_path}"')
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        # print(network)
        hyperparameters = checkpoint["hyperparameters"]
        # pprint(hyperparameters)
        training_parameters = checkpoint["training_parameters"]
        # pprint(training_parameters)
        print()
        # sys.exit()
    else:
        hyperparameters, training_parameters = generate_training_session(args)

        # neural network instantiation
        ############################################################################
        # num_protein_letters = len(dataset.protein_letters)
        num_protein_letters = 27
        input_size = num_protein_letters

        network = Sequence_LSTM(
            input_size=input_size,
            output_size=hyperparameters["num_most_frequent_symbols"],
            hidden_size=hyperparameters["hidden_size"],
            num_layers=hyperparameters["num_layers"],
            lstm_dropout_probability=hyperparameters["lstm_dropout_probability"],
            final_dropout_probability=hyperparameters["final_dropout_probability"],
        )
        # print(network)
        # print()
        ############################################################################

        network.to(DEVICE)

    if hyperparameters["random_state"] is not None:
        torch.manual_seed(hyperparameters["random_state"])

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        hyperparameters["num_most_frequent_symbols"], hyperparameters["sequence_length"]
    )

    # split dataset into train, validation, and test datasets
    validation_size = int(hyperparameters["validation_ratio"] * len(dataset))
    test_size = int(hyperparameters["test_ratio"] * len(dataset))
    training_size = len(dataset) - validation_size - test_size

    training_dataset, validation_dataset, test_dataset = random_split(
        dataset, lengths=(training_size, validation_size, test_size)
    )

    num_training = len(training_dataset)
    num_validation = len(validation_dataset)
    num_test = len(test_dataset)
    print(
        f"dataset split to train ({num_training}), validation ({num_validation}), and test ({num_test}) datasets"
    )
    print()

    # set the batch size to the size of the smallest dataset if larger than that
    min_dataset_size = min(num_training, num_validation, num_test)
    if hyperparameters["batch_size"] > min_dataset_size:
        hyperparameters["batch_size"] = min_dataset_size

    drop_last = True
    # drop_last = False
    training_loader = DataLoader(
        training_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        drop_last=drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=True,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=hyperparameters["batch_size"],
        shuffle=False,
        drop_last=drop_last,
    )
    ############################################################################

    # train network
    if args.train:
        print(f"training neural network")
        print()

        verbose = True

        train_network(
            network,
            hyperparameters,
            training_parameters,
            training_loader,
            validation_loader,
            num_training,
            verbose,
        )

        summary_writer.flush()
        summary_writer.close()

    # test trained network
    if args.test:
        test_network(network, hyperparameters, training_parameters, test_loader)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
