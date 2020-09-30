#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Raw sequences neural network pipeline.
"""


# standard library imports
import argparse
import datetime
import pathlib
import pickle
import sys

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset, random_split

# project imports
import dataset_generation


RANDOM_STATE = None
# RANDOM_STATE = 5
# RANDOM_STATE = 7
# RANDOM_STATE = 11

USE_CACHE = True

data_directory = pathlib.Path("data")


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.

    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, n, sequence_length):
        print(
            f"Loading dataset of the {n} most frequent symbols sequences...", end=""
        )
        data_pickle_path = data_directory / f"most_frequent_{n}.pickle"
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

    def init_hidden(self, batch_size, gpu_available):
        """
        Initializes hidden state

        Creates two new tensors with sizes num_layers x batch_size x hidden_size,
        initialized to zero, for the hidden state and cell state of the LSTM
        """
        hidden = tuple(
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
            for _count in range(2)
        )

        if gpu_available:
            hidden = tuple(tensor.cuda() for tensor in hidden)

        return hidden


def get_training_progress(epoch, num_epochs, batch_counter, loss, validation_loss_list):
    """
    """
    loss_ = loss.item()
    validation_loss = np.mean(validation_loss_list)

    num_epochs_length = len(str(num_epochs))

    training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, step {batch_counter:3} | loss: {loss_:.3f}, validation loss: {validation_loss:.3f}"

    return training_progress


def train_network(
    network,
    criterion,
    train_loader,
    validation_loader,
    batch_size,
    lr,
    num_epochs,
    gpu_available,
):
    """
    """
    # optimization function
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)

    clip_max_norm = 5

    statistics_output_delay = 10

    # train for num_epochs
    network.train()
    batch_counter = 0
    for epoch in range(1, num_epochs + 1):
        # initialize hidden state
        h = network.init_hidden(batch_size, gpu_available)

        # process training examples in batches
        for inputs, labels in train_loader:
            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            # generate new variables for the hidden state
            h = tuple(tensor.data for tensor in h)

            # zero accumulated gradients
            network.zero_grad()

            # get network output and hidden state
            output, h = network(inputs, h)

            with torch.no_grad():
                # get class indexes from the labels one hot encoding
                labels = torch.argmax(labels, dim=1)

            # calculate the loss and perform back propagation
            loss = criterion(output, labels)
            # perform back propagation
            loss.backward()
            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)
            optimizer.step()

            # print training statistics
            batch_counter += 1
            if batch_counter == 1 or batch_counter % statistics_output_delay == 0:
                validation_loss_list = []

                # get validation loss
                validation_h = network.init_hidden(batch_size, gpu_available)

                network.eval()

                for inputs, labels in validation_loader:
                    # create new variables for the hidden state
                    validation_h = tuple(tensor.data for tensor in validation_h)

                    if gpu_available:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, validation_h = network(inputs, validation_h)

                    labels = torch.argmax(labels, dim=1)

                    validation_loss = criterion(output, labels)

                    validation_loss_list.append(validation_loss.item())

                training_progress = get_training_progress(epoch, num_epochs, batch_counter, loss, validation_loss_list)
                print(training_progress)

                network.train()

    # save trained network
    datetime_now = datetime.datetime.now().replace(microsecond=0).isoformat()

    network_filename = f"Sequence_LSTM-{datetime_now}.net"
    network_path = data_directory / network_filename

    torch.save(network, network_path)
    print(f"trained neural network saved at {network_path}")


def load_network(network_filename, gpu_available):
    """
    load saved network
    """
    network_path = data_directory / network_filename

    if gpu_available:
        network = torch.load(network_path)
    else:
        device = torch.device("cpu")
        network = torch.load(network_path, map_location=device)

    return network


def test_network(network, criterion, test_loader, batch_size, gpu_available):
    """
    Calculate test loss and generate metrics.
    """
    # initialize hidden state
    h = network.init_hidden(batch_size, gpu_available)

    network.eval()

    test_losses = []
    num_correct_predictions = 0
    for inputs, labels in test_loader:
        if gpu_available:
            inputs, labels = inputs.cuda(), labels.cuda()

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
    print("average test loss: {:.3f}".format(np.mean(test_losses)))

    # test predictions accuracy
    test_accuracy = num_correct_predictions / len(test_loader.dataset)
    print("test accuracy: {:.3f}".format(test_accuracy))


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--train", action="store_true")
    argument_parser.add_argument("--load")
    argument_parser.add_argument("--test", action="store_true")

    args = argument_parser.parse_args()

    # DEBUG
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    # print version and environment information
    # print(f"{torch.__version__=}")
    # print(f"{torch.version.cuda=}")
    # print(f"{torch.backends.cudnn.enabled=}")
    # print(f"{torch.cuda.is_available()=}")
    # print()

    if RANDOM_STATE is not None:
        torch.manual_seed(RANDOM_STATE)

    # n = 101
    n = 3

    sequence_length = 1000
    test_size = 0.2
    validation_size = 0.2

    # batch_size = 1
    # batch_size = 4
    # batch_size = 64
    # batch_size = 200
    batch_size = 256

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(n, sequence_length)

    # split dataset into train, validation, and test datasets
    validation_ratio = 0.2
    test_ratio = 0.2
    validation_size = int(validation_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - validation_size - test_size

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, lengths=(train_size, validation_size, test_size)
    )

    num_train = len(train_dataset)
    num_validation = len(validation_dataset)
    num_test = len(test_dataset)
    print(
        f"dataset split to train ({num_train}), validation ({num_validation}), and test ({num_test}) datasets"
    )
    print()

    drop_last = True
    # drop_last = False
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )
    ############################################################################

    # neural network instantiation
    ############################################################################
    num_protein_letters = len(dataset.protein_letters)
    input_size = num_protein_letters

    output_size = n

    # hidden_size = 128
    # hidden_size = 256
    hidden_size = 512
    # hidden_size = 1024

    num_layers = 1
    # num_layers = 2

    if num_layers == 1:
        lstm_dropout_probability = 0
    else:
        lstm_dropout_probability = 1 / 3
        # lstm_dropout_probability = 1 / 4

    final_dropout_probability = 1 / 4
    # final_dropout_probability = 1 / 5

    network = Sequence_LSTM(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lstm_dropout_probability=lstm_dropout_probability,
        final_dropout_probability=final_dropout_probability,
    )
    # print(network)
    # print()

    # loss function
    criterion = nn.NLLLoss()
    ############################################################################

    gpu_available = torch.cuda.is_available()

    # move network to GPU, if available
    if gpu_available:
        network.cuda()

    # training
    if args.train:
        print(f"training neural network, batch_size: {batch_size}")
        print()

        lr = 0.001
        # lr = 0.01

        # num_epochs = 10
        num_epochs = 100
        # num_epochs = 1000

        train_network(
            network,
            criterion,
            train_loader,
            validation_loader,
            batch_size,
            lr,
            num_epochs,
            gpu_available,
        )

    # load trained network
    if args.load:
        network_filename = args.load
        print(f'loading neural network "{network_filename}"')
        network = load_network(network_filename, gpu_available)
        # print(network)
        print()

    # testing
    if args.test:
        test_network(network, criterion, test_loader, batch_size, gpu_available)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
