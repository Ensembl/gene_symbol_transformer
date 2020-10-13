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

from pprint import pprint

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_directory = pathlib.Path("data")


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.

    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, num_most_frequent_symbols, sequence_length):
        print(f"Loading dataset of the {num_most_frequent_symbols} most frequent symbols sequences...", end="")
        data_pickle_path = data_directory / f"most_frequent_{num_most_frequent_symbols}.pickle"
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
    criterion,
    training_loader,
    validation_loader,
    num_training,
    hyperparameters,
    verbose=False,
):
    """
    """
    training_parameters = {
        "network": network,
        "criterion": criterion,
        "hyperparameters": hyperparameters,
    }

    batch_size = hyperparameters["batch_size"]
    lr = hyperparameters["lr"]
    num_epochs = hyperparameters["num_epochs"]

    # optimization function
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    training_parameters["optimizer"] = optimizer

    clip_max_norm = 5


    datetime_now = datetime.datetime.now().replace(microsecond=0).isoformat()
    checkpoint_filename = f"Sequence_LSTM-{datetime_now}.net"
    checkpoint_path = data_directory / checkpoint_filename
    patience = 11
    loss_delta = 0.001
    stop_early = EarlyStopping(checkpoint_path, patience, loss_delta)
    print(f"checkpoints of the training neural network will be saved at {checkpoint_path}")

    num_epochs_length = len(str(num_epochs))

    num_batches = int(num_training / batch_size)
    num_batches_length = len(str(num_batches))

    average_training_losses = []
    average_validation_losses = []

    for epoch in range(1, num_epochs + 1):
        training_parameters["epoch"] = epoch

        # training
        ########################################################################
        training_losses = []
        h = network.init_hidden(batch_size)

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
        average_training_losses.append(average_training_loss)
        training_parameters["average_training_losses"] = average_training_losses

        # validation
        ########################################################################
        validation_losses = []
        h = network.init_hidden(batch_size)

        # set the network in evaluation mode
        network.eval()

        # disable gradient calculation
        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                # generate new variables for the hidden state
                h = tuple(tensor.data for tensor in h)

                output, h = network(inputs, h)
                labels = torch.argmax(labels, dim=1)
                validation_loss = criterion(output, labels)
                validation_losses.append(validation_loss.item())

        average_validation_loss = np.average(validation_losses)
        average_validation_losses.append(average_validation_loss)
        training_parameters["average_validation_losses"] = average_validation_losses

        training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, "
        if verbose:
            training_progress += f"batch {batch_number:{num_batches_length}} of {num_batches} "
        training_progress += f"| average training loss: {average_training_loss:.4f}, average validation loss: {average_validation_loss:.4f}"
        print(training_progress)

        if stop_early(network, training_parameters, average_validation_loss):
            break


def load_checkpoint(checkpoint_filename):
    """
    Load saved training checkpoint.
    """
    checkpoint_path = data_directory / checkpoint_filename

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    return checkpoint


def test_network(network, criterion, test_loader, batch_size):
    """
    Calculate test loss and generate metrics.
    """
    # initialize hidden state
    h = network.init_hidden(batch_size)

    network.eval()

    test_losses = []
    num_correct_predictions = 0

    with torch.no_grad():
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


def save_training_checkpoint(network, training_parameters, checkpoint_path):
    """
    """
    checkpoint = {
            "network": network,
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

    def __call__(self, network, training_parameters, validation_loss):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            print("saving initial network checkpoint...")
            save_training_checkpoint(network, training_parameters, self.checkpoint_path)
            return False

        elif validation_loss <= self.min_validation_loss + self.loss_delta:
            validation_loss_improvement = self.min_validation_loss - validation_loss
            print(f"validation loss decreased by {validation_loss_improvement:.4f}, saving network checkpoint...")
            save_training_checkpoint(network, training_parameters, self.checkpoint_path)
            self.min_validation_loss = validation_loss
            self.no_progress = 0
            return False

        else:
            self.no_progress += 1
            if self.no_progress == self.patience:
                print(f"{self.no_progress} calls with no validation loss improvement. Stopping training.")
                return True


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
    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(f"{torch.backends.cudnn.enabled=}")
    print(f"{torch.cuda.is_available()=}")
    print(f"{DEVICE=}")
    if torch.cuda.is_available():
        print(f"{torch.cuda.get_device_properties(DEVICE)}")
        # print(f"{torch.cuda.memory_summary(DEVICE)}")
        # torch.cuda.empty_cache()
    print()
    # sys.exit()

    if RANDOM_STATE is not None:
        torch.manual_seed(RANDOM_STATE)

    # num_most_frequent_symbols = 3
    # num_most_frequent_symbols = 101
    # num_most_frequent_symbols = 1013
    num_most_frequent_symbols = 10059

    # hyperparameters dictionary
    hyperparameters = {}

    # padding or truncating length
    sequence_length = 1000
    hyperparameters["sequence_length"] = sequence_length

    dataset_split_ratio = 0.1
    # dataset_split_ratio = 0.2

    test_ratio = dataset_split_ratio
    validation_ratio = dataset_split_ratio
    hyperparameters["test_ratio"] = test_ratio
    hyperparameters["validation_ratio"] = validation_ratio

    # batch_size = 1
    # batch_size = 4
    # batch_size = 64
    # batch_size = 128
    # batch_size = 200
    # batch_size = 256
    batch_size = 512
    hyperparameters["batch_size"] = batch_size

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(num_most_frequent_symbols, sequence_length)

    # split dataset into train, validation, and test datasets
    validation_size = int(validation_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    training_size = len(dataset) - validation_size - test_size

    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
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
    if batch_size > min_dataset_size:
        batch_size = min_dataset_size

    drop_last = True
    # drop_last = False
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    training_loader = DataLoader(
        training_dataset, batch_size=batch_size, shuffle=True, drop_last=drop_last
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

    output_size = num_most_frequent_symbols

    # hidden_size = 128
    # hidden_size = 256
    # hidden_size = 512
    hidden_size = 1024
    hyperparameters["hidden_size"] = hidden_size

    num_layers = 1
    # num_layers = 2
    hyperparameters["num_layers"] = num_layers

    if num_layers == 1:
        lstm_dropout_probability = 0
    else:
        lstm_dropout_probability = 1 / 3
        # lstm_dropout_probability = 1 / 4
        hyperparameters["lstm_dropout_probability"] = lstm_dropout_probability

    final_dropout_probability = 1 / 4
    # final_dropout_probability = 1 / 5
    hyperparameters["final_dropout_probability"] = final_dropout_probability

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

    network.to(DEVICE)

    # train network
    if args.train:
        lr = 0.001
        # lr = 0.01
        hyperparameters["lr"] = lr

        # num_epochs = 10
        num_epochs = 100
        # num_epochs = 1000
        hyperparameters["num_epochs"] = num_epochs

        print(f"training neural network, hyperparameters:")
        pprint(hyperparameters)
        print()

        verbose = True

        train_network(
            network,
            criterion,
            training_loader,
            validation_loader,
            num_training,
            hyperparameters,
            verbose,
        )

    # load trained network
    if args.load:
        checkpoint_filename = args.load
        print(f'loading neural network "{checkpoint_filename}"')
        checkpoint = load_checkpoint(checkpoint_filename)
        network = checkpoint["network"]
        # print(network)
        print()

    # test trained network
    if args.test:
        test_network(network, criterion, test_loader, batch_size)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
