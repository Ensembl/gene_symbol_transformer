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

# project imports
import dataset_generation


USE_CACHE = True

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

data_directory = pathlib.Path("data")
networks_directory = pathlib.Path("networks")


class GeneSymbols:
    """
    Class to hold the categorical data type for gene symbols and methods to translate
    between text labels and one-hot encoding.
    """

    def __init__(self, labels):
        # generate a categorical data type for symbols
        labels = sorted(labels)
        self.symbol_categorical_datatype = pd.CategoricalDtype(
            categories=labels, ordered=True
        )

    def symbol_to_one_hot_encoding(self, symbol):
        symbol_categorical = pd.Series(symbol, dtype=self.symbol_categorical_datatype)
        one_hot_symbol = pd.get_dummies(symbol_categorical, prefix="symbol")

        return one_hot_symbol

    def one_hot_encoding_to_symbol(self, one_hot_symbol):
        symbol = self.symbol_categorical_datatype.categories[one_hot_symbol]
        return symbol


class ProteinSequences:
    """
    Class to hold the categorical data type for protein letters and methods to translate
    between protein letters and one-hot encoding.
    """

    def __init__(self, protein_letters):
        # generate a categorical data type for protein letters
        protein_letters = sorted(protein_letters)
        self.protein_letters_categorical_datatype = pd.CategoricalDtype(
            categories=protein_letters, ordered=True
        )

    def protein_letters_to_one_hot_encoding(self, sequence):
        protein_letters_categorical = pd.Series(
            list(sequence), dtype=self.protein_letters_categorical_datatype
        )
        one_hot_sequence = pd.get_dummies(
            protein_letters_categorical, prefix="protein_letter"
        )

        return one_hot_sequence


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

        print("Generating gene symbols object...", end="")
        labels = self.data["symbol"].unique().tolist()
        self.gene_symbols = GeneSymbols(labels)
        print(" Done.")

        print("Generating protein sequences object...", end="")
        protein_letters = get_protein_letters()
        self.protein_sequences = ProteinSequences(protein_letters)
        print(" Done.")

        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data.iloc[index]["sequence"]
        symbol = self.data.iloc[index]["symbol"]

        one_hot_sequence = self.protein_sequences.protein_letters_to_one_hot_encoding(
            sequence
        )
        one_hot_symbol = self.gene_symbols.symbol_to_one_hot_encoding(symbol)

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


def load_checkpoint(checkpoint_path, verbose=False):
    """
    Load saved training checkpoint.
    """
    if verbose:
        print(f'Loading training checkpoint "{checkpoint_path}"...', end="")

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

    if verbose:
        print(" Done.")

    return checkpoint


def save_training_checkpoint(network, training_session, checkpoint_path):
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
            checkpoint_path (path-like object): Path to save the checkpoint.
            patience (int): Number of calls to continue training if validation loss is not improving. Defaults to 7.
            loss_delta (float): Minimum change in the monitored quantity to qualify as an improvement. Defaults to 0.
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.loss_delta = loss_delta

        self.no_progress = 0
        self.min_validation_loss = np.Inf

    def __call__(self, network, training_session, validation_loss):
        if self.min_validation_loss == np.Inf:
            self.min_validation_loss = validation_loss
            print("saving initial network checkpoint...")
            print()
            save_training_checkpoint(network, training_session, self.checkpoint_path)
            return False

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            print(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )
            print()
            save_training_checkpoint(network, training_session, self.checkpoint_path)
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
                print()
                return True


class TrainingSession:
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
        # self.num_most_frequent_symbols = 26007
        # self.num_most_frequent_symbols = 27137
        # self.num_most_frequent_symbols = 28197
        # self.num_most_frequent_symbols = 29041
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
        # self.batch_size = 64
        # self.batch_size = 128
        # self.batch_size = 256
        # self.batch_size = 512
        self.batch_size = 1024

        self.learning_rate = 0.001
        # self.learning_rate = 0.01

        # self.num_epochs = 1
        # self.num_epochs = 3
        # self.num_epochs = 10
        self.num_epochs = 100
        # self.num_epochs = 1000

        self.num_complete_epochs = 0

        # larger patience for short epochs and smaller patience for longer epochs
        if self.num_most_frequent_symbols in {3, 101, 1013}:
            self.patience = 11
        else:
            self.patience = 7

        self.loss_delta = 0.001

        self.checkpoint_filename = (
            f"n={self.num_most_frequent_symbols}_{self.datetime}.pth"
        )

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


if __name__ == "__main__":
    print("library file, import to use")
