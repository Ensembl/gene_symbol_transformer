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
import pprint
import sys
import warnings

from types import SimpleNamespace

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from loguru import logger
from torch.utils.data import Dataset

# project imports
import dataset_generation


def specify_device():
    """
    Specify the device to run training and inference.
    """
    # use a context manager to suppress the warning:
    # UserWarning: CUDA initialization: Found no NVIDIA driver on your system.
    # NOTE
    # This warning was removed in PyTorch 1.8.0, delete the context manager after
    # upgrading to it.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DEVICE = specify_device()

data_directory = pathlib.Path("data")
experiments_directory = pathlib.Path("experiments")


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

    def __init__(self):
        stop_codon = ["*"]
        extended_IUPAC_protein_letters = Bio.Alphabet.IUPAC.ExtendedIUPACProtein.letters
        protein_letters = list(extended_IUPAC_protein_letters) + stop_codon

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

    def __init__(self, num_symbols, sequence_length):
        logger.info(f"Loading {num_symbols} most frequent symbols sequences dataset...")
        data_pickle_path = data_directory / f"most_frequent_{num_symbols}.pickle"
        data = pd.read_pickle(data_pickle_path)
        logger.info(f"{num_symbols} most frequent symbols sequences dataset loaded")
        # print()

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

        logger.info("Generating gene symbols object...")
        labels = self.data["symbol"].unique().tolist()
        self.gene_symbols = GeneSymbols(labels)
        logger.info("Gene symbols objects generated.")

        logger.info("Generating protein sequences object...")
        self.protein_sequences = ProteinSequences()
        logger.info("Protein sequences object generated.")

        # print()

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


def get_unique_protein_letters():
    """
    Generate and return a list of the unique protein letters that occur in the dataset.
    """
    extended_IUPAC_protein_letters = Bio.Alphabet.IUPAC.ExtendedIUPACProtein.letters
    stop_codon = ["*"]

    data = dataset_generation.load_data()

    # generate a list of all protein letters that occur in the dataset
    dataset_letters = set(data["sequence"].str.cat())

    extra_letters = [
        letter
        for letter in dataset_letters
        if letter not in extended_IUPAC_protein_letters
    ]
    assert extra_letters == stop_codon

    protein_letters = list(extended_IUPAC_protein_letters) + stop_codon
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


class PrettySimpleNamespace(SimpleNamespace):
    """
    Add a pretty formatting printing to the SimpleNamespace.

    NOTE
    This will most probably not be needed from Python version 3.9 on, as support
    for pretty-printing types.SimpleNamespace has been added to pprint in that version.
    """

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


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


def load_checkpoint(checkpoint_path):
    """
    Load saved training checkpoint.
    """
    logger.info(f'Loading training checkpoint "{checkpoint_path}"...')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    logger.info(f'"{checkpoint_path}" training checkpoint loaded')

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
            logger.info("saving initial network checkpoint...")
            # print()
            save_training_checkpoint(network, training_session, self.checkpoint_path)
            return False

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            logger.info(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )
            # print()
            save_training_checkpoint(network, training_session, self.checkpoint_path)
            self.min_validation_loss = validation_loss
            self.no_progress = 0
            return False

        else:
            self.no_progress += 1
            # print()

            if self.no_progress == self.patience:
                logger.info(
                    f"{self.no_progress} calls with no validation loss improvement. Stopping training."
                )
                # print()
                return True


class TrainingSession:
    def __init__(
        self,
        num_symbols,
        datetime,
        random_state,
        test_ratio,
        validation_ratio,
        sequence_length,
        batch_size,
        learning_rate,
        num_epochs,
        patience,
        loss_delta=0.001,
    ):
        # training parameters
        self.datetime = datetime
        self.random_state = random_state
        self.num_symbols = num_symbols
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

        # hyperparameters
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # training epochs and early stopping
        self.num_epochs = num_epochs
        self.num_complete_epochs = 0
        self.patience = patience
        self.loss_delta = loss_delta

        self.checkpoint_filename = f"n={self.num_symbols}_{self.datetime}.pth"

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


if __name__ == "__main__":
    print("library file, import to use")
