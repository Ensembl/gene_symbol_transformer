#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Raw sequences neural network pipeline.
"""


# standard library imports
import datetime
import pathlib
import pickle
import sys

# third party imports
import Bio
import numpy as np
import pandas as pd
import sklearn
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


class SequencesDataset(Dataset):
    """
    Custom Dataset for raw sequences.

    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, n, sequence_length):
        print(
            f"Loading the dataset for the {n} most frequent symbols sequences...", end=""
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
        protein_letters = get_protein_letters()
        protein_letters.sort()
        self.protein_letters_categorical_datatype = pd.CategoricalDtype(
            categories=protein_letters, ordered=True
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


def main():
    """
    main function
    """
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
    batch_size = 4
    # batch_size = 64
    # batch_size = 200
    # batch_size = 256

    dataset = SequencesDataset(n, sequence_length)
    # print(f"{len(dataset)=}")

    # sample_features, sample_labels = dataset[0]
    # print(f"{sample_features=}")
    # print(f"{sample_labels=}")
    # print()

    # split dataset into train, validation, and test datasets
    validation_ratio = 0.2
    test_ratio = 0.2
    validation_size = int(validation_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    train_size = len(dataset) - validation_size - test_size

    if RANDOM_STATE is None:
        generator = torch.default_generator
    else:
        generator = torch.Generator.manual_seed(RANDOM_STATE)
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split
    train_dataset, validation_dataset, test_dataset = random_split(
        dataset, lengths=(train_size, validation_size, test_size), generator=generator
    )

    num_train = len(train_dataset)
    num_validation = len(validation_dataset)
    num_test = len(test_dataset)
    print(
        f"dataset split to train ({num_train}), validation ({num_validation}), and test ({num_test}) datasets"
    )

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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
