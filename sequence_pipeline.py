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

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# project imports
import dataset_generation


# RANDOM_STATE = None
# RANDOM_STATE = 5
# RANDOM_STATE = 7
RANDOM_STATE = 11

USE_CACHE = True

data_directory = pathlib.Path("data")


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


class SequencesDataset(Dataset):
    """
    Custom Dataset for raw sequences.

    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """
    def __init__(self, n, sequence_length):
        """
        """
        self.sequence_length = sequence_length

        print(
            f"Loading the dataset for the {n} most frequent symbols sequences...", end=""
        )
        data_pickle_path = data_directory / f"most_frequent_{n}.pickle"
        data = pd.read_pickle(data_pickle_path)
        print(" Done.")
        print()

        # only the sequences as features and the symbols as labels are needed
        self.data = data[["sequence", "symbol"]]

        # generate a categorical data type for symbols
        labels = data["symbol"].unique().tolist()
        labels.sort()
        self.symbol_categorical_datatype = pd.CategoricalDtype(categories=labels, ordered=True)

        # generate a categorical data type for protein letters
        protein_letters = get_protein_letters()
        protein_letters.sort()
        self.protein_letters_categorical_datatype = pd.CategoricalDtype(categories=protein_letters, ordered=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sequence = self.data.iloc[index]["sequence"]
        symbol = self.data.iloc[index]["symbol"]

        # generate one-hot encoding of the sequence
        protein_letters_categorical = pd.Series(list(sequence), dtype=self.protein_letters_categorical_datatype)
        one_hot_sequence = pd.get_dummies(protein_letters_categorical, prefix="protein_letter")

        # generate one-hot encoding of the label (symbol)
        symbol_categorical = pd.Series(symbol, dtype=self.symbol_categorical_datatype)
        one_hot_symbol = pd.get_dummies(symbol_categorical, prefix="symbol")

        item = one_hot_sequence, one_hot_symbol

        return item


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

    # train_loader, validation_loader, test_loader = load_dataset(
    dataset = SequencesDataset(n, sequence_length)

    sample_features, sample_labels = dataset[0]
    print(f"{sample_features=}")
    print(f"{sample_labels=}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
