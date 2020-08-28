#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Neural network pipeline.
"""


# standard library imports
import pathlib
import pickle
import sys

# third party imports
import numpy as np
import pandas as pd
import sklearn
import torch

from sklearn.model_selection import train_test_split

# project imports


RANDOM_STATE = None

data_directory = pathlib.Path("data")


class BlastFeaturesDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for BLAST features.
    """
    def __init__(self, features, labels):
        """
        The features data type is NumPy object, due to the different sizes of
        its contained blast_values arrays. A way to deal with this is to use a batch
        of size one and convert the NumPy arrays to PyTorch tensors one at a time.
        """
        self.features = features
        self.labels = torch.from_numpy(labels)

        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        assert isinstance(index, int), f"{index=}, {type(index)=}"

        sample = {"features": torch.from_numpy(self.features[index]), "labels": self.labels[index]}
        return sample


def train_model():
    """
    """
    # n = 101
    n = 3

    # load features and labels
    print("Loading features and labels...")
    blast_features_pickle_path = (
        data_directory / f"most_frequent_{n}-blast_features.pickle"
    )
    with open(blast_features_pickle_path, "rb") as f:
        blast_features = pickle.load(f)

    # split features and labels
    features = [value["blast_values"].to_numpy() for value in blast_features.values()]
    labels = [value["one_hot_symbol"].to_numpy()[0] for value in blast_features.values()]

    # convert lists to NumPy arrays
    features = np.asarray(features)
    labels = np.asarray(labels)

    # shuffle examples
    features, labels = sklearn.utils.shuffle(features, labels, random_state=RANDOM_STATE)

    # split examples to training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE
    )

    num_train = len(train_features)
    num_test = len(test_features)
    print(f"dataset split to {num_train} training and {num_test} test samples")

    train_set = BlastFeaturesDataset(train_features, train_labels)
    test_set = BlastFeaturesDataset(test_features, test_labels)


def main():
    """
    main function
    """
    # DEBUG
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    train_model()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
