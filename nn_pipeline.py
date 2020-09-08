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


def pad_truncate_blast_features(original_features, num_hits):
    """
    Pad each sequence's BLAST features that have fewer than num_hits hits with zeros,
    and truncate to num_hits those that have more hits than that.
    """
    num_columns = original_features[0].shape[1]

    equisized_features = []
    for example_features in original_features:
        num_example_hits = len(example_features)
        if num_example_hits < num_hits:
            num_rows = num_hits - num_example_hits
            padding = np.zeros((num_rows, num_columns))
            equisized_example_features = np.vstack((example_features, padding))
        elif num_example_hits > num_hits:
            equisized_example_features = example_features[:num_hits]
        # num_example_hits == num_hits
        else:
            equisized_example_features = example_features

        equisized_features.append(equisized_example_features)

    equisized_features = np.array(equisized_features)

    return equisized_features


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

    # number of BLAST hits to pad to or truncate to existing BLAST features
    num_hits = 100
    features = pad_truncate_blast_features(features, num_hits)

    # shuffle examples
    features, labels = sklearn.utils.shuffle(features, labels, random_state=RANDOM_STATE)

    # split examples into train, validation, and test sets
    test_size = 0.2
    # validation_size: 0.25 of the train_validation set
    validation_size = 0.2 / (1 - test_size)
    train_validation_features, test_features, train_validation_labels, test_labels = train_test_split(features, labels, test_size=test_size, random_state=RANDOM_STATE)
    train_features, validation_features, train_labels, validation_labels = train_test_split(
        train_validation_features,
        train_validation_labels,
        test_size=validation_size,
        random_state=RANDOM_STATE,
    )

    # print(f"{train_features.shape=}")
    # print(f"{train_labels.shape=}")
    # print(f"{validation_features.shape=}")
    # print(f"{validation_labels.shape=}")
    # print(f"{test_features.shape=}")
    # print(f"{test_labels.shape=}")

    num_train = len(train_features)
    num_validation = len(validation_features)
    num_test = len(test_features)
    print(f"dataset split to {num_train} training, {num_validation} and {num_test} test samples")

    sys.exit()

    train_set = BlastFeaturesDataset(train_features, train_labels)
    validation_set = BlastFeaturesDataset(validation_features, validation_labels)
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
