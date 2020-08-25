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
import pandas as pd
import torch
import sklearn

from sklearn.model_selection import train_test_split

# project imports


RANDOM_STATE = None

data_directory = pathlib.Path("data")


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
    features = [value["blast_values"] for value in blast_features.values()]
    labels = [value["symbol"] for value in blast_features.values()]
    # num_items = 1
    num_items = 3

    # shuffle examples
    features, labels = sklearn.utils.shuffle(features, labels, random_state=RANDOM_STATE)

    # split examples to training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.3, random_state=RANDOM_STATE
    )

    num_train = len(train_features)
    num_test = len(test_features)
    print(f"dataset split to {num_train} training and {num_test} test samples")
    print()


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
