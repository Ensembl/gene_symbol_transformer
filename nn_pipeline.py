#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Dimitrios Paraschas (dimitrios@ebi.ac.uk)


"""
Neural network pipeline.
"""


# standard library imports
import pathlib
import pickle

# third party imports
import torch

# project imports


data_directory = pathlib.Path("data")


def train_model():
    """
    """
    # n = 101
    n = 3

    # load features and labels
    print("Loading features and labels...")
    blast_features_pickle_path = (
        data_directory / f"blast_features-most_frequent_{n}.pickle"
    )
    with open(blast_features_pickle_path, "rb") as f:
        blast_features = pickle.load(f)

    print(blast_features)
    print(len(blast_features))


def main():
    """
    main function
    """
    train_model()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
