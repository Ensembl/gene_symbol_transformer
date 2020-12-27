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
Fully connected neural network pipeline.
"""


# standard library imports
import argparse
import pathlib
import sys

# third party imports
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

# project imports
from generic_pipeline import load_checkpoint, test_network, train_network, TrainingSession,SequenceDataset


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CNN_network(nn.Module):
    """
    A convolutional neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(
        self,
        pass
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        pass

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        pass

        return output


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--random_state", type=int)
    argument_parser.add_argument("--num_most_frequent_symbols", type=int)
    argument_parser.add_argument("--train", action="store_true")
    argument_parser.add_argument("--test", action="store_true")
    argument_parser.add_argument("--load")
    argument_parser.add_argument("--datetime")

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
        print(f"{torch.cuda.device_count()=}")
        print(f"{torch.cuda.get_device_properties(DEVICE)}")
        # print(f"{torch.cuda.memory_summary(DEVICE)}")
    print()

    # load training checkpoint or generate new training session
    if args.load:
        checkpoint_path = pathlib.Path(args.load)
        print(f'Loading training checkpoint "{checkpoint_path}"...', end="")
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]
        print(" Done.")
    else:
        training_session = TrainingSession(args)

        pass

        # neural network instantiation
        ############################################################################
        pass

        # print(network)
        # print()
        ############################################################################

        network.to(DEVICE)

    if training_session.random_state is not None:
        torch.manual_seed(training_session.random_state)

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        training_session.num_most_frequent_symbols, training_session.sequence_length
    )

    # split dataset into train, validation, and test datasets
    validation_size = int(training_session.validation_ratio * len(dataset))
    test_size = int(training_session.test_ratio * len(dataset))
    training_size = len(dataset) - validation_size - test_size

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
    if training_session.batch_size > min_dataset_size:
        training_session.batch_size = min_dataset_size

    drop_last = True
    # drop_last = False
    training_loader = DataLoader(
        training_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_session.batch_size,
        shuffle=False,
        drop_last=drop_last,
    )
    ############################################################################

    print("network:")
    print(network)
    print()
    print("training_session:")
    print(training_session)
    print()

    # train network
    if args.train:
        print(f"training neural network")
        print()

        verbose = True

        train_network(
            network,
            training_session,
            training_loader,
            validation_loader,
            num_training,
            verbose,
        )

    # test trained network
    if args.test:
        test_network(network, training_session, test_loader)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
