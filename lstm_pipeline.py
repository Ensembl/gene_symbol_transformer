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
LSTM neural network pipeline.
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

        # training_session.hidden_size = 128
        # training_session.hidden_size = 256
        training_session.hidden_size = 512
        # training_session.hidden_size = 1024

        training_session.num_layers = 1
        # training_session.num_layers = 2

        if training_session.num_layers == 1:
            training_session.lstm_dropout_probability = 0
        else:
            training_session.lstm_dropout_probability = 0.1
            # training_session.lstm_dropout_probability = 0.2

        training_session.final_dropout_probability = 0.1
        # training_session.final_dropout_probability = 0.2

        # loss function
        training_session.criterion = nn.NLLLoss()

        # neural network instantiation
        ############################################################################
        # num_protein_letters = len(dataset.protein_letters)
        num_protein_letters = 27
        input_size = num_protein_letters

        network = Sequence_LSTM(
            input_size=input_size,
            output_size=training_session.num_most_frequent_symbols,
            hidden_size=training_session.hidden_size,
            num_layers=training_session.num_layers,
            lstm_dropout_probability=training_session.lstm_dropout_probability,
            final_dropout_probability=training_session.final_dropout_probability,
        )
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
