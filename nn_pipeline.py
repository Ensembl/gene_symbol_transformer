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
import torch.nn as nn

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, TensorDataset

# project imports


RANDOM_STATE = None
# RANDOM_STATE = 5

data_directory = pathlib.Path("data")


class BlastFeaturesDataset(Dataset):
    """
    Custom Dataset for BLAST features.

    https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset
    """

    def __init__(self, features, labels):
        """
        If the `blast_values` arrays contained in `features` are non uniform,
        i.e. of different sizes, the `features` data type will be NumPy object.
        A way to deal with this is to use a batch of size one and convert
        the NumPy arrays to PyTorch tensors one at a time.
        """
        # non uniform example features
        # self.features = features

        # uniform example features
        self.features = features
        self.labels = torch.from_numpy(labels)

        assert len(self.features) == len(self.labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        assert isinstance(index, int), f"{index=}, {type(index)=}"

        # non uniform example features
        # item = torch.from_numpy(self.features[index]), self.labels[index]

        # uniform example features
        item = self.features[index], self.labels[index]

        return item


class LSTM_Alpha(nn.Module):
    """
    An LSTM neural network for gene classification using BLAST features.
    """

    def __init__(
        self,
        num_features,
        output_size,
        hidden_size,
        num_layers,
        lstm_dropout_probability,
        final_dropout_probability,
        batch_first=True,
    ):
        """
        Initialize the model by setting up the layers.

        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.
        batch_first: If True, then the input and output tensors are provided as
        (batch, seq, feature).
        lstm_dropout_probability: If non-zero, introduces a Dropout layer on
        the outputs of each LSTM layer except the last layer, with dropout probability
        equal to dropout.
        final_dropout_probability: Probability of an element in the dropout layer
        to be zeroed.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=batch_first,
            dropout=lstm_dropout_probability,
        )

        # https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html
        self.final_dropout = nn.Dropout(final_dropout_probability)

        # https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        self.linear = nn.Linear(self.hidden_size, output_size)

        # final activation function
        # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
        # self.softmax = nn.Softmax()

    def forward(self, x, hidden_state):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        # # print(f"{x.size()=}")
        # # (batch_size, num_hits, num_features)
        # print(f"{x.type()=}")
        # # x.type()='torch.FloatTensor'
        # for h in hidden_state:
        #     print(f"{h.size()=}")
        #     # (num_layers, batch_size, hidden_size)
        #     print(f"{h.type()=}")
        #     # h.type()='torch.FloatTensor'

        output, hidden_state = self.lstm(x, hidden_state)
        # print(f"{output.size()=}")
        # output.size()=torch.Size([batch_size, num_hits, hidden_size])

        # for h in hidden_state:
        #     print(f"{h.size()=}")
        #     # h.size()=torch.Size([num_layers, batch_size, hidden_size])

        # stack up LSTM output
        # output = output.reshape(-1, self.hidden_size)
        # print(f"{output.size()=}")

        output = self.final_dropout(output)

        output = self.linear(output)

        # "nn.CrossEntropyLoss expects raw logits, so you should not apply softmax on your outputs"
        # https://discuss.pytorch.org/t/runtimeerror-multi-target-not-supported-newbie/10216/37
        # The input is expected to contain raw, unnormalized scores for each class.
        # output = self.softmax(output)
        # print(f"{output.size()=}")

        # reshape to be batch_size first
        # output = output.reshape(batch_size, -1)
        # print(f"{output.size()=}")

        # get last batch of labels
        # output = output[:, -1]
        # print(f"{output.size()=}")

        # return last output and hidden state
        return output, hidden_state

    def init_hidden(self, batch_size, gpu_available):
        """
        Initializes hidden state

        Creates two new tensors with sizes num_layers x batch_size x hidden_size,
        initialized to zero, for the hidden state and cell state of the LSTM
        """
        hidden = tuple(
            torch.zeros(self.num_layers, batch_size, self.hidden_size)
            for _count in range(2)
        )

        if gpu_available:
            hidden = tuple(tensor.cuda() for tensor in hidden)

        return hidden


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
    n = 101
    # n = 3

    # load features and labels
    print(f"Loading features and labels of {n} most frequent symbols sequences...", end="")
    blast_features_pickle_path = (
        data_directory / f"most_frequent_{n}-blast_features.pickle"
    )
    with open(blast_features_pickle_path, "rb") as f:
        blast_features = pickle.load(f)
    print(" Done.")
    print()

    # split features and labels
    features = [value["blast_values"].to_numpy() for value in blast_features.values()]
    labels = [value["one_hot_symbol"].to_numpy()[0] for value in blast_features.values()]

    # convert lists to NumPy arrays
    features = np.asarray(features)
    labels = np.asarray(labels)
    # print(f"{features.dtype=}")
    # features.dtype=dtype('O')
    # print(f"{labels.dtype=}")
    # labels.dtype=dtype('uint8')

    # number of BLAST hits to pad to or truncate to existing BLAST features
    num_hits = 150
    features = pad_truncate_blast_features(features, num_hits)
    # print(f"{features.dtype=}")
    # features.dtype=dtype('float64')

    # Cast the features array to `np.float32` data type, so that the PyTorch tensors
    # will be generated with type `torch.FloatTensor`.
    features = features.astype(np.float32)
    # print(f"{features.dtype=}")
    # features.dtype=dtype('float32')

    # Cast the labels array to `np.long` data type, so that the PyTorch tensors
    # will be generated with type `torch.LongTensor`.
    labels = labels.astype(np.long)
    # print(f"{labels.dtype=}")
    # labels.dtype=dtype('int64')

    # shuffle examples
    features, labels = sklearn.utils.shuffle(features, labels, random_state=RANDOM_STATE)

    # split examples into train, validation, and test sets
    test_size = 0.2
    # validation_size: 0.25 of the train_validation set
    validation_size = 0.2 / (1 - test_size)
    (
        train_validation_features,
        test_features,
        train_validation_labels,
        test_labels,
    ) = train_test_split(features, labels, test_size=test_size, random_state=RANDOM_STATE)
    (
        train_features,
        validation_features,
        train_labels,
        validation_labels,
    ) = train_test_split(
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
    # print()

    num_train = len(train_features)
    num_validation = len(validation_features)
    num_test = len(test_features)
    print(
        f"dataset split to {num_train} training, {num_validation} validation, and {num_test} test samples"
    )
    print()

    train_set = BlastFeaturesDataset(train_features, train_labels)
    validation_set = BlastFeaturesDataset(validation_features, validation_labels)
    test_set = BlastFeaturesDataset(test_features, test_labels)

    # batch_size = 64
    batch_size = 200
    # batch_size = 256
    drop_last = True
    # drop_last = False
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    validation_loader = DataLoader(
        validation_set, batch_size=batch_size, shuffle=True, drop_last=drop_last
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=drop_last
    )

    gpu_available = torch.cuda.is_available()

    if gpu_available:
        print("GPU is available, using it for training.")
    else:
        print("No GPU is available, training on CPU.")
    print()

    # get features and labels batch dimensions
    sample_dataiter = iter(train_loader)
    sample_features, sample_labels = sample_dataiter.next()
    # print(f"{sample_features.size()=}")
    # print(f"{sample_labels.size()=}")
    # print(f"{sample_features.type()=}")
    # sample_features.type()='torch.FloatTensor'
    # print(f"{sample_labels.type()=}")
    # sample_labels.type()='torch.ByteTensor'

    feature_batch_size = sample_features.size()
    label_batch_size = sample_labels.size()

    num_features = feature_batch_size[-1]
    output_size = label_batch_size[-1]
    # print(f"{num_features=}")
    # print(f"{output_size=}")

    hidden_size = 256
    num_layers = 2
    batch_first = True
    lstm_dropout_probability = 1 / 3
    final_dropout_probability = 1 / 5

    net = LSTM_Alpha(
        num_features=num_features,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        lstm_dropout_probability=lstm_dropout_probability,
        final_dropout_probability=final_dropout_probability,
    )
    print(net)
    print()

    # training
    ############################################################################
    print("training the neural network...")
    print()

    print(f"batch size: {batch_size}")
    print()

    # loss function
    # https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()

    # optimization function
    # https://pytorch.org/docs/stable/optim.html#torch.optim.Adam
    lr = 0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    # print(optimizer)

    clip_max_norm = 5

    statistics_output_delay = 10

    # move model to GPU, if available
    if gpu_available:
        net.cuda()

    # train for num_epochs
    net.train()
    batch_counter = 0
    num_epochs = 1000
    for epoch in range(1, num_epochs + 1):
        # initialize hidden state
        h = net.init_hidden(batch_size, gpu_available)

        # process batches
        for inputs, labels in train_loader:
            # print(f"{inputs.type()=}")
            # print(f"{labels.type()=}")

            if gpu_available:
                inputs, labels = inputs.cuda(), labels.cuda()

            # create new variables for the hidden state
            h = tuple(tensor.data for tensor in h)

            # zero accumulated gradients
            net.zero_grad()

            # get model output and hidden state
            output, h = net(inputs, h)

            # calculate the loss and perform back propagation
            # print(f"{output.size()=}")
            # output.size()=torch.Size([batch_size, n])
            # print(f"{labels.size()=}")
            # labels.size()=torch.Size([batch_size, n])
            loss = criterion(output, labels)
            loss.backward()
            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(net.parameters(), clip_max_norm)
            optimizer.step()

            # print training statistics
            batch_counter += 1
            if batch_counter == 1 or batch_counter % statistics_output_delay == 0:
                validation_loss_list = []

                # get validation loss
                validation_h = net.init_hidden(batch_size, gpu_available)

                net.eval()

                for inputs, labels in validation_loader:
                    # create new variables for the hidden state
                    validation_h = tuple(tensor.data for tensor in validation_h)

                    if gpu_available:
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, validation_h = net(inputs, validation_h)
                    validation_loss = criterion(output, labels)

                    validation_loss_list.append(validation_loss.item())

                print(f"epoch {epoch} of {num_epochs}, step {batch_counter} loss: {loss.item():.4f}, validation loss: {np.mean(validation_loss_list):.4f}")

                net.train()
    ############################################################################


def main():
    """
    main function
    """
    # DEBUG
    # pd.options.display.max_columns = None
    # pd.options.display.max_rows = None

    # print version and environment information
    print(f"{torch.__version__=}")
    print(f"{torch.version.cuda=}")
    print(f"{torch.backends.cudnn.enabled=}")
    print(f"{torch.cuda.is_available()=}")
    print()

    if RANDOM_STATE is not None:
        torch.manual_seed(RANDOM_STATE)

    train_model()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print("Interrupted with CTRL-C, exiting...")
        sys.exit()
