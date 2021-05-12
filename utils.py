#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# See the NOTICE file distributed with this work for additional information
# regarding copyright ownership.
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
import itertools
import pathlib
import pprint
import warnings

from types import SimpleNamespace

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from Bio import SeqIO
from loguru import logger
from torch.utils.data import Dataset


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
    return DEVICE


DEVICE = specify_device()

data_directory = pathlib.Path("data")
experiments_directory = pathlib.Path("experiments")

dev_datasets_symbol_frequency = {
    3: 342,
    100: 262,
    1059: 213,
}


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
        extended_IUPAC_protein_letters = Bio.Data.IUPACData.extended_protein_letters
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
        data = load_dataset(num_symbols)

        # only the sequences and the symbols are needed as features and labels
        self.data = data[["sequence", "symbol"]]

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
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


def read_fasta_in_chunks(fasta_file_path, num_entries_in_chunk=1024):
    """
    Read a FASTA file in chunks, returning a list of tuples of two strings,
    the FASTA description line without the leading ">" character, and
    the sequence with any whitespace removed.

    Args:
        fasta_file_path (Path or str): FASTA file path
        num_entries_in_chunk (int): number of entries in each chunk
    Returns:
        generator that produces lists of FASTA entries
    """
    # Count the number of entries in the FASTA file up to the maximum of
    # the num_entries_in_chunk chunk size. If the FASTA file has fewer entries
    # than num_entries_in_chunk, re-assign the latter to that smaller value.
    with open(fasta_file_path) as fasta_file:
        num_entries_counter = 0
        for _ in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            num_entries_counter += 1
            if num_entries_counter == num_entries_in_chunk:
                break
        else:
            num_entries_in_chunk = num_entries_counter

    # read the FASTA file in chunks
    with open(fasta_file_path) as fasta_file:
        fasta_generator = SeqIO.FastaIO.SimpleFastaParser(fasta_file)
        args = [fasta_generator] * num_entries_in_chunk
        fasta_chunks_iterator = itertools.zip_longest(*args)

        for fasta_entries in fasta_chunks_iterator:
            if fasta_entries[-1] is None:
                fasta_entries = [entry for entry in fasta_entries if entry is not None]
            yield fasta_entries


def fasta_to_dict(fasta_file_path):
    """
    Read a FASTA file to a dictionary with keys the first word of each description
    and values the corresponding sequence.

    Args:
        fasta_file_path (Path or str): FASTA file path
    Returns:
        dict: FASTA entries dictionary mapping the first word of each entry
        description to the corresponding sequence
    """
    fasta_dict = {}

    for fasta_entries in read_fasta_in_chunks(fasta_file_path):
        if fasta_entries[-1] is None:
            fasta_entries = [
                fasta_entry for fasta_entry in fasta_entries if fasta_entry is not None
            ]

        for fasta_entry in fasta_entries:
            description = fasta_entry[0]
            first_word = description.split(" ")[0]
            sequence = fasta_entry[1]

            # verify entry keys are unique
            assert first_word not in fasta_dict, f"{first_word=} already in fasta_dict"
            fasta_dict[first_word] = {"description": description, "sequence": sequence}

    return fasta_dict


def pad_or_truncate_string(string, normalized_length):
    """
    Pad or truncate string to be exactly `normalized_length` letters long.
    """
    string_length = len(string)

    if string_length <= normalized_length:
        string = " " * (normalized_length - string_length) + string
    else:
        string = string[:normalized_length]

    return string


def transform_sequences(sequences, normalized_length):
    """
    Convert a list of protein sequences to an one-hot encoded sequences tensor.
    """
    protein_sequences = ProteinSequences()

    one_hot_sequences = []
    for sequence in sequences:
        sequence = pad_or_truncate_string(sequence, normalized_length)

        one_hot_sequence = protein_sequences.protein_letters_to_one_hot_encoding(sequence)

        # convert features and labels to NumPy arrays
        one_hot_sequence = one_hot_sequence.to_numpy()

        # cast the arrays to `np.float32` data type, so that the PyTorch tensors
        # will be generated with type `torch.FloatTensor`.
        one_hot_sequence = one_hot_sequence.astype(np.float32)

        one_hot_sequences.append(one_hot_sequence)

    one_hot_sequences = np.stack(one_hot_sequences)

    one_hot_tensor_sequences = torch.from_numpy(one_hot_sequences)

    return one_hot_tensor_sequences


class PrettySimpleNamespace(SimpleNamespace):
    """
    Add a pretty formatting printing to the SimpleNamespace.

    NOTE
    This will most probably not be needed from Python version 3.9 on, as support
    for pretty-printing types.SimpleNamespace has been added to pprint in that version.
    """

    def __repr__(self):
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


def load_dataset(num_symbols=None):
    """
    Load dataset.

    Args:
        num_symbols (int): number of most frequent symbols and their sequences to load
    Returns:
        pandas DataFrame containing the loaded dataset
    """
    full_dataset_pickle_path = data_directory / "dataset.pickle"
    if num_symbols is None:
        logger.info(f"loading full dataset {full_dataset_pickle_path} ...")
        dataset = pd.read_pickle(full_dataset_pickle_path)
        logger.info("full dataset loaded")
    elif num_symbols in dev_datasets_symbol_frequency:
        dataset_pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
        dataset = pd.read_pickle(dataset_pickle_path)
        logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")
    # num_symbols not in dev_datasets_symbol_frequency
    else:
        logger.info(
            f"loading {num_symbols} most frequent symbols samples from full dataset..."
        )
        dataset = pd.read_pickle(full_dataset_pickle_path)

        # create the dataset subset of num_symbols most frequent symbols and sequences
        symbol_counts = dataset["symbol"].value_counts()
        dataset = dataset[dataset["symbol"].isin(symbol_counts[:num_symbols].index)]

        logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")

    return dataset


def load_checkpoint(checkpoint_path):
    """
    Load training checkpoint.
    """
    logger.info(f'loading training checkpoint "{checkpoint_path}" ...')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    logger.info(f'"{checkpoint_path}" training checkpoint loaded')

    return checkpoint


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024
    return f"{num:.1f} Yi{suffix}"


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
            checkpoint = {
                "network": network,
                "training_session": training_session,
            }
            torch.save(checkpoint, self.checkpoint_path)
            return False

        elif validation_loss <= self.min_validation_loss - self.loss_delta:
            validation_loss_decrease = self.min_validation_loss - validation_loss
            assert (
                validation_loss_decrease > 0
            ), f"{validation_loss_decrease=}, should be a positive number"
            logger.info(
                f"validation loss decreased by {validation_loss_decrease:.4f}, saving network checkpoint..."
            )
            checkpoint = {
                "network": network,
                "training_session": training_session,
            }
            torch.save(checkpoint, self.checkpoint_path)
            self.min_validation_loss = validation_loss
            self.no_progress = 0
            return False

        else:
            self.no_progress += 1

            if self.no_progress == self.patience:
                logger.info(
                    f"{self.no_progress} calls with no validation loss improvement. Stopping training."
                )
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
        num_connections,
        dropout_probability,
        learning_rate,
        num_epochs,
        patience,
        loss_delta=0.001,
    ):
        # dataset
        self.num_symbols = num_symbols

        # experiment parameters
        self.datetime = datetime
        self.random_state = random_state

        # test and validation sets
        self.test_ratio = test_ratio
        self.validation_ratio = validation_ratio

        # samples and batches
        self.sequence_length = sequence_length
        self.batch_size = batch_size

        # network
        self.num_connections = num_connections
        self.dropout_probability = dropout_probability
        self.learning_rate = learning_rate

        # training length and early stopping
        self.num_epochs = num_epochs
        self.num_complete_epochs = 0
        self.patience = patience
        self.loss_delta = loss_delta

        self.checkpoint_filename = f"n={self.num_symbols}_{self.datetime}.pth"

    def __repr__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


if __name__ == "__main__":
    print("library file, import to use")
