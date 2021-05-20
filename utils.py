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
General project functions and classes.
"""


# standard library imports
import itertools
import pathlib
import pprint
import sys
import warnings

from types import SimpleNamespace

# third party imports
import Bio
import numpy as np
import pandas as pd
import torch

from Bio import SeqIO
from icecream import ic
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
data_directory.mkdir(exist_ok=True)
experiments_directory = pathlib.Path("experiments")
experiments_directory.mkdir(exist_ok=True)

dev_datasets_symbol_frequency = {
    3: 342,
    100: 262,
    1059: 213,
}

genebuild_clades = {
    "Rodentia": "rodentia",
    "Primates": "primates",
    "Mammalia": "mammalia",
    "Amphibia": "amphibians",
    "Teleostei": "teleostei",
    "Marsupialia": "marsupials",
    "Aves": "aves",
    "Sauropsida": "reptiles",
    "Chondrichthyes": "sharks",
    "Eukaryota": "non_vertebrates",
    "Metazoa": "metazoa",
    "Viral": "viral",
    "Viridiplantae": "plants",
    "Arthropoda": "arthropods",
    "Lepidoptera": "lepidoptera",
    "Insecta": "insects",
    "Alveolata": "protists",
    "Amoebozoa": "protists",
    "Choanoflagellida": "protists",
    "Fornicata": "protists",
    "Euglenozoa": "protists",
    "Cryptophyta": "protists",
    "Heterolobosea": "protists",
    "Parabasalia": "protists",
    "Rhizaria": "protists",
    "Stramenopiles": "protists",
}


class GeneSymbolsMapper:
    """
    Class to hold the categorical data type for gene symbols and methods to translate
    between text labels and one-hot encoding.
    """

    def __init__(self, symbols):
        # generate a categorical data type for symbols
        self.symbols = sorted(symbols)
        self.symbol_categorical_datatype = pd.CategoricalDtype(
            categories=symbols, ordered=True
        )

    def symbol_to_one_hot(self, symbol):
        symbol_categorical = pd.Series(symbol, dtype=self.symbol_categorical_datatype)
        one_hot_symbol = pd.get_dummies(symbol_categorical, prefix="symbol")

        return one_hot_symbol

    def one_hot_to_symbol(self, one_hot_symbol):
        symbol = self.symbol_categorical_datatype.categories[one_hot_symbol]

        return symbol


class ProteinSequencesMapper:
    """
    Class to hold the categorical data type for protein letters and methods to translate
    between protein letters and one-hot encoding.
    """

    def __init__(self):
        # get unique protein letters
        stop_codon = ["*"]
        extended_IUPAC_protein_letters = Bio.Data.IUPACData.extended_protein_letters
        protein_letters = list(extended_IUPAC_protein_letters) + stop_codon
        self.protein_letters = sorted(protein_letters)

        # generate a categorical data type for protein letters
        self.protein_letters_categorical_datatype = pd.CategoricalDtype(
            categories=self.protein_letters, ordered=True
        )

    def protein_letters_to_one_hot(self, sequence):
        protein_letters_categorical = pd.Series(
            list(sequence), dtype=self.protein_letters_categorical_datatype
        )
        one_hot_sequence = pd.get_dummies(
            protein_letters_categorical, prefix="protein_letter"
        )

        return one_hot_sequence


class CladesMapper:
    """
    Class to hold the categorical data type for species clade and methods to translate
    between text labels and one-hot encoding.
    """

    def __init__(self, clades):
        # generate a categorical data type for clades
        self.clades = sorted(clades)
        self.clade_categorical_datatype = pd.CategoricalDtype(
            categories=self.clades, ordered=True
        )

    def clade_to_one_hot(self, clade):
        clade_categorical = pd.Series(clade, dtype=self.clade_categorical_datatype)
        one_hot_clade = pd.get_dummies(clade_categorical, prefix="clade")

        return one_hot_clade

    def one_hot_to_clade(self, one_hot_clade):
        clade = self.clade_categorical_datatype.categories[one_hot_clade]

        return clade


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.
    """

    def __init__(self, num_symbols, sequence_length):
        data = load_dataset(num_symbols)

        # select the features and labels columns
        self.data = data[["sequence", "clade", "symbol"]]

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.data["sequence"] = self.data["sequence"].str.pad(
                width=sequence_length, side="left", fillchar=" "
            )
            self.data["sequence"] = self.data["sequence"].str.slice(stop=sequence_length)

        logger.info("Generating gene symbols mapper...")
        labels = self.data["symbol"].unique().tolist()
        self.gene_symbols_mapper = GeneSymbolsMapper(labels)
        logger.info("Gene symbols objects generated.")

        logger.info("Generating protein sequences mapper...")
        self.protein_sequences_mapper = ProteinSequencesMapper()
        logger.info("Protein sequences mapper generated.")

        logger.info("Generating clades mapper...")
        clades = {value for _, value in genebuild_clades.items()}
        self.clades_mapper = CladesMapper(clades)
        logger.info("Clades mapper generated.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index].to_dict()

        sequence = data_row["sequence"]
        clade = data_row["clade"]
        symbol = data_row["symbol"]

        one_hot_sequence = self.protein_sequences_mapper.protein_letters_to_one_hot(
            sequence
        )
        one_hot_clade = self.clades_mapper.clade_to_one_hot(clade)
        one_hot_symbol = self.gene_symbols_mapper.symbol_to_one_hot(symbol)
        # one_hot_sequence.shape: (sequence_length, num_protein_letters)
        # one_hot_clade.shape: (num_clades,)
        # one_hot_symbol.shape: (num_symbols,)

        # convert features and labels dataframes to NumPy arrays
        one_hot_sequence = one_hot_sequence.to_numpy(dtype=np.float32)
        one_hot_clade = one_hot_clade.to_numpy(dtype=np.float32)
        one_hot_symbol = one_hot_symbol.to_numpy(dtype=np.float32)

        # flatten sequence matrix to a vector
        flat_one_hot_sequence = one_hot_sequence.flatten()
        # flat_one_hot_sequence.shape: (sequence_length * num_protein_letters,)

        # remove extra dimension for a single example
        one_hot_clade = np.squeeze(one_hot_clade)
        one_hot_symbol = np.squeeze(one_hot_symbol)

        one_hot_features = np.concatenate([flat_one_hot_sequence, one_hot_clade], axis=0)
        # one_hot_features.shape: ((sequence_length * num_protein_letters) + num_clades,)

        item = one_hot_features, one_hot_symbol

        return item


def read_fasta_in_chunks(fasta_file_path, num_chunk_entries=1024):
    """
    Read a FASTA file in chunks, returning a list of tuples of two strings,
    the FASTA description line without the leading ">" character, and
    the sequence with any whitespace removed.

    Args:
        fasta_file_path (Path or str): FASTA file path
        num_chunk_entries (int): number of entries in each chunk
    Returns:
        generator that produces lists of FASTA entries
    """
    # Count the number of entries in the FASTA file up to the maximum of
    # the num_chunk_entries chunk size. If the FASTA file has fewer entries
    # than num_chunk_entries, re-assign the latter to that smaller value.
    with open(fasta_file_path) as fasta_file:
        num_entries_counter = 0
        for _ in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            num_entries_counter += 1
            if num_entries_counter == num_chunk_entries:
                break
        else:
            num_chunk_entries = num_entries_counter

    # read the FASTA file in chunks
    with open(fasta_file_path) as fasta_file:
        fasta_generator = SeqIO.FastaIO.SimpleFastaParser(fasta_file)
        args = [fasta_generator] * num_chunk_entries
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


if __name__ == "__main__":
    print("library file, import to use")
