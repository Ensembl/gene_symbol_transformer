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
import csv
import gzip
import itertools
import logging
import pathlib
import sys
import time

from types import SimpleNamespace

# third party imports
import Bio
import ensembl_rest
import numpy as np
import pandas as pd
import pymysql
import pytorch_lightning as pl
import requests
import torch
import torch.nn.functional as F
import torchmetrics

from Bio import SeqIO
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


# logging formats
logging_formatter_time_message = logging.Formatter(
    fmt="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging_formatter_message = logging.Formatter(fmt="%(message)s")

# set up base logger
logger = logging.getLogger("main_logger")
logger.setLevel(logging.DEBUG)
logger.propagate = False
# create console handler and add to logger
console_handler = logging.StreamHandler(sys.stderr)
console_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging_formatter_time_message)
logger.addHandler(console_handler)

data_directory = pathlib.Path("data")
main_release_sequences_directory = data_directory / "main_release_protein_sequences"
rapid_release_sequences_directory = data_directory / "rapid_release_protein_sequences"

dev_datasets_num_symbols = [3, 100, 1000]

genebuild_clades = {
    "Amphibia": "amphibians",
    "Arthropoda": "arthropods",
    "Aves": "aves",
    "Homo": "humans",
    "Insecta": "insects",
    "Lepidoptera": "lepidoptera",
    "Mammalia": "mammalia",
    "Marsupialia": "marsupials",
    "Metazoa": "metazoa",
    "Eukaryota": "non_vertebrates",
    "Viridiplantae": "plants",
    "Primates": "primates",
    "Alveolata": "protists",
    "Amoebozoa": "protists",
    "Choanoflagellida": "protists",
    "Cryptophyta": "protists",
    "Euglenozoa": "protists",
    "Fornicata": "protists",
    "Heterolobosea": "protists",
    "Parabasalia": "protists",
    "Rhizaria": "protists",
    "Stramenopiles": "protists",
    # "Sauropsida": "reptiles",
    "Crocodylia": "reptiles",
    "Lepidosauria": "reptiles",
    "Testudines": "reptiles",
    "Rodentia": "rodentia",
    "Chondrichthyes": "sharks",
    "Teleostei": "teleostei",
    "Viral": "viral",
}

get_canonical_translations_sql = """
-- gene transcripts of canonical translations
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  transcript.stable_id AS 'transcript.stable_id',
  transcript.version AS 'transcript.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version'
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
WHERE gene.biotype = 'protein_coding';
"""

get_xref_symbols_sql = """
-- Xref symbols for canonical translations
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  transcript.stable_id AS 'transcript.stable_id',
  transcript.version AS 'transcript.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'Xref_symbol',
  xref.description AS 'xref.description',
  external_db.db_display_name AS 'external_db.db_display_name'
FROM gene
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
INNER JOIN xref
  ON gene.display_xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
WHERE gene.biotype = 'protein_coding';
"""

get_entrezgene_symbols_sql = """
-- EntrezGene symbols for translations with no Xref symbols
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  transcript.stable_id AS 'transcript.stable_id',
  transcript.version AS 'transcript.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'EntrezGene_symbol',
  xref.description AS 'xref.description',
  external_db.db_display_name AS 'external_db.db_display_name'
FROM gene
INNER JOIN object_xref
  ON gene.gene_id = object_xref.ensembl_id
INNER JOIN xref
  ON object_xref.xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
WHERE gene.gene_id IN (
  -- ensembl_id of canonical translations with no Xref symbols
  SELECT
    gene.gene_id
  FROM gene
  INNER JOIN transcript
    ON gene.canonical_transcript_id = transcript.transcript_id
  INNER JOIN translation
    ON transcript.canonical_translation_id = translation.translation_id
  WHERE gene.biotype = 'protein_coding'
  AND gene.display_xref_id IS NULL
)
AND gene.biotype = 'protein_coding'
AND object_xref.ensembl_object_type = 'Gene'
AND external_db.db_name = 'EntrezGene';
"""

get_uniprot_gn_symbols_sql = """
-- Uniprot_gn symbols for translations with no Xref and no EntrezGene symbols
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  transcript.stable_id AS 'transcript.stable_id',
  transcript.version AS 'transcript.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'Uniprot_gn_symbol',
  xref.description AS 'xref.description',
  external_db.db_display_name AS 'external_db.db_display_name'
FROM gene
INNER JOIN object_xref
  ON gene.gene_id = object_xref.ensembl_id
INNER JOIN xref
  ON object_xref.xref_id = xref.xref_id
INNER JOIN external_db
  ON xref.external_db_id = external_db.external_db_id
INNER JOIN transcript
  ON gene.canonical_transcript_id = transcript.transcript_id
INNER JOIN translation
  ON transcript.canonical_translation_id = translation.translation_id
WHERE gene.gene_id IN (
  SELECT
    gene.gene_id
  FROM gene
  INNER JOIN transcript
    ON gene.canonical_transcript_id = transcript.transcript_id
  INNER JOIN translation
    ON transcript.canonical_translation_id = translation.translation_id
  WHERE gene.biotype = 'protein_coding'
  AND gene.display_xref_id IS NULL
  AND gene.gene_id NOT IN (
    -- ensembl_id of canonical translations with no Xref and no EntrezGene symbols
    SELECT
      gene.gene_id
    FROM gene
    INNER JOIN object_xref
      ON gene.gene_id = object_xref.ensembl_id
    INNER JOIN xref
      ON object_xref.xref_id = xref.xref_id
    INNER JOIN external_db
      ON xref.external_db_id = external_db.external_db_id
    INNER JOIN transcript
      ON gene.canonical_transcript_id = transcript.transcript_id
    INNER JOIN translation
      ON transcript.canonical_translation_id = translation.translation_id
    WHERE gene.gene_id IN (
      -- ensembl_id of canonical translations with no Xref symbols
      SELECT
        gene.gene_id
      FROM gene
      INNER JOIN transcript
        ON gene.canonical_transcript_id = transcript.transcript_id
      INNER JOIN translation
        ON transcript.canonical_translation_id = translation.translation_id
      WHERE gene.biotype = 'protein_coding'
      AND gene.display_xref_id IS NULL
    )
    AND gene.biotype = 'protein_coding'
    AND object_xref.ensembl_object_type = 'Gene'
    AND external_db.db_name = 'EntrezGene'
  )
)
AND gene.biotype = 'protein_coding'
AND object_xref.ensembl_object_type = 'Gene'
AND external_db.db_name = 'Uniprot_gn';
"""


class GeneSymbolClassifier(pl.LightningModule):
    """
    A fully connected neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.sequence_length = self.hparams.sequence_length
        self.padding_side = self.hparams.padding_side
        self.num_protein_letters = self.hparams.num_protein_letters
        self.num_clades = self.hparams.num_clades
        self.num_symbols = self.hparams.num_symbols
        self.num_connections = self.hparams.num_connections
        self.dropout_probability = self.hparams.dropout_probability
        self.symbol_mapper = self.hparams.symbol_mapper
        self.protein_sequence_mapper = self.hparams.protein_sequence_mapper
        self.clade_mapper = self.hparams.clade_mapper

        self.num_sample_predictions = self.hparams.num_sample_predictions

        input_size = (self.sequence_length * self.num_protein_letters) + self.num_clades
        output_size = self.num_symbols

        self.input_layer = nn.Linear(
            in_features=input_size, out_features=self.num_connections
        )
        self.dropout = nn.Dropout(self.dropout_probability)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=self.num_connections, out_features=output_size
        )

        self.final_activation = nn.LogSoftmax(dim=1)

        self.best_validation_accuracy = 0

    def forward(self, x):
        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def on_pretrain_routine_end(self):
        logger.info("start network training")
        logger.info(f"configuration:\n{self.hparams}")

    def training_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        # loss function
        training_loss = F.nll_loss(output, labels)
        self.log("training_loss", training_loss)

        return training_loss

    def on_validation_start(self):
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-and-devices
        self.validation_accuracy = torchmetrics.Accuracy(num_classes=self.num_symbols).to(
            self.device
        )

    def validation_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        # loss function
        validation_loss = F.nll_loss(output, labels)
        self.log("validation_loss", validation_loss)

        # get predicted label indexes from output
        predictions, _ = self.get_prediction_indexes_probabilities(output)

        self.validation_accuracy(predictions, labels)

        return validation_loss

    def on_validation_end(self):
        self.best_validation_accuracy = max(
            self.best_validation_accuracy,
            self.validation_accuracy.compute().item(),
        )

    def on_train_end(self):
        # NOTE: disabling saving network to TorchScript, seems buggy
        # save network in TorchScript format
        # experiment_directory_path = pathlib.Path(self.hparams.experiment_directory)
        # torchscript_path = experiment_directory_path / "torchscript_network.pt"
        # torchscript = self.to_torchscript()
        # torch.jit.save(torchscript, torchscript_path)
        pass

    def on_test_start(self):
        self.test_accuracy = torchmetrics.Accuracy().to(self.device)
        self.test_precision = torchmetrics.Precision(
            num_classes=self.num_symbols, average="macro"
        ).to(self.device)
        self.test_recall = torchmetrics.Recall(
            num_classes=self.num_symbols, average="macro"
        ).to(self.device)

        self.sample_labels = torch.empty(0).to(self.device)
        self.sample_predictions = torch.empty(0).to(self.device)

    def test_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

        # get predicted label indexes from output
        predictions, _ = self.get_prediction_indexes_probabilities(output)

        self.test_accuracy(predictions, labels)
        self.test_precision(predictions, labels)
        self.test_recall(predictions, labels)

        if self.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(len(labels))

            sample_labels = labels[permutation[0 : self.num_sample_predictions]]
            sample_predictions = predictions[permutation[0 : self.num_sample_predictions]]

            self.sample_labels = torch.cat((self.sample_labels, sample_labels))
            self.sample_predictions = torch.cat(
                (self.sample_predictions, sample_predictions)
            )

    def on_test_end(self):
        # log statistics
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        logger.info(
            f"test accuracy: {accuracy:.4f} | precision: {precision:.4f} | recall: {recall:.4f}"
        )
        logger.info(f"(best validation accuracy: {self.best_validation_accuracy:.4f})")

        if self.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(len(self.sample_labels))

            self.sample_labels = self.sample_labels[
                permutation[0 : self.num_sample_predictions]
            ].tolist()
            self.sample_predictions = self.sample_predictions[
                permutation[0 : self.num_sample_predictions]
            ].tolist()

            # change logging format to raw messages
            for handler in logger.handlers:
                handler.setFormatter(logging_formatter_message)

            labels = [
                self.symbol_mapper.index_to_label(label) for label in self.sample_labels
            ]
            assignments = [
                self.symbol_mapper.index_to_label(prediction)
                for prediction in self.sample_predictions
            ]

            logger.info("\nsample assignments")
            logger.info("assignment | true label")
            logger.info("-----------------------")
            for assignment, label in zip(assignments, labels):
                if assignment == label:
                    logger.info(f"{assignment:>10} | {label:>10}")
                else:
                    logger.info(f"{assignment:>10} | {label:>10}  !!!")

    def configure_optimizers(self):
        # optimization function
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def predict_probabilities(self, sequences, clades):
        """
        Get symbol predictions for a list of protein sequences, along with
        the probabilities of predictions.
        """
        features_tensor = self.generate_features_tensor(sequences, clades)

        # run inference
        with torch.no_grad():
            self.eval()
            output = self.forward(features_tensor)

        prediction_indexes, probabilities = self.get_prediction_indexes_probabilities(
            output
        )

        predictions = [
            self.symbol_mapper.index_to_label(prediction.item())
            for prediction in prediction_indexes
        ]

        predictions_probabilities = [
            (prediction, probability.item())
            for prediction, probability in zip(predictions, probabilities)
        ]

        return predictions_probabilities

    @staticmethod
    def get_prediction_indexes_probabilities(output):
        """
        Get predicted labels from network's forward pass output, along with
        the probabilities of predictions.
        """
        predicted_probabilities = torch.exp(output)
        # get class indexes from the one-hot encoded labels
        predictions = torch.argmax(predicted_probabilities, dim=1)
        # get max probability
        probabilities, _indices = torch.max(predicted_probabilities, dim=1)
        return (predictions, probabilities)

    def generate_features_tensor(self, sequences, clades):
        """
        Convert lists of protein sequences and species clades to an one-hot
        encoded features tensor.
        """
        padding_side_to_align = {"left": ">", "right": "<"}

        one_hot_features_list = []
        for sequence, clade in zip(sequences, clades):
            # pad or truncate sequence to be exactly `self.sequence_length` letters long
            sequence = "{string:{align}{string_length}.{truncate_length}}".format(
                string=sequence,
                align=padding_side_to_align[self.padding_side],
                string_length=self.sequence_length,
                truncate_length=self.sequence_length,
            )

            one_hot_sequence = self.protein_sequence_mapper.protein_letters_to_one_hot(
                sequence
            )
            one_hot_clade = self.clade_mapper.label_to_one_hot(clade)

            # flatten sequence matrix to a vector
            flat_one_hot_sequence = torch.flatten(one_hot_sequence)

            one_hot_features_vector = torch.cat([flat_one_hot_sequence, one_hot_clade])

            one_hot_features_list.append(one_hot_features_vector)

        one_hot_features = np.stack(one_hot_features_list)

        features_tensor = torch.from_numpy(one_hot_features)

        return features_tensor


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.
    """

    def __init__(self, configuration):
        if "num_symbols" in configuration:
            assert (
                "min_frequency" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.num_symbols}_num_symbols"
            data = load_dataset(num_symbols=configuration.num_symbols)
        elif "min_frequency" in configuration:
            assert (
                "num_symbols" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.min_frequency}_min_frequency"
            data = load_dataset(min_frequency=configuration.min_frequency)
            configuration.num_symbols = data["symbol"].nunique()
        else:
            raise KeyError(
                'missing configuration value: one of "num_symbols", "min_frequency" is required'
            )

        self.num_symbols = configuration.num_symbols

        # select the features and labels columns
        self.data = data[["sequence", "clade", "symbol", "scientific_name"]]

        if configuration.excluded_genera is not None:
            num_total_samples = len(self.data)

            for genus in configuration.excluded_genera:
                scientific_name_prefix = f"{genus} "
                self.data = self.data[
                    ~self.data["scientific_name"].str.startswith(scientific_name_prefix)
                ]
            num_used_samples = len(self.data)

            logger.info(
                f"excluded genera {configuration.excluded_genera}, using {num_used_samples} out of {num_total_samples} total samples"
            )

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.data["sequence"] = self.data["sequence"].str.pad(
                width=configuration.sequence_length,
                side=configuration.padding_side,
                fillchar=" ",
            )
            self.data["sequence"] = self.data["sequence"].str.slice(
                stop=configuration.sequence_length
            )

        # generate gene symbols CategoryMapper
        symbols = sorted(self.data["symbol"].unique().tolist())
        self.symbol_mapper = CategoryMapper(symbols)

        # generate protein sequences mapper
        self.protein_sequence_mapper = ProteinSequenceMapper()

        # generate clades CategoryMapper
        clades = sorted(set(genebuild_clades.values()))
        self.clade_mapper = CategoryMapper(clades)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_row = self.data.iloc[index].to_dict()

        sequence = data_row["sequence"]
        clade = data_row["clade"]
        symbol = data_row["symbol"]

        one_hot_sequence = self.protein_sequence_mapper.protein_letters_to_one_hot(
            sequence
        )
        # one_hot_sequence.shape: (sequence_length, num_protein_letters)

        # flatten sequence matrix to a vector
        flat_one_hot_sequence = torch.flatten(one_hot_sequence)
        # flat_one_hot_sequence.shape: (sequence_length * num_protein_letters,)

        one_hot_clade = self.clade_mapper.label_to_one_hot(clade)
        # one_hot_clade.shape: (num_clades,)

        # concatenate features to a single vector
        one_hot_features = torch.cat([flat_one_hot_sequence, one_hot_clade])
        # one_hot_features.shape: ((sequence_length * num_protein_letters) + num_clades,)

        symbol_index = self.symbol_mapper.label_to_index(symbol)

        item = one_hot_features, symbol_index

        return item


class CategoryMapper:
    """
    Categorical data mapping class, with methods to translate from the category
    text labels to one-hot encoding and vice versa.
    """

    def __init__(self, categories):
        self.categories = sorted(categories)
        self.num_categories = len(self.categories)
        self.label_to_index_dict = {
            label: index for index, label in enumerate(categories)
        }
        self.index_to_label_dict = {
            index: label for index, label in enumerate(categories)
        }

    def label_to_index(self, label):
        """
        Get the class index of label.
        """
        return self.label_to_index_dict[label]

    def index_to_label(self, index):
        """
        Get the label string from its class index.
        """
        return self.index_to_label_dict[index]

    def label_to_one_hot(self, label):
        """
        Get the one-hot representation of label.
        """
        one_hot_label = F.one_hot(
            torch.tensor(self.label_to_index_dict[label]), num_classes=self.num_categories
        )
        one_hot_label = one_hot_label.type(torch.float32)
        return one_hot_label

    def one_hot_to_label(self, one_hot_label):
        """
        Get the label string from its one-hot representation.
        """
        index = torch.argmax(one_hot_label)
        label = self.index_to_label_dict[index]
        return label


class ProteinSequenceMapper:
    """
    Class to hold the categorical data type for protein letters and a method
    to translate from protein letters to one-hot encoding.
    """

    def __init__(self):
        extended_IUPAC_protein_letters = Bio.Data.IUPACData.extended_protein_letters
        stop_codon = ["*"]
        padding_character = [" "]

        self.protein_letters = (
            list(extended_IUPAC_protein_letters) + stop_codon + padding_character
        )

        self.protein_letter_to_index = {
            protein_letter: index
            for index, protein_letter in enumerate(self.protein_letters)
        }

        self.index_to_protein_letter = {
            index: protein_letter
            for index, protein_letter in enumerate(self.protein_letters)
        }

        self.num_protein_letters = len(self.protein_letters)

    def protein_letters_to_one_hot(self, sequence):
        sequence_indexes = [
            self.protein_letter_to_index[protein_letter] for protein_letter in sequence
        ]
        one_hot_sequence = F.one_hot(
            torch.tensor(sequence_indexes), num_classes=self.num_protein_letters
        )
        one_hot_sequence = one_hot_sequence.type(torch.float32)

        return one_hot_sequence


class AttributeDict(dict):
    """
    Extended dictionary accessible with dot notation.
    """

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value


def assign_symbols(
    network,
    symbols_metadata,
    sequences_fasta,
    scientific_name=None,
    taxonomy_id=None,
    output_directory=None,
):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    sequences_fasta_path = pathlib.Path(sequences_fasta)

    if scientific_name is not None:
        taxonomy_id = get_species_taxonomy_id(scientific_name)
    clade = get_taxonomy_id_clade(taxonomy_id)
    # logger.info(f"got clade {clade} for {scientific_name}")

    if output_directory is None:
        output_directory = sequences_fasta_path.parent
    assignments_csv_path = pathlib.Path(
        f"{output_directory}/{sequences_fasta_path.stem}_symbols.csv"
    )

    # read the FASTA file in chunks and assign symbols
    with open(assignments_csv_path, "w+", newline="") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol", "probability", "description", "source"]
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(field_names)

        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            if fasta_entries[-1] is None:
                fasta_entries = [
                    fasta_entry
                    for fasta_entry in fasta_entries
                    if fasta_entry is not None
                ]

            identifiers = [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            sequences = [fasta_entry[1] for fasta_entry in fasta_entries]
            clades = [clade for _ in range(len(fasta_entries))]

            assignments_probabilities = network.predict_probabilities(sequences, clades)
            # save assignments and probabilities to the CSV file
            for identifier, (assignment, probability) in zip(
                identifiers, assignments_probabilities
            ):
                symbol_description = symbols_metadata[assignment]["description"]
                symbol_source = symbols_metadata[assignment]["source"]
                csv_writer.writerow(
                    [
                        identifier,
                        assignment,
                        probability,
                        symbol_description,
                        symbol_source,
                    ]
                )

    logger.info(f"symbol assignments saved at {assignments_csv_path}")


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


def fix_assembly(assembly):
    """
    Fixes for cases that the FASTA pep file naming doesn't mirror the assembly name.
    """
    # fix for a few assembly names
    # http://ftp.ensembl.org/pub/release-104/fasta/erinaceus_europaeus/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/homo_sapiens/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/loxodonta_africana/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/poecilia_formosa/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/sorex_araneus/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/tetraodon_nigroviridis/pep/
    # http://ftp.ensembl.org/pub/release-104/fasta/tupaia_belangeri/pep/
    names_map = {
        "eriEur1": "HEDGEHOG",
        "GRCh38.p13": "GRCh38",
        "Loxafr3.0": "loxAfr3",
        "Poecilia_formosa-5.1.2": "PoeFor_5.1.2",
        "sorAra1": "COMMON_SHREW1",
        "TETRAODON 8.0": "TETRAODON8",
        "tupBel1": "TREESHREW",
    }

    if assembly in names_map:
        return names_map[assembly]

    # remove spaces in the assembly name
    return assembly.replace(" ", "")


def download_file(file_url, file_path):
    """
    Download a file from the Ensembl HTTP server.

    Requesting the file includes a retry loop, because sometimes an erroneous
    "404 Not Found" response is issued by the server for actually existing files,
    which subsequently are correctly downloaded on a following request.

    Args:
        file_url (str): URL of the file to be downloaded
        file_path (Path or str): path to save the downloaded file
    """
    while not file_path.exists():
        response = requests.get(file_url)
        if response.ok:
            with open(file_path, "wb+") as file:
                file.write(response.content)
        else:
            logger.info(f"retrying downloading {file_url}")
            # delay retry
            time.sleep(5)


def generate_canonical_protein_sequences_fasta(assembly, ensembl_release):
    """
    Download and extract the archived protein sequences FASTA file for the species
    described in the assembly object.
    """
    if ensembl_release == "rapid_release":
        base_url = "http://ftp.ensembl.org/pub/rapid-release/species/"
        sequences_directory = rapid_release_sequences_directory
    else:
        base_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/fasta/"
        sequences_directory = main_release_sequences_directory

    # download and extract archived protein sequences FASTA file
    archived_fasta_filename = f"{assembly.fasta_filename}.gz"
    if ensembl_release == "rapid_release":
        archived_fasta_url = "{}{}/{}/geneset/{}/{}".format(
            base_url,
            assembly.species.replace(" ", "_"),
            assembly.assembly_accession,
            assembly.geneset,
            archived_fasta_filename,
        )
    else:
        archived_fasta_url = f"{base_url}{assembly.species}/pep/{archived_fasta_filename}"

    sequences_directory.mkdir(parents=True, exist_ok=True)
    archived_fasta_path = sequences_directory / archived_fasta_filename
    fasta_path = archived_fasta_path.with_suffix("")
    if ensembl_release == "rapid_release":
        canonical_fasta_filename = assembly.fasta_filename.replace(
            "pep.fa", "pep_canonical.fa"
        )
    else:
        canonical_fasta_filename = assembly.fasta_filename.replace(
            "pep.all.fa", "pep.all_canonical.fa"
        )
    canonical_fasta_path = sequences_directory / canonical_fasta_filename
    if (
        not archived_fasta_path.exists()
        or not fasta_path.exists()
        or not canonical_fasta_path.exists()
    ):
        # download archived FASTA file
        download_file(archived_fasta_url, archived_fasta_path)
        logger.info(f"downloaded {archived_fasta_filename}")

        # extract archived FASTA file
        with gzip.open(archived_fasta_path, "rb") as file:
            file_content = file.read()
        with open(fasta_path, "wb+") as file:
            file.write(file_content)
        logger.info(f"extracted {fasta_path}")

        # save FASTA file with just protein coding canonical translations
        translations_dict = fasta_to_dict(fasta_path)
        num_translations = len(translations_dict)

        if ensembl_release == "rapid_release":
            canonical_translations = get_canonical_translations(
                ensembl_core_database=assembly.core_db,
                host="mysql-ens-mirror-5",
                port=4692,
                user="ensro",
            )
        else:
            canonical_translations = get_canonical_translations(
                ensembl_core_database=assembly.core_db,
                host="ensembldb.ensembl.org",
                user="anonymous",
            )
        canonical_translations[
            "translation_stable_id_version"
        ] = canonical_translations.apply(
            lambda x: merge_stable_id_version(
                x["translation.stable_id"], x["translation.version"]
            ),
            axis=1,
        )
        canonical_translations = set(
            canonical_translations["translation_stable_id_version"].tolist()
        )

        num_canonical_translations = 0
        with open(canonical_fasta_path, "w+") as canonical_fasta_file:
            for translation_stable_id_version, values_dict in translations_dict.items():
                if translation_stable_id_version in canonical_translations:
                    description = ">" + values_dict["description"]
                    sequence = values_dict["sequence"]
                    fasta_entry = f"{description}\n{sequence}\n"
                    canonical_fasta_file.write(fasta_entry)
                    num_canonical_translations += 1

        logger.info(
            f"saved {num_canonical_translations} canonical out of {num_translations} translations sequences to {canonical_fasta_path}"
        )

    return canonical_fasta_path


def merge_stable_id_version(stable_id, version):
    if version == "None":
        stable_id_version = stable_id
    else:
        stable_id_version = f"{stable_id}.{version}"

    return stable_id_version


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


def get_assemblies_metadata():
    """
    Get metadata for all genome assemblies in the latest Ensembl release.

    The metadata are loaded from the `species_EnsemblVertebrates.txt` file of
    the latest Ensembl release and the Ensembl REST API `/info/info_genomes_assembly`
    endpoint.
    """
    assemblies_metadata_path = data_directory / "assemblies_metadata.pickle"
    if assemblies_metadata_path.exists():
        assemblies_metadata_df = pd.read_pickle(assemblies_metadata_path)
        assemblies_metadata = [
            SimpleNamespace(**values.to_dict())
            for index, values in assemblies_metadata_df.iterrows()
        ]
        return assemblies_metadata

    ensembl_release = get_ensembl_release()

    # download the `species_EnsemblVertebrates.txt` file
    species_data_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/species_EnsemblVertebrates.txt"
    data_directory.mkdir(exist_ok=True)
    species_data_path = data_directory / "species_EnsemblVertebrates.txt"
    if not species_data_path.exists():
        download_file(species_data_url, species_data_path)
        logger.info(f"downloaded {species_data_path}")

    assemblies_df = pd.read_csv(species_data_path, sep="\t", index_col=False)
    assemblies_df = assemblies_df.rename(columns={"#name": "name"})
    assemblies = [
        SimpleNamespace(**genome_row._asdict())
        for genome_row in assemblies_df.itertuples()
    ]

    logger.info("retrieving additional metadata from the Ensembl REST API")
    assemblies_metadata = []
    for assembly in assemblies:
        # skip assembly if assembly_accession is missing
        # (the value is converted to a `nan` float)
        if isinstance(assembly.assembly_accession, float):
            continue

        # delay between REST API calls
        time.sleep(0.1)

        assembly_metadata = SimpleNamespace()

        # retrieve additional information for the assembly from the REST API
        # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.info_genomes_assembly
        response = ensembl_rest.info_genomes_assembly(assembly.assembly_accession)
        rest_assembly = SimpleNamespace(**response)

        assembly_metadata.assembly_accession = assembly.assembly_accession
        assembly_metadata.scientific_name = rest_assembly.scientific_name
        assembly_metadata.common_name = rest_assembly.display_name
        assembly_metadata.taxonomy_id = assembly.taxonomy_id

        # get the Genebuild defined clade for the species
        assembly_metadata.clade = get_taxonomy_id_clade(assembly.taxonomy_id)

        assembly_metadata.species = assembly.species
        assembly_metadata.core_db = assembly.core_db

        assembly_metadata.fasta_filename = "{}.{}.pep.all.fa".format(
            assembly.species.capitalize(),
            fix_assembly(assembly.assembly),
        )

        # delay between REST API calls
        time.sleep(0.1)

        assemblies_metadata.append(assembly_metadata)

        logger.info(
            f"retrieved metadata for {assembly.name}, {rest_assembly.scientific_name}, {assembly.assembly_accession}"
        )

    assemblies_metadata_df = pd.DataFrame(
        assembly_metadata.__dict__ for assembly_metadata in assemblies_metadata
    )
    assemblies_metadata_df.to_pickle(assemblies_metadata_path)
    logger.info(f"dataset metadata saved at {assemblies_metadata_path}")

    assemblies_metadata = [
        SimpleNamespace(**values.to_dict())
        for index, values in assemblies_metadata_df.iterrows()
    ]

    return assemblies_metadata


def get_xref_canonical_translations(
    ensembl_core_database,
    host="ensembldb.ensembl.org",
    user="anonymous",
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Get canonical translation sequences from the genome assembly with
    the ensembl_core_database database at the public Ensembl MySQL server.
    """
    connection = pymysql.connect(
        host=host,
        user=user,
        database=ensembl_core_database,
        cursorclass=pymysql.cursors.DictCursor,
    )

    sql_queries = [get_xref_symbols_sql]

    if EntrezGene:
        sql_queries.append(get_entrezgene_symbols_sql)

    if Uniprot_gn:
        sql_queries.append(get_uniprot_gn_symbols_sql)

    columns = [
        "gene.stable_id",
        "gene.version",
        "transcript.stable_id",
        "transcript.version",
        "translation.stable_id",
        "translation.version",
        "Xref_symbol",
        "xref.description",
        "external_db.db_display_name",
    ]
    xref_canonical_translations_df = pd.DataFrame(columns=columns)

    canonical_translations_list = []
    with connection:
        for sql_query in sql_queries:
            with connection.cursor() as cursor:
                cursor.execute(sql_query)
                response = cursor.fetchall()
            canonical_translations_list.extend(response)

    xref_canonical_translations_df = pd.concat(
        [xref_canonical_translations_df, pd.DataFrame(canonical_translations_list)],
        ignore_index=True,
    )

    xref_canonical_translations_df["gene.version"] = xref_canonical_translations_df[
        "gene.version"
    ].astype(str)
    xref_canonical_translations_df["transcript.version"] = xref_canonical_translations_df[
        "transcript.version"
    ].astype(str)
    xref_canonical_translations_df[
        "translation.version"
    ] = xref_canonical_translations_df["translation.version"].astype(str)

    return xref_canonical_translations_df


def get_canonical_translations(ensembl_core_database, host, user, port=3306):
    """
    Get canonical transcripts of protein coding genes from the genome assembly with
    the ensembl_core_database database at the public Ensembl MySQL server.
    """
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
        database=ensembl_core_database,
        cursorclass=pymysql.cursors.DictCursor,
    )

    with connection:
        with connection.cursor() as cursor:
            cursor.execute(get_canonical_translations_sql)
            canonical_transcripts_list = cursor.fetchall()

    columns = [
        "gene.stable_id",
        "gene.version",
        "transcript.stable_id",
        "transcript.version",
        "translation.stable_id",
        "translation.version",
    ]
    canonical_transcripts_df = pd.DataFrame(canonical_transcripts_list, columns=columns)

    canonical_transcripts_df["gene.version"] = canonical_transcripts_df[
        "gene.version"
    ].astype(str)
    canonical_transcripts_df["transcript.version"] = canonical_transcripts_df[
        "transcript.version"
    ].astype(str)
    canonical_transcripts_df["translation.version"] = canonical_transcripts_df[
        "translation.version"
    ].astype(str)

    return canonical_transcripts_df


def get_ensembl_release():
    """
    retrieve the version number of the latest Ensembl release
    """
    # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.data
    ensembl_release = max(ensembl_rest.data()["releases"])
    return ensembl_release


class SuppressSettingWithCopyWarning:
    """
    Suppress SettingWithCopyWarning warning.

    https://stackoverflow.com/a/53954986
    """

    def __init__(self):
        self.original_setting = None

    def __enter__(self):
        self.original_setting = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.original_setting


def generate_dataloaders(configuration):
    """
    Generate training, validation, and test dataloaders from the dataset files.

    Args:
        configuration (AttributeDict): experiment configuration AttributeDict
    Returns:
        tuple containing the training, validation, and test dataloaders
    """
    dataset = SequenceDataset(configuration)

    configuration.symbol_mapper = dataset.symbol_mapper
    configuration.protein_sequence_mapper = dataset.protein_sequence_mapper
    configuration.clade_mapper = dataset.clade_mapper

    configuration.num_protein_letters = (
        configuration.protein_sequence_mapper.num_protein_letters
    )
    configuration.num_clades = configuration.clade_mapper.num_categories

    logger.info(
        "gene symbols:\n{}".format(pd.Series(configuration.symbol_mapper.categories))
    )

    # calculate the training_validation and test set sizes
    dataset_size = len(dataset)
    configuration.test_size = int(configuration.test_ratio * dataset_size)
    training_validation_size = dataset_size - configuration.test_size

    if configuration.test_size > 0:
        # split dataset into training_validation and test datasets
        training_validation_dataset, test_dataset = random_split(
            dataset,
            lengths=(training_validation_size, configuration.test_size),
            generator=torch.Generator().manual_seed(configuration.random_seed),
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=configuration.batch_size,
            num_workers=configuration.num_workers,
            # pin_memory=torch.cuda.is_available(),
        )

    # configuration.test_size == 0:
    else:
        training_validation_dataset = dataset
        test_dataloader = None

    # calculate the training and validation set sizes
    configuration.validation_size = int(configuration.validation_ratio * dataset_size)
    configuration.training_size = (
        dataset_size - configuration.validation_size - configuration.test_size
    )

    # split training_validation into training and validation datasets
    training_dataset, validation_dataset = random_split(
        training_validation_dataset,
        lengths=(
            configuration.training_size,
            configuration.validation_size,
        ),
        generator=torch.Generator().manual_seed(configuration.random_seed),
    )

    training_dataloader = DataLoader(
        training_dataset,
        batch_size=configuration.batch_size,
        shuffle=True,
        num_workers=configuration.num_workers,
        # pin_memory=torch.cuda.is_available(),
    )

    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
        # pin_memory=torch.cuda.is_available(),
    )

    logger.info(
        f"dataset split to training ({configuration.training_size}), validation ({configuration.validation_size}), and test ({configuration.test_size}) datasets"
    )

    return (training_dataloader, validation_dataloader, test_dataloader)


def load_dataset(num_symbols=None, min_frequency=None):
    """
    Load full dataset if none of num_symbols and min_frequency are specified.
    With num_symbols specified, load the dataset subset of the num_symbols
    most frequent symbols.
    With min_frequency specified, load the dataset subset of symbols with at least
    min_frequency sequences.

    num_symbols and min_frequency are mutually exclusive.

    Args:
        num_symbols (int): number of most frequent symbols to be included in the dataset
        min_frequency (int): minimum frequency for a symbol to be included in the dataset
    Returns:
        pandas DataFrame containing the loaded dataset
    """
    if num_symbols is not None and min_frequency is not None:
        raise ValueError(
            "num_symbols and min_frequency are mutually exclusive, please select the one to use"
        )

    full_dataset_pickle_path = data_directory / "dataset.pickle"

    if num_symbols is None and min_frequency is None:
        logger.info(f"loading full dataset {full_dataset_pickle_path} ...")
        dataset = pd.read_pickle(full_dataset_pickle_path)
        logger.info("full dataset loaded")

    elif num_symbols is not None:
        if num_symbols in dev_datasets_num_symbols:
            dataset_pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
            dataset = pd.read_pickle(dataset_pickle_path)
            logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")
        # num_symbols not in dev_datasets_num_symbols
        else:
            logger.info(
                f"loading {num_symbols} most frequent symbols samples from full dataset..."
            )
            dataset = pd.read_pickle(full_dataset_pickle_path)

            # create the dataset subset of num_symbols most frequent symbols and sequences
            symbol_counts = dataset["symbol"].value_counts()
            dataset = dataset[dataset["symbol"].isin(symbol_counts[:num_symbols].index)]

            logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")

    # min_frequency is not None
    else:
        logger.info(
            f"loading symbols with {min_frequency} minimum frequency from full dataset..."
        )
        dataset = pd.read_pickle(full_dataset_pickle_path)

        # create the dataset subset of symbols with min_frequency minimum frequency
        symbol_counts = dataset["symbol"].value_counts()
        dataset = dataset[
            dataset["symbol"].isin(symbol_counts[symbol_counts >= min_frequency].index)
        ]

        num_symbols = dataset["symbol"].nunique()
        logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")

    return dataset


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024
    return f"{num:.1f} Yi{suffix}"


def get_taxonomy_id_clade(taxonomy_id):
    """
    Get the Genebuild-defined clade for the species with taxonomy_id taxonomy ID.

    NOTE
    The function logic makes the assumption that the species' taxons are returned
    in increasing ranking from the REST API endpoint called.

    Args:
        taxonomy_id (str or int): taxonomy ID of the species to map to a clade
    Returns:
        string containing the clade of the species
    """
    taxonomy_id = str(taxonomy_id)

    homo_sapiens_taxonomy_id = "9606"
    if taxonomy_id == homo_sapiens_taxonomy_id:
        return "humans"

    # get taxonomy classification from the REST API
    # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.taxonomy_classification
    taxonomy_classification = ensembl_rest.taxonomy_classification(taxonomy_id)
    for taxon in taxonomy_classification:
        taxon_name = taxon["name"]
        if taxon_name in genebuild_clades:
            clade = genebuild_clades[taxon_name]
            break

    return clade


def get_species_taxonomy_id(scientific_name):
    """
    Get the taxonomy ID for the `scientific_name` species.

    Args:
        scientific_name (str): scientific name of the species to map to a clade
    Returns:
        string containing the taxonomy ID of the species
    """
    # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.taxonomy_name
    response = ensembl_rest.taxonomy_name(scientific_name)
    assert len(response) == 1

    taxonomy_id = response[0]["id"]

    return taxonomy_id


def log_pytorch_cuda_info():
    """
    Log PyTorch and CUDA info and device to be used.
    """
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.debug(f"{torch.__version__=}")
    logger.debug(f"{DEVICE=}")
    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")

    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")


def add_log_file_handler(
    logger, log_file_path, logging_formatter=logging_formatter_time_message
):
    """
    Create file handler and add to logger.
    """
    file_handler = logging.FileHandler(log_file_path, mode="a+")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging_formatter)
    logger.addHandler(file_handler)


class ConciseReprDict(dict):
    """
    Dictionary with a concise representation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"dictionary with {len(self.items())} items"


if __name__ == "__main__":
    print("this is a module, import to use")
