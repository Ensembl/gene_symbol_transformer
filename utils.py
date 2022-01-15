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
import gzip
import itertools
import pathlib
import pprint
import time
import warnings

from types import SimpleNamespace

# third party imports
import Bio
import ensembl_rest
import numpy as np
import pandas as pd
import pymysql
import requests
import torch
import torch.nn.functional as F

from Bio import SeqIO
from loguru import logger
from torch import nn
from torch.utils.data import Dataset


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_directory = pathlib.Path("data")
sequences_directory = data_directory / "protein_sequences"

dev_datasets_num_symbols = [3, 100, 1000]

logging_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

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


class GeneSymbolClassifier(nn.Module):
    """
    A fully connected neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(
        self,
        sequence_length,
        padding_side,
        num_protein_letters,
        num_clades,
        num_symbols,
        num_connections,
        dropout_probability,
        symbol_mapper,
        protein_sequence_mapper,
        clade_mapper,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.padding_side = padding_side
        self.dropout_probability = dropout_probability
        self.symbol_mapper = symbol_mapper
        self.protein_sequence_mapper = protein_sequence_mapper
        self.clade_mapper = clade_mapper

        input_size = (self.sequence_length * num_protein_letters) + num_clades
        output_size = num_symbols

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        if self.dropout_probability > 0:
            self.dropout = nn.Dropout(self.dropout_probability)

        self.relu = nn.ReLU()

        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )

        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        x = self.input_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        if self.dropout_probability > 0:
            x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def predict_probabilities(self, sequences, clades):
        """
        Get symbol predictions for a list of protein sequences, along with
        the probabilities of predictions.
        """
        features_tensor = self.generate_features_tensor(sequences, clades)
        features_tensor = features_tensor.to(DEVICE)

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

    def __init__(
        self,
        num_symbols=None,
        min_frequency=None,
        sequence_length=None,
        padding_side=None,
        excluded_genera=None,
    ):
        self.num_symbols = num_symbols

        data = load_dataset(num_symbols, min_frequency)

        # select the features and labels columns
        self.data = data[["sequence", "clade", "symbol", "scientific_name"]]

        if excluded_genera is not None:
            num_total_samples = len(self.data)

            for genus in excluded_genera:
                scientific_name_prefix = f"{genus} "
                self.data = self.data[
                    ~self.data["scientific_name"].str.startswith(scientific_name_prefix)
                ]
            num_used_samples = len(self.data)

            logger.info(
                f"excluded genera {excluded_genera}, using {num_used_samples} out of {num_total_samples} total samples"
            )

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.data["sequence"] = self.data["sequence"].str.pad(
                width=sequence_length, side=padding_side, fillchar=" "
            )
            self.data["sequence"] = self.data["sequence"].str.slice(stop=sequence_length)

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
    else:
        base_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/fasta/"

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
            PrettySimpleNamespace(**values.to_dict())
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
        PrettySimpleNamespace(**genome_row._asdict())
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

        assembly_metadata = PrettySimpleNamespace()

        # retrieve additional information for the assembly from the REST API
        # https://ensemblrest.readthedocs.io/en/latest/#ensembl_rest.EnsemblClient.info_genomes_assembly
        response = ensembl_rest.info_genomes_assembly(assembly.assembly_accession)
        rest_assembly = PrettySimpleNamespace(**response)

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
        PrettySimpleNamespace(**values.to_dict())
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


class PrettySimpleNamespace(SimpleNamespace):
    """
    Add a pretty formatting printing to the SimpleNamespace.

    NOTE
    This will most probably not be needed from Python version 3.9 on, as support
    for pretty-printing types.SimpleNamespace has been added to pprint in that version.
    """

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)


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


def load_dataset(num_symbols=None, min_frequency=None):
    """
    Load full dataset if none of num_symbols and min_frequency are specified.
    With num_symbols specified, load the dataset subset of the num_symbols
    most frequent symbols.
    With min_frequency specified, load the dataset subset of symbols with
    at least min_frequency sequences.

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

        logger.info(f"{num_symbols} most frequent symbols samples dataset loaded")

    return dataset


def load_checkpoint(checkpoint_path):
    """
    Load a experiment checkpoint and return the experiment, network, optimizer objects
    and symbols metadata dictionary.


    Args:
        checkpoint_path (path-like object): path to the saved experiment checkpoint
    Returns:
        tuple[Experiment, torch.nn.Module, torch.optim.Optimizer, dict]
        containing the experiment state, classifier network, optimizer and the
        symbols metadata dictionary
    """
    logger.info(f'loading experiment checkpoint "{checkpoint_path}" ...')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    logger.info(f'"{checkpoint_path}" experiment checkpoint loaded')

    experiment = checkpoint["experiment"]

    network = GeneSymbolClassifier(
        experiment.sequence_length,
        experiment.padding_side,
        experiment.num_protein_letters,
        experiment.num_clades,
        experiment.num_symbols,
        experiment.num_connections,
        experiment.dropout_probability,
        experiment.symbol_mapper,
        experiment.protein_sequence_mapper,
        experiment.clade_mapper,
    )
    network.load_state_dict(checkpoint["network_state_dict"])
    network.to(DEVICE)

    optimizer = torch.optim.Adam(network.parameters(), lr=experiment.learning_rate)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    symbols_metadata = checkpoint["symbols_metadata"]

    return (experiment, network, optimizer, symbols_metadata)


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


if __name__ == "__main__":
    print("this is a module, import to use")
