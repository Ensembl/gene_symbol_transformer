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
sequences_directory = data_directory / "protein_sequences"

logging_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

dev_datasets_num_symbols = [3, 100, 1000]

genebuild_clades = {
    "Amphibia": "amphibians",
    "Arthropoda": "arthropods",
    "Aves": "aves",
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

get_xref_symbols_for_canonical_gene_transcripts_sql = """
-- Xref symbols for canonical translations
SELECT
  gene.stable_id AS 'gene.stable_id',
  gene.version AS 'gene.version',
  transcript.stable_id AS 'transcript.stable_id',
  transcript.version AS 'transcript.version',
  translation.stable_id AS 'translation.stable_id',
  translation.version AS 'translation.version',
  xref.display_label AS 'Xref_symbol',
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
    Class to hold the categorical data type for protein letters and a method
    to translate from protein letters to one-hot encoding.
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
    Class to hold the categorical data type for species clade and a method
    to translate from text labels to one-hot encoding.
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


class SequenceDataset(Dataset):
    """
    Custom Dataset for raw sequences.
    """

    def __init__(self, num_symbols=None, min_frequency=None, sequence_length=500):
        data = load_dataset(num_symbols, min_frequency)

        # select the features and labels columns
        self.data = data[["sequence", "clade", "symbol"]]

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.data["sequence"] = self.data["sequence"].str.pad(
                width=sequence_length, side="right", fillchar=" "
            )
            self.data["sequence"] = self.data["sequence"].str.slice(stop=sequence_length)

        # create categorical data mappers
        labels = self.data["symbol"].unique().tolist()
        self.gene_symbols_mapper = GeneSymbolsMapper(labels)
        self.protein_sequences_mapper = ProteinSequencesMapper()
        clades = {value for _, value in genebuild_clades.items()}
        self.clades_mapper = CladesMapper(clades)

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
            with open(file_path, "wb+") as f:
                f.write(response.content)
        else:
            # delay retry
            time.sleep(5)


def download_protein_sequences_fasta(assembly, ensembl_release):
    """
    Download and extract the archived protein sequences FASTA file for the species
    described in the assembly object.
    """
    base_url = f"http://ftp.ensembl.org/pub/release-{ensembl_release}/fasta/"

    # download and extract archived protein sequences FASTA file
    archived_fasta_filename = f"{assembly.fasta_filename}.gz"
    species = assembly.scientific_name.replace(" ", "_").lower()
    archived_fasta_url = f"{base_url}{species}/pep/{archived_fasta_filename}"
    sequences_directory.mkdir(parents=True, exist_ok=True)
    archived_fasta_path = sequences_directory / archived_fasta_filename
    fasta_path = archived_fasta_path.with_suffix("")
    if not archived_fasta_path.exists() or not fasta_path.exists():
        download_file(archived_fasta_url, archived_fasta_path)
        logger.info(f"downloaded {archived_fasta_filename}")

        with gzip.open(archived_fasta_path, "rb") as f:
            file_content = f.read()
        with open(fasta_path, "wb+") as f:
            f.write(file_content)
        logger.info(f"extracted {fasta_path}")

    return fasta_path


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

    logger.info(f"retrieving additional metadata from the Ensembl REST API")
    assemblies_metadata = []
    for assembly in assemblies:
        # skip assembly if assembly_accession is missing
        # (the value is converted to a `nan` float)
        if type(assembly.assembly_accession) is float:
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

    sql_queries = [
        get_xref_symbols_for_canonical_gene_transcripts_sql,
    ]

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

    return xref_canonical_translations_df


def get_canonical_translations(
    ensembl_core_database,
    host="ensembldb.ensembl.org",
    user="anonymous",
):
    """
    Get canonical transcripts of protein coding genes from the genome assembly with
    the ensembl_core_database database at the public Ensembl MySQL server.
    """
    connection = pymysql.connect(
        host=host,
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
        pass

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
            if not dataset_pickle_path.exists():
                logger.info(f"generating dedicated files for the dev datasets...")
                dataset = pd.read_pickle(full_dataset_pickle_path)
                save_dev_datasets(dataset=dataset)
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


def save_dev_datasets(dataset=None, num_samples=100):
    """
    Generate and save subsets of the full dataset for faster loading during development,
    the datasets as FASTA files, and FASTA files with a small number of sample sequences
    for quick reference.

    Args:
        dataset (pandas DataFrame): full dataset dataframe
        num_samples (int): number of samples to include in the samples FASTA files
    """
    if dataset is None:
        dataset = load_dataset()

    symbol_counts = dataset["symbol"].value_counts()

    for num_symbols in dev_datasets_num_symbols:
        dev_dataset = dataset[dataset["symbol"].isin(symbol_counts[:num_symbols].index)]

        # save dataframe to a pickle file
        pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
        dev_dataset.to_pickle(pickle_path)
        logger.info(
            f"{num_symbols} most frequent symbols dev dataset saved at {pickle_path}"
        )

        # save sequences to a FASTA file
        fasta_path = data_directory / f"{num_symbols}_symbols.fasta"

        dataframe_to_fasta(dev_dataset, fasta_path)
        logger.info(
            f"{num_symbols} most frequent symbols dev dataset FASTA file saved at {fasta_path}"
        )

        # pick random sample sequences
        samples = dev_dataset.sample(num_samples)
        samples = samples.sort_index()

        # save sample sequences to a FASTA file
        fasta_path = data_directory / f"{num_symbols}_symbols-{num_samples}_samples.fasta"
        dataframe_to_fasta(samples, fasta_path)
        logger.info(
            f"{num_symbols} most frequent symbols {num_samples} samples FASTA file saved at {fasta_path}"
        )


def dataframe_to_fasta(df, fasta_path):
    """
    Save a dataframe containing entries of sequences and metadata to a FASTA file.
    """
    with open(fasta_path, "w+") as fasta_file:
        for index, values in df.iterrows():
            row_dict = values.to_dict()
            description_text = "\t".join(
                f"{key}:{value}"
                for key, value in row_dict.items()
                if key not in {"Index", "sequence"}
            )

            description = ">" + description_text

            sequence = row_dict["sequence"]
            fasta_entry = f"{description}\n{sequence}\n"
            fasta_file.write(fasta_entry)


def load_checkpoint(checkpoint_path):
    """
    Load a experiment checkpoint and return the network and experiment objects.

    Args:
        checkpoint_path (path-like object): path to the saved experiment checkpoint
    Returns:
        tuple[Experiment, torch.nn.Module] with the experiment state and the classifier
    """
    logger.info(f'loading experiment checkpoint "{checkpoint_path}" ...')
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    logger.info(f'"{checkpoint_path}" experiment checkpoint loaded')

    experiment = checkpoint["experiment"]

    network = checkpoint["network"]
    network.to(DEVICE)

    return (experiment, network)


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
    print("library file, import to use")
