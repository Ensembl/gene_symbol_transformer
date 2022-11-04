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
import pandas as pd
import pymysql
import requests
import torch
import torch.nn.functional as F

from Bio import SeqIO
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
data_directory.mkdir(parents=True, exist_ok=True)
main_release_sequences_directory = data_directory / "main_release_protein_sequences"
main_release_sequences_directory.mkdir(parents=True, exist_ok=True)
rapid_release_sequences_directory = data_directory / "rapid_release_protein_sequences"
rapid_release_sequences_directory.mkdir(parents=True, exist_ok=True)

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

selected_genome_assemblies = {
    "GCA_002007445.2": ("Ailuropoda melanoleuca", "Giant panda"),
    "GCA_900496995.2": ("Aquila chrysaetos chrysaetos", "Golden eagle"),
    "GCA_009873245.2": ("Balaenoptera musculus", "Blue whale"),
    "GCA_002263795.2": ("Bos taurus", "Cow"),
    "GCA_005887515.1": ("Bos grunniens", "Domestic yak"),
    "GCA_000298355.1": ("Bos mutus", "Wild yak"),
    "GCA_000002285.2": ("Canis lupus familiaris", "Dog"),
    "GCA_900186095.1": ("Cricetulus griseus", "Chinese hamster CHOK1GS"),
    "GCA_000223135.1": ("Cricetulus griseus", "Chinese hamster CriGri"),
    "GCA_003668045.1": ("Cricetulus griseus", "Chinese hamster PICR"),
    "GCA_000951615.2": ("Cyprinus carpio", "Common carp"),
    "GCA_000002035.4": ("Danio rerio", "Zebrafish"),
    "GCA_000001215.4": ("Drosophila melanogaster", "Drosophila melanogaster"),
    "GCA_000181335.4": ("Felis catus", "Cat"),
    "GCA_000002315.5": ("Gallus gallus", "Chicken"),
    "GCA_000001405.28": ("Homo sapiens", "Human"),
    "GCA_000001905.1": ("Loxodonta africana", "Elephant"),
    "GCA_000001635.9": ("Mus musculus", "Mouse"),
    "GCA_000003625.1": ("Oryctolagus cuniculus", "Rabbit"),
    "GCA_002742125.1": ("Ovis aries", "Sheep"),
    "GCA_000001515.5": ("Pan troglodytes", "Chimpanzee"),
    "GCA_008795835.1": ("Panthera leo", "Lion"),
    "GCA_000146045.2": ("Saccharomyces cerevisiae", "Saccharomyces cerevisiae"),
    "GCA_905237065.2": ("Salmo salar", "Atlantic salmon"),
    "GCA_901001165.1": ("Salmo trutta", "Brown trout"),
    "GCA_002872995.1": ("Oncorhynchus tshawytscha", "Chinook salmon"),
    "GCA_002021735.2": ("Oncorhynchus kisutch", "Coho salmon"),
    "GCA_013265735.3": ("Oncorhynchus mykiss", "Rainbow trout"),
    "GCA_000003025.6": ("Sus scrofa", "Pig"),
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


class TrainingDataset(Dataset):
    """
    Dataset loading the original training dataset including raw sequences, clades, and
    gene symbols.
    """

    def __init__(self, configuration):
        self.configuration = configuration

        if "min_frequency" in configuration:
            configuration.dataset_id = f"{configuration.min_frequency}_min_frequency"
            data = load_dataset(min_frequency=configuration.min_frequency)
            if "num_symbols" not in configuration:
                configuration.num_symbols = data["symbol"].nunique()
        elif "num_symbols" in configuration:
            configuration.dataset_id = f"{configuration.num_symbols}_num_symbols"
            data = load_dataset(num_symbols=configuration.num_symbols)
        else:
            raise KeyError(
                'missing configuration value: one of "min_frequency", "num_symbols" is required'
            )

        self.num_symbols = configuration.num_symbols

        # select the features and labels columns
        self.dataset = data[["sequence", "clade", "symbol", "scientific_name"]]

        if configuration.excluded_genera is not None:
            num_total_samples = len(self.dataset)

            for genus in configuration.excluded_genera:
                scientific_name_prefix = f"{genus} "
                self.dataset = self.dataset[
                    ~self.dataset["scientific_name"].str.startswith(
                        scientific_name_prefix
                    )
                ]
            num_used_samples = len(self.dataset)

            logger.info(
                f"excluded genera {configuration.excluded_genera}, using {num_used_samples} out of {num_total_samples} total samples"
            )

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.dataset["sequence"] = self.dataset["sequence"].str.pad(
                width=configuration.sequence_length,
                side=configuration.padding_side,
                fillchar=" ",
            )
            self.dataset["sequence"] = self.dataset["sequence"].str.slice(
                stop=configuration.sequence_length
            )

        # generate gene symbols CategoryMapper
        symbols = sorted(self.dataset["symbol"].unique().tolist())
        self.symbol_mapper = CategoryMapper(symbols)

        # generate protein sequences mapper
        self.protein_sequence_mapper = ProteinSequenceMapper()

        # generate clades CategoryMapper
        clades = sorted(set(genebuild_clades.values()))
        self.clade_mapper = CategoryMapper(clades)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Generate a dataset item with
        (sequence label encoding, sequence one-hot encoding, clade one-hot encoding, symbol index)
        """
        dataset_row = self.dataset.iloc[index].to_dict()

        sequence = dataset_row["sequence"]
        clade = dataset_row["clade"]
        symbol = dataset_row["symbol"]

        features = generate_sequence_features(sequence, self.protein_sequence_mapper)

        if self.configuration.clade:
            one_hot_clade = self.clade_mapper.label_to_one_hot(clade)
            # one_hot_clade.shape: (num_clades,)
            clade_features = one_hot_clade
        else:
            # generate a null clade features tensor
            clade_features = torch.zeros(self.configuration.num_clades)

        symbol_index = self.symbol_mapper.label_to_index(symbol)

        features["clade_features"] = clade_features

        item = (features, symbol_index)

        return item


class InferenceDataset(Dataset):
    """
    Dataset generated from a FASTA file containing raw sequences and the species scientific name.
    """

    def __init__(self, sequences_fasta_path, clade, configuration):
        self.configuration = configuration

        identifiers = []
        sequences = []
        clades = []

        # create a DataFrame from the FASTA file
        for fasta_entries in read_fasta_in_chunks(sequences_fasta_path):
            identifiers.extend(
                [fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries]
            )
            sequences.extend([fasta_entry[1] for fasta_entry in fasta_entries])
            clades.extend([clade for _ in range(len(fasta_entries))])

        self.dataset = pd.DataFrame(
            {"identifier": identifiers, "sequence": sequences, "clade": clades}
        )

        # pad or truncate all sequences to size `sequence_length`
        with SuppressSettingWithCopyWarning():
            self.dataset["sequence"] = self.dataset["sequence"].str.pad(
                width=configuration.sequence_length,
                side=configuration.padding_side,
                fillchar=" ",
            )
            self.dataset["sequence"] = self.dataset["sequence"].str.slice(
                stop=configuration.sequence_length
            )

        self.protein_sequence_mapper = configuration.protein_sequence_mapper
        self.clade_mapper = configuration.clade_mapper

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Generate dataset item.
        """
        dataset_row = self.dataset.iloc[index].to_dict()

        identifier = dataset_row["identifier"]
        sequence = dataset_row["sequence"]
        clade = dataset_row["clade"]

        features = generate_sequence_features(sequence, self.protein_sequence_mapper)

        if self.configuration.clade:
            one_hot_clade = self.clade_mapper.label_to_one_hot(clade)
            # one_hot_clade.shape: (num_clades,)
            clade_features = one_hot_clade
        else:
            # generate a null clade features tensor
            clade_features = torch.zeros(self.configuration.num_clades)

        features["clade_features"] = clade_features

        item = (features, identifier)

        return item


def generate_sequence_features(
    sequence: str,
    protein_sequence_mapper,
    sequence_length: int = None,
    padding_side: str = None,
):
    """
    Generate features for a protein sequence.

    Args:
        sequence: a protein sequence
        protein_sequence_mapper: ProteinSequenceMapper,
        sequence_length: the length to pad or truncate the sequence
        padding_side: the side to pad the sequence if shorter than sequence_length,
            one of ["left", "right"]
    """
    if sequence_length and padding_side:
        sequence = normalize_string_length(sequence, sequence_length, padding_side)

    label_encoded_sequence = protein_sequence_mapper.sequence_to_label_encoding(
        sequence
    )
    # label_encoded_sequence.shape: (sequence_length,)

    one_hot_sequence = protein_sequence_mapper.sequence_to_one_hot(sequence)
    # one_hot_sequence.shape: (sequence_length, num_protein_letters)

    # flatten sequence matrix to a vector
    flat_one_hot_sequence = torch.flatten(one_hot_sequence)
    # flat_one_hot_sequence.shape: (sequence_length * num_protein_letters,)

    sequence_features = {
        "label_encoded_sequence": label_encoded_sequence,
        "flat_one_hot_sequence": flat_one_hot_sequence,
    }

    return sequence_features


def normalize_string_length(string: str, length: int, padding_side: str):
    """
    Normalize a string length by padding or truncating it to be exactly `length`
    characters long.

    Args:
        string: a string
        length: the length to pad or truncate the sequence
        padding_side: the side to pad the sequence if shorter than length
    """
    if len(string) == length:
        return string

    padding_side_to_align = {"left": ">", "right": "<"}

    # pad or truncate the string to be exactly `length` characters long
    string = "{string:{align}{string_length}.{truncate_length}}".format(
        string=string,
        align=padding_side_to_align[padding_side],
        string_length=length,
        truncate_length=length,
    )

    return string


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
            torch.tensor(self.label_to_index_dict[label]),
            num_classes=self.num_categories,
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

    def sequence_to_one_hot(self, sequence):
        sequence_indexes = [
            self.protein_letter_to_index[protein_letter] for protein_letter in sequence
        ]
        one_hot_sequence = F.one_hot(
            torch.tensor(sequence_indexes), num_classes=self.num_protein_letters
        )
        one_hot_sequence = one_hot_sequence.type(torch.float32)

        return one_hot_sequence

    def sequence_to_label_encoding(self, sequence):
        label_encoded_sequence = [
            self.protein_letter_to_index[protein_letter] for protein_letter in sequence
        ]

        label_encoded_sequence = torch.tensor(label_encoded_sequence, dtype=torch.int32)

        return label_encoded_sequence


class ConciseReprDict(dict):
    """
    Dictionary with a concise representation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"dictionary with {len(self.items())} items"


def assign_symbols(
    trainer,
    network,
    sequences_fasta,
    scientific_name,
    output_directory=None,
):
    """
    Use the trained network to assign symbols to the sequences in the FASTA file.
    """
    start_time = time.time()

    configuration = network.hparams

    sequences_fasta_path = pathlib.Path(sequences_fasta)
    symbols_metadata = configuration.symbols_metadata

    taxonomy_id = get_species_taxonomy_id(scientific_name)
    clade = get_taxonomy_id_clade(taxonomy_id)
    logger.debug(f"got clade {clade} for {scientific_name}")

    if output_directory is None:
        output_directory = sequences_fasta_path.parent
    assignments_csv_path = pathlib.Path(
        f"{output_directory}/{sequences_fasta_path.stem}_symbols.csv"
    )

    predict_dataset = InferenceDataset(sequences_fasta_path, clade, configuration)

    predict_dataloader = DataLoader(
        predict_dataset,
        batch_size=configuration.batch_size,
        num_workers=configuration.num_workers,
        # pin_memory=torch.cuda.is_available(),
    )

    batches_predictions = trainer.predict(network, dataloaders=predict_dataloader)

    predictions = []
    for batch_predictions in batches_predictions:
        for prediction in batch_predictions:
            predictions.append(prediction)

    with open(assignments_csv_path, "w+", newline="") as csv_file:
        # generate a csv writer, create the CSV file with a header
        field_names = ["stable_id", "symbol", "probability", "description", "source"]
        csv_writer = csv.writer(csv_file, delimiter="\t", lineterminator="\n")
        csv_writer.writerow(field_names)

        # save assignments and probabilities to the CSV file
        for prediction in predictions:
            identifier, assignment, probability = prediction
            # probability = str(round(probability, 4))
            probability = f"{probability:.4f}"
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

    end_time = time.time()
    assignment_time = end_time - start_time

    logger.info(
        f"symbol assignments generated in {assignment_time:.1f} seconds, saved at {assignments_csv_path}"
    )

    return assignment_time


def evaluate_network(trainer, network, checkpoint_path, complete=False):
    """
    Evaluate a trained network by assigning gene symbols to the protein sequences
    of genome assemblies in the latest Ensembl release, and comparing them to the existing
    Xref assignments.

    Args:
        trainer (pl.Trainer): PyTorch Lightning Trainer object
        network (GST): GST network object
        checkpoint_path (Path): path to the experiment checkpoint
        complete (bool): Whether or not to run the evaluation for all genome assemblies.
            Defaults to False, which runs the evaluation only for a selection of
            the most important species genome assemblies.
    """
    evaluation_directory_path = (
        checkpoint_path.parent / f"{checkpoint_path.stem}_evaluation"
    )

    configuration = network.hparams

    symbols_set = set(
        symbol.lower() for symbol in configuration.symbol_mapper.categories
    )

    assemblies = get_assemblies_metadata()
    comparison_statistics_list = []
    for assembly in assemblies:
        if (
            not complete
            and assembly.assembly_accession not in selected_genome_assemblies
        ):
            continue

        canonical_fasta_filename = assembly.fasta_filename.replace(
            "pep.all.fa", "pep.all_canonical.fa"
        )
        canonical_fasta_path = (
            main_release_sequences_directory / canonical_fasta_filename
        )

        # assign symbols
        assignments_csv_path = (
            evaluation_directory_path / f"{canonical_fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {canonical_fasta_path}")

            assignment_time = assign_symbols(
                trainer,
                network,
                canonical_fasta_path,
                scientific_name=assembly.scientific_name,
                output_directory=evaluation_directory_path,
            )

        comparisons_csv_path = (
            evaluation_directory_path / f"{assignments_csv_path.stem}_compare.csv"
        )
        if not comparisons_csv_path.exists():
            comparison_successful = compare_with_database(
                assignments_csv_path,
                assembly.core_db,
                assembly.scientific_name,
                symbols_set,
            )
            if not comparison_successful:
                continue

        comparison_statistics = get_comparison_statistics(comparisons_csv_path)
        comparison_statistics["scientific_name"] = assembly.scientific_name
        comparison_statistics["taxonomy_id"] = assembly.taxonomy_id
        comparison_statistics["clade"] = assembly.clade
        comparison_statistics["assignment_time"] = assignment_time

        comparison_statistics_list.append(comparison_statistics)

        message = "{}: {} assignments, {} exact matches ({:.2f}%), {} fuzzy matches ({:.2f}%), {} total matches ({:.2f}%), {:.1f} sec assignment time".format(
            comparison_statistics["scientific_name"],
            comparison_statistics["num_assignments"],
            comparison_statistics["num_exact_matches"],
            comparison_statistics["matching_percentage"],
            comparison_statistics["num_fuzzy_matches"],
            comparison_statistics["fuzzy_percentage"],
            comparison_statistics["num_total_matches"],
            comparison_statistics["total_matches_percentage"],
            comparison_statistics["assignment_time"],
        )
        logger.info(message)

    dataframe_columns = [
        "clade",
        "scientific_name",
        "num_assignments",
        "num_exact_matches",
        "matching_percentage",
        "num_fuzzy_matches",
        "fuzzy_percentage",
        "num_total_matches",
        "total_matches_percentage",
        "assignment_time",
    ]
    comparison_statistics = pd.DataFrame(
        comparison_statistics_list,
        columns=dataframe_columns,
    )

    clade_groups = comparison_statistics.groupby(["clade"])
    clade_groups_statistics = []
    aggregated_statistics = []
    for clade, group in clade_groups:
        with pd.option_context("display.float_format", "{:.2f}".format):
            group_string = group.to_string(index=False)

        num_assignments_sum = group["num_assignments"].sum()
        num_exact_matches_sum = group["num_exact_matches"].sum()
        num_fuzzy_matches_sum = group["num_fuzzy_matches"].sum()
        num_total_matches_sum = num_exact_matches_sum + num_fuzzy_matches_sum

        matching_percentage_weighted_average = (
            num_exact_matches_sum / num_assignments_sum
        ) * 100
        fuzzy_percentage_weighted_average = (
            num_fuzzy_matches_sum / num_assignments_sum
        ) * 100
        total_percentage_weighted_average = (
            num_total_matches_sum / num_assignments_sum
        ) * 100

        assignment_time_weighted_average = group["assignment_time"].mean()

        averages_message = "{} weighted averages: {:.2f}% exact matches, {:.2f}% fuzzy matches, {:.2f}% total matches, {:.1f} sec assignment time".format(
            clade,
            matching_percentage_weighted_average,
            fuzzy_percentage_weighted_average,
            total_percentage_weighted_average,
            assignment_time_weighted_average,
        )

        aggregated_statistics.append(
            {
                "clade": clade,
                "exact matches": matching_percentage_weighted_average,
                "fuzzy matches": fuzzy_percentage_weighted_average,
                "total matches": total_percentage_weighted_average,
                "assignment time": assignment_time_weighted_average,
            }
        )

        clade_statistics = f"{group_string}\n{averages_message}"

        clade_groups_statistics.append(clade_statistics)

    comparison_statistics_string = "comparison statistics:\n"
    comparison_statistics_string += "\n\n".join(
        clade_statistics for clade_statistics in clade_groups_statistics
    )
    logger.info(comparison_statistics_string)

    aggregated_statistics = pd.DataFrame(aggregated_statistics)
    logger.info(
        "\nclade weighted averages:\n{}".format(
            aggregated_statistics.to_string(
                index=False,
                formatters={
                    "exact matches": "{:.2f}%".format,
                    "fuzzy matches": "{:.2f}%".format,
                    "total matches": "{:.2f}%".format,
                    "assignment time": "{:.1f}".format,
                },
            )
        )
    )


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
        archived_fasta_url = (
            f"{base_url}{assembly.species}/pep/{archived_fasta_filename}"
        )

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
        try:
            response = ensembl_rest.info_genomes_assembly(assembly.assembly_accession)
        # handle missing genome assembly accessions in REST
        except ensembl_rest.HTTPError as ex:
            error_code = ex.response.status_code
            error_message = ex.response.json()["error"]
            if (error_code == 400) and ("not found" in error_message):
                logger.error(
                    f"Error: assembly with accession {assembly.assembly_accession} missing from the REST API"
                )
                continue
            else:
                raise

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
        with connection.cursor() as cursor:
            for sql_query in sql_queries:
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
    xref_canonical_translations_df[
        "transcript.version"
    ] = xref_canonical_translations_df["transcript.version"].astype(str)
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
    dataset = TrainingDataset(configuration)

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


def is_exact_match(symbol_a, symbol_b):
    symbol_a = symbol_a.lower()
    symbol_b = symbol_b.lower()

    if symbol_a == symbol_b:
        return "exact_match"
    else:
        return "no_exact_match"


def is_fuzzy_match(symbol_a, symbol_b):
    symbol_a = symbol_a.lower()
    symbol_b = symbol_b.lower()

    if symbol_a == symbol_b:
        return "no_fuzzy_match"

    if (symbol_a in symbol_b) or (symbol_b in symbol_a):
        return "fuzzy_match"
    else:
        return "no_fuzzy_match"


def is_known_symbol(symbol, symbols_set):
    symbol = symbol.lower()

    if symbol in symbols_set:
        return "known"
    else:
        return "unknown"


def compare_with_database(
    assignments_csv,
    ensembl_database,
    scientific_name=None,
    symbols_set=None,
    EntrezGene=False,
    Uniprot_gn=False,
):
    """
    Compare classifier assignments with the gene symbols in the genome assembly
    ensembl_database core database on the public Ensembl MySQL server.
    """
    assignments_csv_path = pathlib.Path(assignments_csv)

    canonical_translations = get_xref_canonical_translations(
        ensembl_database, EntrezGene=EntrezGene, Uniprot_gn=Uniprot_gn
    )

    if len(canonical_translations) == 0:
        if scientific_name is None:
            logger.info("0 canonical translations retrieved, nothing to compare")
        else:
            logger.info(
                f"{scientific_name}: 0 canonical translations retrieved, nothing to compare"
            )
        return False

    comparisons = []
    with open(assignments_csv_path, "r", newline="") as assignments_file:
        csv_reader = csv.reader(assignments_file, delimiter="\t")
        _csv_field_names = next(csv_reader)

        for csv_row in csv_reader:
            csv_stable_id = csv_row[0]
            classifier_symbol = csv_row[1]
            probability = csv_row[2]

            translation_stable_id = csv_stable_id.split(".")[0]

            if (
                translation_stable_id
                in canonical_translations["translation.stable_id"].values
            ):
                xref_symbol = canonical_translations.loc[
                    canonical_translations["translation.stable_id"]
                    == translation_stable_id,
                    "Xref_symbol",
                ].values[0]
                comparisons.append(
                    (csv_stable_id, xref_symbol, classifier_symbol, probability)
                )

    dataframe_columns = [
        "csv_stable_id",
        "xref_symbol",
        "classifier_symbol",
        "probability",
    ]
    compare_df = pd.DataFrame(comparisons, columns=dataframe_columns)

    compare_df["exact_match"] = compare_df.apply(
        lambda x: is_exact_match(x["classifier_symbol"], x["xref_symbol"]),
        axis=1,
        result_type="reduce",
    )

    compare_df["fuzzy_match"] = compare_df.apply(
        lambda x: is_fuzzy_match(x["classifier_symbol"], x["xref_symbol"]),
        axis=1,
        result_type="reduce",
    )

    if symbols_set:
        compare_df["known_symbol"] = compare_df.apply(
            lambda x: is_known_symbol(x["xref_symbol"], symbols_set),
            axis=1,
            result_type="reduce",
        )

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    compare_df.to_csv(comparisons_csv_path, sep="\t", index=False)

    return True


def get_comparison_statistics(comparisons_csv_path):
    compare_df = pd.read_csv(comparisons_csv_path, sep="\t", index_col=False)

    num_assignments = len(compare_df)

    if num_assignments > 0:
        num_exact_matches = len(compare_df[compare_df["exact_match"] == "exact_match"])
        num_fuzzy_matches = len(compare_df[compare_df["fuzzy_match"] == "fuzzy_match"])

        matching_percentage = (num_exact_matches / num_assignments) * 100
        fuzzy_percentage = (num_fuzzy_matches / num_assignments) * 100
        num_total_matches = num_exact_matches + num_fuzzy_matches
        total_matches_percentage = (num_total_matches / num_assignments) * 100

        comparison_statistics = {
            "num_assignments": num_assignments,
            "num_exact_matches": num_exact_matches,
            "matching_percentage": matching_percentage,
            "num_fuzzy_matches": num_fuzzy_matches,
            "fuzzy_percentage": fuzzy_percentage,
            "num_total_matches": num_total_matches,
            "total_matches_percentage": total_matches_percentage,
        }
    else:
        comparison_statistics = {
            "num_assignments": 0,
            "num_exact_matches": 0,
            "matching_percentage": 0,
            "num_fuzzy_matches": 0,
            "fuzzy_percentage": 0,
            "num_total_matches": 0,
            "total_matches_percentage": 0,
        }

    return comparison_statistics


def compare_assignments(
    assignments_csv, ensembl_database, scientific_name, network=None
):
    """Compare assignments with the ones on the latest Ensembl release."""
    assignments_csv_path = pathlib.Path(assignments_csv)

    log_file_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.log"
    )
    add_log_file_handler(logger, log_file_path)

    if network is None:
        symbols_set = None
    else:
        configuration = network.hparams

        symbols_set = set(
            symbol.lower() for symbol in configuration.symbol_mapper.categories
        )

    comparisons_csv_path = pathlib.Path(
        f"{assignments_csv_path.parent}/{assignments_csv_path.stem}_compare.csv"
    )
    if not comparisons_csv_path.exists():
        compare_with_database(
            assignments_csv_path, ensembl_database, scientific_name, symbols_set
        )

    comparison_statistics = get_comparison_statistics(comparisons_csv_path)

    taxonomy_id = get_species_taxonomy_id(scientific_name)
    clade = get_taxonomy_id_clade(taxonomy_id)

    comparison_statistics["scientific_name"] = scientific_name
    comparison_statistics["taxonomy_id"] = taxonomy_id
    comparison_statistics["clade"] = clade

    message = "{} assignments, {} exact matches ({:.2f}%), {} fuzzy matches ({:.2f}%), {} total matches ({:.2f}%)".format(
        comparison_statistics["num_assignments"],
        comparison_statistics["num_exact_matches"],
        comparison_statistics["matching_percentage"],
        comparison_statistics["num_fuzzy_matches"],
        comparison_statistics["fuzzy_percentage"],
        comparison_statistics["num_total_matches"],
        comparison_statistics["total_matches_percentage"],
    )
    logger.info(message)

    dataframe_columns = [
        "clade",
        "scientific_name",
        "num_assignments",
        "num_exact_matches",
        "matching_percentage",
        "num_fuzzy_matches",
        "fuzzy_percentage",
        "num_total_matches",
        "total_matches_percentage",
    ]
    comparison_statistics = pd.DataFrame(
        [comparison_statistics],
        columns=dataframe_columns,
    )
    with pd.option_context("display.float_format", "{:.2f}".format):
        logger.info(
            f"comparison statistics:\n{comparison_statistics.to_string(index=False)}"
        )


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


if __name__ == "__main__":
    print("this is a module, import to use")
