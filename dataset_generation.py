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
Merge original data files, normalize, cleanup, and filter examples to a single pandas dataframe saved as a pickle file.
"""


# standard library imports
import argparse
import pathlib
import pickle
import sys

# third party imports
import pandas as pd

from Bio import SeqIO

# project imports


data_directory = pathlib.Path("data")


def fasta_to_dataframe(fasta_path):
    """
    Generate a pandas dataframe with columns "description" and "sequence" from
    a FASTA file.
    """
    records = []
    with open(fasta_path) as fasta_file:
        for fasta_record in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
            records.append({"description": fasta_record[0], "sequence": fasta_record[1]})

    records_dataframe = pd.DataFrame(records)

    return records_dataframe


def merge_metadata_sequences(debug=False):
    """
    Merge the metadata CSV file and the sequences FASTA file in a single CSV file.
    """
    all_data_pickle_path = data_directory / "all_species_metadata_sequences.pickle"

    # exit function if the output file exists
    if all_data_pickle_path.is_file():
        print(f"{all_data_pickle_path} file already exists")
        return

    metadata_csv_path = data_directory / "original" / "all_species.csv"
    sequences_fasta_path = data_directory / "original" / "all_species.fa"

    # read the metadata csv file to a pandas dataframe
    metadata = pd.read_csv(metadata_csv_path, sep="\t")
    if debug:
        print(metadata.head())
        print()
        metadata.info()
        print()
        print(metadata.describe())
        print()
    assert metadata["stable_id"].nunique() == len(metadata)

    # generate a pandas dataframe from the sequences FASTA file
    sequences = fasta_to_dataframe(sequences_fasta_path)
    if debug:
        print(sequences.head())
        print()
        sequences.info()
        print()
        print(sequences.describe())
        print()
    assert sequences["description"].nunique() == len(sequences)

    # merge the two dataframes in a single one
    merged_data = pd.merge(
        left=metadata, right=sequences, left_on="stable_id", right_on="description"
    )
    if debug:
        print(merged_data.head())
        print()
        merged_data.info()
        print()
    assert (
        merged_data["stable_id"].nunique()
        == merged_data["description"].nunique()
        == len(merged_data)
    )

    # remove duplicate description column
    merged_data.drop(columns=["description"], inplace=True)
    if debug:
        merged_data.info()
        print()

    # save the merged dataframe to a pickle file
    merged_data.to_pickle(all_data_pickle_path)


def data_wrangling():
    """
    - simplify some column names
    - use symbol names in lowercase
    267536 unique original symbol names
    233824 unique lowercase symbol names
    12.6% reduction

    - filter out "Clone-based (Ensembl) gene" examples
      No standard name exists for them. 4691 examples filtered out.
    """
    # load the original data file
    all_data_pickle_path = data_directory / "all_species_metadata_sequences.pickle"
    print(f"loading {all_data_pickle_path} file...")
    data = pd.read_pickle(all_data_pickle_path)
    print(f"{all_data_pickle_path} file loaded")

    # simplify some column names
    print("renaming dataframe columns...")
    data = data.rename(
        columns={
            "display_xref.display_id": "symbol_original",
            "display_xref.db_display_name": "db_display_name",
        }
    )

    # pick the most frequent capitalization for each symbol
    print("picking the most frequent capitalization for each symbol...")
    # store symbol names in lowercase
    data["symbol_lowercase"] = data["symbol_original"].str.lower()
    symbols_capitalization_mapping = (
        data.groupby(["symbol_lowercase"])["symbol_original"]
        .agg(lambda x: pd.Series.mode(x)[0])
        .to_dict()
    )
    data["symbol"] = data["symbol_lowercase"].map(symbols_capitalization_mapping)

    # filter out "Clone-based (Ensembl) gene" examples
    print(
        'creating "include" column and filtering out "Clone-based (Ensembl) gene" examples'
    )
    data["include"] = data["db_display_name"] != "Clone-based (Ensembl) gene"

    # save the dataframe to a new data file
    data_pickle_path = data_directory / "data.pickle"
    data.to_pickle(data_pickle_path)


def save_all_datasets():
    """
    Save the examples for each num_symbols to a pickled dataframe and a FASTA file.
    """
    num_symbols_max_frequencies = [
        [3, 335],
        [101, 297],
        [1013, 252],
        [10059, 165],
        [20147, 70],
        [25028, 23],
        [26007, 17],
        [27137, 13],
        [28197, 10],
        [29041, 8],
        [30591, 5],
    ]

    data = load_data()

    for (num_symbols, max_frequency) in num_symbols_max_frequencies:
        print(f"saving {num_symbols} symbols dataset")
        save_dataset(data, num_symbols, max_frequency)


def save_dataset(data, num_symbols, max_frequency):
    """
    Save a training and testin
    """
    symbol_counts = data["symbol"].value_counts()

    # verify that max_frequency is the cutoff limit for the selected symbols
    if max_frequency is not None:
        assert all(
            symbol_counts[:num_symbols] == symbol_counts[symbol_counts >= max_frequency]
        )
        assert symbol_counts[num_symbols] < max_frequency

    most_frequent_n = data[data["symbol"].isin(symbol_counts[:num_symbols].index)]

    # save dataframe to a pickle file
    pickle_path = data_directory / f"{num_symbols}_symbols.pickle"
    most_frequent_n.to_pickle(pickle_path)
    print(
        f"pickle file of the most {num_symbols} frequent symbol sequences saved at {pickle_path}"
    )

    # save sequences to a FASTA file
    fasta_path = data_directory / f"{num_symbols}_symbols.fasta"
    with open(fasta_path, "w+") as fasta_file:
        for entry in most_frequent_n.itertuples():
            entry_dict = entry._asdict()

            stable_id = entry_dict["stable_id"]
            symbol = entry_dict["symbol"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id};{symbol}\n{sequence}\n")

    print(
        f"FASTA file of the most {num_symbols} frequent symbol sequences saved at {fasta_path}"
    )


def load_data():
    """
    Load data dataframe, excluding filtered out examples.
    """
    data_pickle_path = data_directory / "data.pickle"
    print("loading data...")
    data = pd.read_pickle(data_pickle_path)
    print("data loaded")

    # exclude filtered out examples
    data = data[data["include"] == True]

    return data


def save_all_sample_fasta_files():
    num_samples = 1000

    num_symbols_list = [
        3,
        101,
        1013,
        10059,
        20147,
        25028,
        26007,
        27137,
        28197,
        29041,
        30591,
    ]

    for num_symbols in num_symbols_list:
        print(f"saving {num_symbols} sample fasta file")
        save_sample_fasta(num_samples, num_symbols)


def save_sample_fasta(num_samples, num_symbols):
    data_pickle_path = data_directory / f"{num_symbols}_symbols.pickle"

    dataset = pd.read_pickle(data_pickle_path)

    # get num_samples random samples
    data = dataset.sample(num_samples)

    # only the sequences and the symbols are needed as features and labels
    data = data[["stable_id", "symbol", "sequence"]]

    # save sequences to a FASTA file
    fasta_path = data_directory / f"{num_symbols}_symbols-{num_samples}_samples.fasta"
    with open(fasta_path, "w+") as fasta_file:
        for entry in data.itertuples():
            entry_dict = entry._asdict()

            stable_id = entry_dict["stable_id"]
            symbol = entry_dict["symbol"]
            sequence = entry_dict["sequence"]

            fasta_file.write(f">{stable_id} {symbol}\n{sequence}\n")


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024
    return f"{num:.1f}Yi{suffix}"


def generate_dataset_statistics():
    data = load_data()
    print()

    # data_memory_usage = sys.getsizeof(data)
    # print("data memory usage: {}".format(sizeof_fmt(data_memory_usage)))
    # 3.8GiB

    num_sequences = len(data)
    print(f"{num_sequences} sequences")
    print()
    # 3805809 sequences

    # symbols occurrence frequency
    symbol_counts = data["symbol"].value_counts()
    print(symbol_counts)
    print()
    # nxpe3            378
    # pla1a            339
    # tbc1d9           335
    # NXPH1            329
    # CPNE3            323
    #                 ...
    # TRXH_1             1
    # BnaA10g05860D      1
    # BnaA03g11140D      1
    # BnaA09g47630D      1
    # BnaC04g10230D      1
    # Name: symbol, Length: 229133, dtype: int64

    symbol_counts_mean = symbol_counts.mean()
    symbol_counts_median = symbol_counts.median()
    symbol_counts_standard_deviation = symbol_counts.std()
    print(
        f"symbol counts mean: {symbol_counts_mean:.2f}, median: {symbol_counts_median:.2f}, standard deviation: {symbol_counts_standard_deviation:.2f}"
    )
    print()
    # symbol counts mean: 16.61, median: 1.00, standard deviation: 50.23

    sequence_length_mean = data["sequence"].str.len().mean()
    sequence_length_median = data["sequence"].str.len().median()
    sequence_length_standard_deviation = data["sequence"].str.len().std()
    print(
        f"sequence length mean: {sequence_length_mean:.2f}, median: {sequence_length_median:.2f}, standard deviation: {sequence_length_standard_deviation:.2f}"
    )
    print()
    # sequence length mean: 576.49, median: 442.00, standard deviation: 511.25


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--merge_metadata_sequences", action="store_true")
    argument_parser.add_argument("--data_wrangling", action="store_true")
    argument_parser.add_argument("--save_all_datasets", action="store_true")
    argument_parser.add_argument("--save_all_sample_fasta_files", action="store_true")
    argument_parser.add_argument("--generate_dataset_statistics", action="store_true")

    args = argument_parser.parse_args()

    if args.merge_metadata_sequences:
        merge_metadata_sequences()
    elif args.data_wrangling:
        data_wrangling()
    elif args.save_all_datasets:
        save_all_datasets()
    elif args.save_all_sample_fasta_files:
        save_all_sample_fasta_files()
    elif args.generate_dataset_statistics:
        generate_dataset_statistics()
    else:
        print("Error: missing argument.")
        print(__doc__)
        argument_parser.print_help()


if __name__ == "__main__":
    main()
