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
Create a SQLite database from the OrthoDB release files.
"""


# standard library imports
import argparse
import csv
import pathlib
import sqlite3

# third party imports

# project imports
from utils import add_log_file_handler, logger, read_fasta_in_chunks


data_directory = pathlib.Path("data")

orthodb_directory = data_directory / "OrthoDB_files"

orthodb_tab_files = {
    "odb10v1_OGs.tab": ["og_id", "tax_id", "og_name"],
    "odb10v1_genes.tab": [
        "gene_id",
        "organism_id",
        "sequence_id",
        "synonyms",
        "uniprot_id",
        "ensembl_ids",
        "ncbi_gid_or_gene_name",
        "description",
    ],
    "odb10v1_OG2genes.tab": ["og_id", "gene_id"],
    "odb10v1_gene_xrefs.tab": ["gene_id", "external_id", "external_db"],
}

orthodb_fasta_files = {
    "odb10v1_all_og_fasta.tab": ["gene_id", "public_gene_id", "sequence"],
}


def create_database():
    log_file_path = data_directory / "database_generation.log"
    add_log_file_handler(logger, log_file_path)

    database_filename = "orthologs.db"
    database_file_path = data_directory / database_filename

    database_schema_path = "orthologs_database_schema.sql"
    initialize_database(database_file_path, database_schema_path)

    populate_database(database_file_path)

    index_generation_file_path = "orthologs_database_indexes.sql"
    generate_indexes(database_file_path, index_generation_file_path)

    optimize_database(database_file_path)


def initialize_database(database_file_path, database_schema_path):
    # delete database file if already exists
    database_file_path.unlink(missing_ok=True)

    logger.info(f"creating database {database_file_path} ...")

    connection = sqlite3.connect(database_file_path)
    cursor = connection.cursor()

    # create database according to the schema
    with open(database_schema_path, "r") as database_schema_file:
        database_schema = database_schema_file.read()

    cursor.executescript(database_schema)

    connection.commit()
    connection.close()

    logger.info("database initialized")


def populate_database(database_file_path):
    logger.info("populating tables ...")
    for orthodb_tab_file in orthodb_tab_files:
        tab_file_path = orthodb_directory / orthodb_tab_file
        populate_tab_table(database_file_path, tab_file_path)

    for orthodb_fasta_file in orthodb_fasta_files:
        fasta_file_path = orthodb_directory / orthodb_fasta_file
        populate_fasta_table(database_file_path, fasta_file_path)


def populate_tab_table(database_file_path, tab_file_path):
    connection = sqlite3.connect(database_file_path)
    cursor = connection.cursor()

    table = tab_file_path.stem
    columns = orthodb_tab_files[tab_file_path.name]

    columns_string = str.join(", ", columns)
    qmark_placeholder = str.join(", ", ["?"] * len(columns))
    insert_command = (
        f"INSERT INTO {table} ({columns_string}) VALUES ({qmark_placeholder});"
    )

    with open(tab_file_path, "r") as tab_file:
        csv_reader = csv.reader(tab_file, delimiter="\t")
        for row in csv_reader:
            cursor.execute(insert_command, row)

    connection.commit()
    connection.close()

    logger.info(f"{table} populated")


def populate_fasta_table(database_file_path, fasta_file_path):
    connection = sqlite3.connect(database_file_path)
    cursor = connection.cursor()

    table = fasta_file_path.stem
    columns = orthodb_fasta_files[fasta_file_path.name]

    columns_string = str.join(", ", columns)
    qmark_placeholder = str.join(", ", ["?"] * len(columns))
    insert_command = (
        f"INSERT INTO {table} ({columns_string}) VALUES ({qmark_placeholder});"
    )

    for fasta_entries in read_fasta_in_chunks(fasta_file_path):
        for fasta_entry in fasta_entries:
            description = fasta_entry[0]
            sequence = fasta_entry[1]
            values = description.split("\t") + [sequence]

            cursor.execute(insert_command, values)

    connection.commit()
    connection.close()

    logger.info(f"{table} populated")


def generate_indexes(database_file_path, index_generation_file_path):
    connection = sqlite3.connect(database_file_path)
    cursor = connection.cursor()

    logger.info("generating indexes ...")

    with open(index_generation_file_path, "r") as index_generation_file:
        index_generation = index_generation_file.read()

    cursor.executescript(index_generation)

    connection.commit()
    connection.close()

    logger.info("indexes generation complete")


def optimize_database(database_file_path):
    connection = sqlite3.connect(database_file_path)
    cursor = connection.cursor()

    logger.info("optimizing database ...")

    # generate statistics for query optimization
    optimization_queries = "ANALYZE;"

    cursor.executescript(optimization_queries)

    connection.commit()
    connection.close()

    logger.info("database optimization complete")


def get_max_column_lengths():
    for orthodb_tab_file in orthodb_tab_files:
        tab_file_path = orthodb_directory / orthodb_tab_file
        logger.info(tab_file_path)

        columns = orthodb_tab_files[tab_file_path.name]

        max_lengths = {column: 0 for column in columns}
        with open(tab_file_path, "r") as tab_file:
            csv_dict_reader = csv.DictReader(
                tab_file, fieldnames=columns, delimiter="\t"
            )
            for entry in csv_dict_reader:
                for column, value in entry.items():
                    max_lengths[column] = max(len(value), max_lengths.get(column, 0))

        print(max_lengths)
        print()

    for orthodb_fasta_file in orthodb_fasta_files:
        fasta_file_path = orthodb_directory / orthodb_fasta_file
        print(fasta_file_path)

        columns = orthodb_fasta_files[fasta_file_path.name]

        max_lengths = {column: 0 for column in columns}
        for fasta_entries in read_fasta_in_chunks(fasta_file_path):
            for fasta_entry in fasta_entries:
                description = fasta_entry[0]
                sequence = fasta_entry[1]
                values = description.split("\t") + [sequence]
                entry = {column: value for column, value in zip(columns, values)}

                for column, value in entry.items():
                    max_lengths[column] = max(len(value), max_lengths.get(column, 0))

        print(max_lengths)
        print()


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--max_column_lengths",
        action="store_true",
        help="get the max length for each column in OrthoDB tab files",
    )

    args = argument_parser.parse_args()

    if args.max_column_lengths:
        get_max_column_lengths()
    else:
        create_database()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
