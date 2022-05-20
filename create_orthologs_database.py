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


data_directory = pathlib.Path("data")

orthodb_directory = data_directory / "OrthoDB_files"

orthodb_files = {
    "odb10v1_OGs.tab": ["og_id", "tax_id", "og_name"],
    # "odb10v1_genes.tab": [],
    # "odb10v1_OG2genes.tab": [],
    # "odb10v1_gene_xrefs.tab": [],
}


def initialize_database(database_path):
    # delete database file if already exists
    database_path.unlink(missing_ok=True)

    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    # create database according to the schema
    database_schema_path = "orthologs_database_schema.sql"

    with open(database_schema_path, "r") as schema_file:
        database_schema = schema_file.read()

    cursor.executescript(database_schema)

    connection.commit()
    connection.close()

    print(f"{database_path} database initialized")


def populate_database(database_path):
    for orthodb_file in orthodb_files:
        csv_file_path = orthodb_directory / orthodb_file
        populate_table(database_path, csv_file_path)


def populate_table(database_path, csv_file_path):
    connection = sqlite3.connect(database_path)
    cursor = connection.cursor()

    table = csv_file_path.stem
    columns = orthodb_files[csv_file_path.name]

    print(f"populating table {table}")

    with open(csv_file_path, "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter="\t")

        columns_string = str.join(", ", columns)
        qmark_placeholder = str.join(", ", ["?"] * len(columns))
        insert_command = (
            f"INSERT INTO {table} ({columns_string}) VALUES ({qmark_placeholder});"
        )

        for row in csv_reader:
            cursor.execute(insert_command, row)

    connection.commit()
    print(f"{table} table populated")

    connection.close()


def get_max_column_lengths():
    for csv_file_path in csv_file_paths:
        print(csv_file_path)

        max_lengths = {}
        with open(csv_file_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter="\t")
            for row in csv_reader:
                for index, field in enumerate(row):
                    max_lengths[index] = max(len(field), max_lengths.get(index, 0))

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
        database_filename = "orthologs.db"
        database_path = data_directory / database_filename

        initialize_database(database_path)

        populate_database(database_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
