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
Testing retrieving data from the orghologs SQLite database.
"""


# standard library imports
import argparse
import pathlib
import sqlite3

# third party imports
import pandas as pd

# project imports


data_directory = pathlib.Path("data")


def run_query(database_file_path, query_file_path):
    with open(query_file_path, "r") as query_file:
        query = query_file.read()

    # open database in read-only mode
    connection = sqlite3.connect(f"file:{database_file_path}?mode=ro", uri=True)
    cursor = connection.cursor()

    for row in cursor.execute(query):
        print(row)

    # connection.commit()
    connection.close()


def get_database_statistics(database_file_path):
    # open database in read-only mode
    connection = sqlite3.connect(f"file:{database_file_path}?mode=ro", uri=True)
    cursor = connection.cursor()

    get_tables_list_query = """
        SELECT name
        FROM sqlite_master
        WHERE type='table'
          AND name like 'odb10v1%'
        ;
    """

    for row in cursor.execute(get_tables_list_query):
        table = row[0]
        read_table_query = f"SELECT * FROM {table} LIMIT 1000000;"
        table_df = pd.read_sql_query(read_table_query, connection)
        print(f"generating descriptive statistics for table {table} ...")
        table_df_describe = table_df.describe()
        print(table_df_describe)

    # connection.commit()
    connection.close()


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("query_file", nargs="?", help="SQL query file path")

    args = argument_parser.parse_args()

    database_filename = "orthologs.db"
    database_file_path = data_directory / database_filename

    if args.query_file:
        run_query(database_file_path, args.query_file)
    else:
        get_database_statistics(database_file_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
