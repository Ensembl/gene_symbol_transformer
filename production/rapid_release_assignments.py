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
Generate gene symbol assignments for genome assemblies on the Rapid Release.

https://rapid.ensembl.org/info/about/species.html
"""


# standard library imports
import argparse
import json
import pathlib
import sys
import time

from types import SimpleNamespace

# third party imports
import pandas as pd
import pymysql

# project imports
from utils import (
    GeneSymbolClassifier,
    add_log_file_handler,
    assign_symbols,
    data_directory,
    download_file,
    generate_canonical_protein_sequences_fasta,
    logger,
)


def generate_assignments(checkpoint_path, rapid_ensembl_release):
    """
    Generate gene symbol assignments for genome assemblies on the Rapid Release.

    Args:
        checkpoint_path (Path): path to the experiment checkpoint
    """
    network = GeneSymbolClassifier.load_from_checkpoint(checkpoint_path)
    configuration = network.hparams

    assemblies = get_rapid_release_assemblies_metadata(rapid_ensembl_release)

    for assembly in assemblies:
        canonical_fasta_path = generate_canonical_protein_sequences_fasta(
            assembly, "rapid_release"
        )

        # assign symbols
        assignments_csv_path = pathlib.Path(
            f"{checkpoint_path.parent}/{canonical_fasta_path.stem}_symbols.csv"
        )
        if not assignments_csv_path.exists():
            logger.info(f"assigning gene symbols to {canonical_fasta_path}")
            assign_symbols(
                network,
                configuration.symbols_metadata,
                canonical_fasta_path,
                taxonomy_id=assembly.taxonomy_id,
                output_directory=checkpoint_path.parent,
            )


def get_rapid_release_assemblies_metadata(rapid_ensembl_release):
    """
    Get metadata for genome assemblies on Ensembl Rapid Release.

    The metadata are loaded from the `species_metadata.json` file in the Rapid Release
    FTP root directory.
    """
    assemblies_metadata_path = (
        data_directory / "rapid_release_assemblies_metadata.pickle"
    )
    if assemblies_metadata_path.exists():
        assemblies_metadata_df = pd.read_pickle(assemblies_metadata_path)
        assemblies_metadata = [
            SimpleNamespace(**values.to_dict())
            for index, values in assemblies_metadata_df.iterrows()
        ]
        return assemblies_metadata

    logger.info("generating Rapid Release genome assemblies metadata")

    # download the species metadata file
    assemblies_metadata_json_filename = "species_metadata.json"
    assemblies_metadata_json_url = (
        f"http://ftp.ensembl.org/pub/rapid-release/{assemblies_metadata_json_filename}"
    )
    data_directory.mkdir(exist_ok=True)
    assemblies_metadata_json_path = data_directory / assemblies_metadata_json_filename
    if not assemblies_metadata_json_path.exists():
        download_file(assemblies_metadata_json_url, assemblies_metadata_json_path)
        logger.info(f"downloaded {assemblies_metadata_json_path}")

    with open(assemblies_metadata_json_path) as file:
        assemblies_metadata = json.load(file)

    assemblies_metadata = [
        SimpleNamespace(**assembly) for assembly in assemblies_metadata
    ]

    gca_to_core_dbs = get_rapid_release_core_dbs(rapid_ensembl_release)

    for assembly in assemblies_metadata:
        assembly.core_db = gca_to_core_dbs[assembly.assembly_accession]

        fix_assembly_geneset(assembly)

        assembly.fasta_filename = "{}-{}-{}-pep.fa".format(
            assembly.species.replace(" ", "_"),
            assembly.assembly_accession,
            assembly.geneset,
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


def fix_assembly_geneset(assembly):
    geneset_mappings = {
        "GCA_902859565.1": "2021_02",
        "GCA_000001215.4": "2020_08",
        "GCA_004118075.1": "2021_03",
        "GCA_914969975.1": "2021_05",
        "GCA_000648675.3": "2021_05",
        "GCA_015732765.1": "2021_07",
        "GCA_013339745.1": "2021_08",
        "GCA_013339765.1": "2021_09",
        "GCA_013339685.1": "2021_09",
        "GCA_013358835.1": "2021_09",
        "GCA_016920785.2": "2021_07",
        "GCA_013339725.1": "2021_08",
        "GCA_013339695.1": "2021_08",
        "GCA_014595675.1": "2021_07",
        "GCA_910591555.1": "2021_05",
    }

    if assembly.assembly_accession in geneset_mappings:
        erroneous_geneset = assembly.geneset
        geneset = geneset_mappings[assembly.assembly_accession]
        assembly.geneset = geneset
        logger.info(
            f"assembly {assembly.assembly_accession} geneset fixed from {erroneous_geneset} to {geneset}"
        )


def get_rapid_release_core_dbs(
    rapid_ensembl_release,
    host="mysql-ens-mirror-5",
    port=4692,
    user="ensro",
):
    """
    Get assembly accessions to core database names for Rapid Release.
    """
    connection = pymysql.connect(
        host=host,
        port=port,
        user=user,
    )
    get_core_databases_sql = f"""
    SHOW DATABASES LIKE '%core_{rapid_ensembl_release}%';
    """
    with connection:
        with connection.cursor() as cursor:
            cursor.execute(get_core_databases_sql)
            core_databases = cursor.fetchall()

    skip_databases = {
        "viruses_mixed_collection_core_104_1",
        "viruses_orthocoronavirinae_collection_core_104_1",
    }

    assembly_accession_to_core_databases = {}
    for core_database in core_databases:
        # delay between SQL queries
        time.sleep(0.1)

        assert len(core_database) == 1, f"{core_database}"
        core_database = core_database[0]

        if core_database in skip_databases:
            continue

        connection = pymysql.connect(
            host=host,
            port=port,
            user=user,
            database=core_database,
            cursorclass=pymysql.cursors.DictCursor,
        )

        get_assembly_accessions_sql = """
        SELECT meta_value FROM meta WHERE meta_key='assembly.accession';
        """
        with connection:
            with connection.cursor() as cursor:
                cursor.execute(get_assembly_accessions_sql)
                assembly_accession = cursor.fetchall()
        assert len(assembly_accession) == 1, f"{assembly_accession}"
        assembly_accession = assembly_accession[0]["meta_value"]

        assembly_accession_to_core_databases[assembly_accession] = core_database

    return assembly_accession_to_core_databases


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--checkpoint", help="experiment checkpoint path")
    argument_parser.add_argument(
        "--rapid_ensembl_release",
        help="Rapid Release latest schema version",
        type=int,
    )

    args = argument_parser.parse_args()

    # assign symbols to genome assemblies on the Rapid Release
    if args.checkpoint and args.rapid_ensembl_release:
        checkpoint_path = pathlib.Path(args.checkpoint)
        log_file_path = pathlib.Path(
            f"{checkpoint_path.parent}/{checkpoint_path.stem}_rapid_release.log"
        )
        add_log_file_handler(logger, log_file_path)

        generate_assignments(checkpoint_path, args.rapid_ensembl_release)

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
