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
Generate gene symbol assignments for genome assemblies on the Rapid Release.
"""


# standard library imports
import argparse
import pathlib
import sys

# third party imports
from loguru import logger

# project imports
from gene_symbol_classifier import EarlyStopping, Experiment, GeneSymbolClassifier
from utils import load_checkpoint, logging_format, sequences_directory


def generate_assignments(checkpoint_path):
    """
    Generate gene symbol assignments for genome assemblies on the Rapid Release.

    Args:
        checkpoint_path (Path): path to the experiment checkpoint
    """
    experiment, network = load_checkpoint(checkpoint_path)
    symbols_set = set(symbol.lower() for symbol in experiment.gene_symbols_mapper.symbols)
    logger.info(experiment)
    logger.info(network)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--checkpoint",
        help="experiment checkpoint path",
    )

    args = argument_parser.parse_args()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=logging_format)

    # assign symbols to genome assemblies on the Rapid Release
    if args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        log_file_path = pathlib.Path(
            f"{checkpoint_path.parent}/{checkpoint_path.stem}_evaluate.log"
        )
        logger.add(log_file_path, format=logging_format)

        generate_assignments(checkpoint_path)

    else:
        argument_parser.print_help()
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
