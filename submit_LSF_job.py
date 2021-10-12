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


"""Submit an LSF job to train or test a neural network gene symbol classifier.
"""


# standard library imports
import argparse
import datetime as dt
import pathlib
import subprocess
import sys

# third party imports
import yaml

# project imports
from utils import GeneSymbolClassifier, load_checkpoint

from gene_symbol_classifier import EarlyStopping, Experiment


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "-ex",
        "--experiment_settings",
        help="path to the experiment settings configuration YAML file",
    )
    argument_parser.add_argument(
        "--mem_limit",
        default=8192,
        type=int,
        help="memory limit for all the processes that belong to the job",
    )
    argument_parser.add_argument(
        "--checkpoint",
        help="path to the saved experiment checkpoint",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")
    argument_parser.add_argument(
        "--evaluate", action="store_true", help="evaluate a classifier"
    )
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run the evaluation for all genome assemblies in the Ensembl release",
    )

    args = argument_parser.parse_args()

    # submit new classifier training
    if args.experiment_settings:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        with open(args.experiment_settings) as f:
            experiment_settings = yaml.safe_load(f)

        num_symbols = experiment_settings["num_symbols"]

        experiment = Experiment(experiment_settings, datetime)
        job_name = experiment.filename

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--datetime {datetime}",
            f"--experiment_settings {args.experiment_settings}",
            "--train",
            "--test",
        ]

        root_directory = "experiments"

    # resume training, test, or evaluate a classifier
    elif args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        experiment, _network, _optimizer, _symbols_metadata = load_checkpoint(
            checkpoint_path
        )

        num_symbols = experiment.num_symbols

        job_name = checkpoint_path.stem
        root_directory = checkpoint_path.parent

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--checkpoint {args.checkpoint}",
        ]

        if args.train:
            pipeline_command_elements.append("--train")

        if args.test:
            pipeline_command_elements.append("--test")

        if args.evaluate:
            pipeline_command_elements.append("--evaluate")

        if args.complete:
            pipeline_command_elements.append("--complete")

    # no task specified
    else:
        print(__doc__)
        argument_parser.print_help()
        sys.exit()

    pipeline_command = " ".join(pipeline_command_elements)

    # specify lower mem_limit for dev datasets jobs
    num_symbols_mem_limit = {3: 1024, 100: 2048, 1000: 2048}
    if num_symbols in num_symbols_mem_limit.keys():
        mem_limit = num_symbols_mem_limit[num_symbols]
    elif args.evaluate:
        mem_limit = 2048
    else:
        mem_limit = args.mem_limit

    # common arguments for any job type
    bsub_command_elements = [
        "bsub",
        f"-q production",
        f"-M {mem_limit}",
        f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}]"',
        f"-o {root_directory}/{job_name}-stdout.log",
        f"-e {root_directory}/{job_name}-stderr.log",
    ]

    bsub_command_elements.append(pipeline_command)

    bsub_command = " ".join(bsub_command_elements)
    print(f"running command:\n{bsub_command}")

    try:
        command_output = subprocess.run(bsub_command, check=True, shell=True)
    except subprocess.CalledProcessError as ex:
        print(ex)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
