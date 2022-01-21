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
Submit an LSF job to train, test, or evaluate a neural network gene symbol classifier.
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
from utils import AttributeDict, GeneSymbolClassifier


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration YAML file",
    )
    argument_parser.add_argument(
        "--mem_limit",
        default=16384,
        type=int,
        help="memory limit for all the processes that belong to the job",
    )
    argument_parser.add_argument(
        "--gpu",
        action="store_true",
        help="submit training job to the gpu queue",
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
    if args.configuration:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)
        configuration = AttributeDict(configuration)

        num_symbols = configuration.num_symbols

        configuration.datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        job_name = f"{configuration.experiment_prefix}_{configuration.num_symbols}_symbols_{configuration.datetime}"

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--datetime {datetime}",
            f"--configuration {args.configuration}",
            "--train",
            "--test",
        ]

        root_directory = configuration.save_directory

    # test or evaluate a classifier
    elif args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        network = GeneSymbolClassifier.load_from_checkpoint(checkpoint_path)

        num_symbols = network.hparams.num_symbols

        job_name = checkpoint_path.stem
        root_directory = checkpoint_path.parent

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--checkpoint {args.checkpoint}",
        ]

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
    num_symbols_mem_limit = {3: 1024, 100: 2048, 1000: 8192}
    if num_symbols in num_symbols_mem_limit:
        mem_limit = num_symbols_mem_limit[num_symbols]
    elif args.evaluate:
        mem_limit = 2048
    else:
        mem_limit = args.mem_limit

    # common job arguments
    bsub_command_elements = [
        "bsub",
        f"-M {mem_limit}",
        f"-o {root_directory}/{job_name}/stdout.log",
        f"-e {root_directory}/{job_name}/stderr.log",
    ]

    if args.gpu:
        num_gpus = 1
        # gpu_memory = 16384  # 16 GiBs
        gpu_memory = 32510  # ~32 GiBs, total Tesla V100 memory

        bsub_command_elements.extend(
            [
                "-q gpu",
                f'-gpu "num={num_gpus}:gmem={gpu_memory}:j_exclusive=yes"',
                f"-M {mem_limit}",
                f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}] span[hosts=1]"',
            ]
        )
    else:
        bsub_command_elements.extend(
            [
                "-q production",
                f'-R"select[mem>{mem_limit}] rusage[mem={mem_limit}]"',
            ]
        )

    bsub_command_elements.append(pipeline_command)

    bsub_command = " ".join(bsub_command_elements)
    print(f"running command:\n{bsub_command}")

    subprocess.run(bsub_command, shell=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
