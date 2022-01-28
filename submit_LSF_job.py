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
import shutil
import subprocess
import sys

# third party imports
import yaml

# project imports
from utils import AttributeDict


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--pipeline", help="pipeline script path")
    argument_parser.add_argument(
        "--configuration",
        help="experiment configuration file path",
    )
    argument_parser.add_argument(
        "--mem_limit", default=16384, type=int, help="LSF job memory limit"
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
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")
    argument_parser.add_argument(
        "--evaluate", action="store_true", help="evaluate a classifier"
    )
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run evaluation for all genome assemblies in Ensembl main release",
    )

    args = argument_parser.parse_args()

    # submit new classifier training
    if args.pipeline and args.configuration:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        pipeline_path = pathlib.Path(args.pipeline)

        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)
        configuration = AttributeDict(configuration)

        job_name = (
            f"{configuration.experiment_prefix}_{configuration.dataset_id}_{datetime}"
        )
        root_directory = configuration.save_directory

        experiment_directory = pathlib.Path(f"{root_directory}/{job_name}")
        experiment_directory.mkdir(exist_ok=True)

        # copy pipeline script, configuration file, and dependencies
        pipeline_copy = shutil.copy(pipeline_path, experiment_directory)
        configuration_copy = shutil.copy(args.configuration, experiment_directory)
        shutil.copy(pipeline_path.parent / "utils.py", experiment_directory)

        pipeline_command_elements = [
            f"python {pipeline_copy}",
            f"--datetime {datetime}",
            f"--configuration {configuration_copy}",
            "--test",
            "--train",
        ]

    # test or evaluate a classifier
    elif args.pipeline and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        job_name = checkpoint_path.stem
        root_directory = checkpoint_path.parent

        pipeline_command_elements = [
            f"python {args.pipeline}",
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

    # common job arguments
    bsub_command_elements = [
        "bsub",
        f"-M {args.mem_limit}",
        f"-o {experiment_directory}/stdout.log",
        f"-e {experiment_directory}/stderr.log",
    ]

    if args.gpu:
        num_gpus = 1
        # gpu_memory = 16384  # 16 GiBs
        gpu_memory = 32510  # ~32 GiBs, total Tesla V100 memory

        bsub_command_elements.extend(
            [
                "-q gpu",
                f'-gpu "num={num_gpus}:gmem={gpu_memory}:j_exclusive=yes"',
                f"-M {args.mem_limit}",
                f'-R"select[mem>{args.mem_limit}] rusage[mem={args.mem_limit}] span[hosts=1]"',
            ]
        )
    else:
        bsub_command_elements.extend(
            [
                "-q production",
                f'-R"select[mem>{args.mem_limit}] rusage[mem={args.mem_limit}]"',
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
