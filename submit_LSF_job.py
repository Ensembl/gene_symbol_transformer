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
        "--job_type",
        default="standard",
        help='submitted job type, one of "standard", "gpu", or "parallel"',
    )
    argument_parser.add_argument(
        "--compute_node",
        default="gpu-009",
        help='name of compute node to submit the job, for GPU one of "gpu-009" or "gpu-011"',
    )
    argument_parser.add_argument(
        "--num_tasks",
        default=1,
        type=int,
        help="number of tasks for a parallel job",
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
    argument_parser.add_argument(
        "--test", action="store_true", help="test a trained classifier"
    )

    args = argument_parser.parse_args()

    # submit new classifier training
    if args.experiment_settings:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        with open(args.experiment_settings) as f:
            experiment_settings = yaml.safe_load(f)

        num_symbols = experiment_settings["num_symbols"]

        job_name = f"n={num_symbols}_{datetime}"

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--datetime {datetime}",
            f"--experiment_settings {args.experiment_settings}",
            "--train",
            "--test",
        ]
    # resume training or test a saved classifier
    elif args.checkpoint:
        job_name = pathlib.Path(args.checkpoint).stem

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--checkpoint {args.checkpoint}",
        ]

        if args.train:
            pipeline_command_elements.append("--train")

        if args.test:
            pipeline_command_elements.append("--test")
    # no task specified
    else:
        print(__doc__)
        argument_parser.print_help()
        sys.exit()

    pipeline_command = " ".join(pipeline_command_elements)

    # common arguments for any job type
    bsub_command_elements = [
        "bsub",
        f"-M {args.mem_limit}",
        f'-R"select[mem>{args.mem_limit}] rusage[mem={args.mem_limit}]"',
        f"-o experiments/{job_name}.stdout.log",
        f"-e experiments/{job_name}.stderr.log",
    ]

    # GPU node job
    if args.job_type == "gpu":
        bsub_command_elements.extend(
            [
                "-P gpu",
                f'-gpu "num={args.num_tasks}:j_exclusive=yes"',
                f"-m {args.compute_node}.ebi.ac.uk",
            ]
        )

    # parallel job
    if args.job_type == "parallel":
        if args.num_tasks == 1:
            raise ValueError("parallel job specified but the number of tasks is set to 1")

        bsub_command_elements.extend(
            [
                f"-n {args.num_tasks}",
                f'-R"span[hosts=1]"',
            ]
        )

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
