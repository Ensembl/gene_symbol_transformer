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


"""Submit a group of training experiments as LSF jobs.
"""


# standard library imports
import argparse
import copy
import datetime as dt
import subprocess
import sys
import time

# third party imports
import yaml

# project imports
from gene_symbol_classifier import Experiment


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--group_settings",
        help="path to the training group settings configuration YAML file",
    )
    argument_parser.add_argument(
        "--mem_limit",
        default=65536,
        type=int,
        help="memory limit for all the processes that belong to the job",
    )

    args = argument_parser.parse_args()

    # no task specified
    if not args.group_settings:
        print(__doc__)
        argument_parser.print_help()
        sys.exit()

    root_directory = "experiments"

    datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

    with open(args.group_settings) as f:
        group_settings = yaml.safe_load(f)

    task_tuning = {
        25228: {"num_workers": 47},
        30241: {"num_workers": 47},
        30568: {"num_workers": 47},
        30911: {"num_workers": 47},
        31235: {"num_workers": 47},
        31630: {"num_workers": 47},
        32068: {"num_workers": 47},
        32563: {"num_workers": 47},
        33260: {"num_workers": 31},
        34461: {"num_workers": 31},
        37440: {"num_workers": 19},
    }

    for num_symbols in task_tuning:
        experiment_settings = copy.deepcopy(group_settings)

        experiment_settings["num_symbols"] = num_symbols
        experiment_settings["num_workers"] = task_tuning[num_symbols]["num_workers"]

        experiment = Experiment(experiment_settings, datetime)
        job_name = experiment.filename

        experiment_settings_file = f"{job_name}.yaml"
        with open(experiment_settings_file, "w") as f:
            yaml.dump(experiment_settings, f, default_flow_style=False, sort_keys=False)

        pipeline_command_elements = [
            "python gene_symbol_classifier.py",
            f"--datetime {datetime}",
            f"--experiment_settings {experiment_settings_file}",
            "--train",
            "--test",
        ]

        pipeline_command = " ".join(pipeline_command_elements)

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
        print(f"submit {num_symbols} num_symbols training job:\n{bsub_command}")

        try:
            command_output = subprocess.run(bsub_command, check=True, shell=True)
        except subprocess.CalledProcessError as ex:
            print(ex)

        time.sleep(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
