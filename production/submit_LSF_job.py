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
Submit an LSF job to train, test, or evaluate a gene symbol classifier model.
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

from pytorch_lightning.utilities import AttributeDict


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--pipeline", type=str, help="pipeline script path")
    argument_parser.add_argument(
        "--configuration", help="experiment configuration file path"
    )
    argument_parser.add_argument(
        "--mem_limit", default=32768, type=int, help="LSF job memory limit"
    )
    argument_parser.add_argument(
        "--gpu",
        default=None,
        choices=["A100", "V100", None],
        type=str,
        help="GPU nodes queue to submit job",
    )
    argument_parser.add_argument(
        "--num_gpus", default=1, type=int, help="number of GPUs to use"
    )
    argument_parser.add_argument(
        "--checkpoint",
        help="path to the saved experiment checkpoint",
    )
    argument_parser.add_argument(
        "--test", action="store_true", help="test a classifier"
    )
    argument_parser.add_argument(
        "--evaluate", action="store_true", help="evaluate a classifier"
    )
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run evaluation for all genome assemblies in Ensembl main release",
    )

    args = argument_parser.parse_args()

    # submit new training job
    if args.pipeline and args.configuration:
        datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")

        pipeline_path = pathlib.Path(args.pipeline)

        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)
        configuration = AttributeDict(configuration)

        if "num_symbols" in configuration:
            assert (
                "min_frequency" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            dataset_id = f"{configuration.num_symbols}_num_symbols"
        elif "min_frequency" in configuration:
            assert (
                "num_symbols" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            dataset_id = f"{configuration.min_frequency}_min_frequency"
        else:
            raise KeyError(
                'missing configuration value: one of "num_symbols", "min_frequency" is required'
            )

        job_name = f"{configuration.experiment_prefix}_{dataset_id}_{datetime}"
        root_directory = configuration.save_directory

        experiment_directory = pathlib.Path(f"{root_directory}/{job_name}")
        experiment_directory.mkdir(parents=True, exist_ok=True)

        # copy pipeline script, configuration file, and dependencies
        pipeline_copy = shutil.copy2(pipeline_path, experiment_directory)
        configuration_copy = shutil.copy2(
            args.configuration, experiment_directory / "configuration.yaml"
        )
        pipeline_files = ["models.py", "utils.py"]
        for pipeline_file in pipeline_files:
            shutil.copy2(pipeline_path.parent / pipeline_file, experiment_directory)

        pipeline_command_elements = [
            f"python {pipeline_copy}",
            f"--datetime {datetime}",
            f"--configuration {configuration_copy}",
            "--test",
            "--train",
        ]

    # test or evaluate a model
    elif args.pipeline and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        job_name = checkpoint_path.stem
        root_directory = checkpoint_path.parent

        experiment_directory = pathlib.Path(f"{root_directory}/{job_name}")
        experiment_directory.mkdir(parents=True, exist_ok=True)

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

    # common job arguments
    bsub_command_elements = [
        "bsub",
        f"-o {experiment_directory}/stdout.log",
        f"-e {experiment_directory}/stderr.log",
    ]

    if args.gpu:
        # run training on an NVIDIA A100 GPU node
        if args.gpu == "A100":
            gpu_memory = 81000  # ~80 GiBs, NVIDIA A100 memory with safety margin
            # gpu_memory = 81920  # 80 GiBs, total NVIDIA A100 memory
            bsub_command_elements.append("-q gpu-a100")

        # run training on an NVIDIA V100 GPU node
        elif args.gpu == "V100":
            gpu_memory = 32256  # 31.5 GiBs, NVIDIA V100 memory with safety margin
            # gpu_memory = 32510  # ~32 GiBs, total NVIDIA V100 memory
            bsub_command_elements.append("-q gpu")

        bsub_command_elements.extend(
            [
                f'-gpu "num={args.num_gpus}:gmem={gpu_memory}:j_exclusive=yes"',
                f'-R"select[mem>{args.mem_limit}] rusage[mem={args.mem_limit}] span[hosts=1]"',
            ]
        )
        pipeline_command_elements.append(f"--num_gpus {args.num_gpus}")

    else:
        bsub_command_elements.extend(
            [
                "-q production",
                f'-R"select[mem>{args.mem_limit}] rusage[mem={args.mem_limit}]"',
            ]
        )

    pipeline_command = " ".join(pipeline_command_elements)

    bsub_command_elements.extend(
        [
            f"-M {args.mem_limit}",
            pipeline_command,
        ]
    )

    bsub_command = " ".join(bsub_command_elements)
    print(f"running command:\n{bsub_command}")

    subprocess.run(bsub_command, shell=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted with CTRL-C, exiting...")
