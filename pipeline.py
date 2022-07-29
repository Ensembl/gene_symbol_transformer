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
Train, test, evaluate, and use a gene symbol classifier to assign gene symbols
to protein sequences.

Evaluate a trained network
A trained network, specified with the `--checkpoint` argument with its path,
is evaluated by assigning symbols to the canonical translations of protein sequences
of annotations in the latest Ensembl release and comparing them to the existing
symbol assignments.

Get statistics for existing symbol assignments
Gene symbol assignments from a classifier can be compared against the existing
assignments in the Ensembl database, by specifying the path to the assignments CSV file
with `--assignments_csv` and the Ensembl database name with `--ensembl_database`.
"""


# standard library imports
import argparse
import datetime as dt
import json
import pathlib
import random
import warnings

# third party imports
import pytorch_lightning as pl
import torch
import yaml

# project imports
from models import GSC
from utils import (
    AttributeDict,
    ConciseReprDict,
    add_log_file_handler,
    assign_symbols,
    compare_assignments,
    data_directory,
    evaluate_network,
    generate_dataloaders,
    log_pytorch_cuda_info,
    logger,
)


def main():
    """
    main function
    """
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument(
        "--datetime",
        help="datetime string; if set this will be used instead of generating a new one",
    )
    argument_parser.add_argument(
        "--configuration",
        help="path to the experiment configuration file",
    )
    argument_parser.add_argument(
        "--num_gpus", default=1, type=int, help="number of GPUs to use"
    )
    argument_parser.add_argument(
        "--checkpoint",
        help="experiment checkpoint path",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument(
        "--test", action="store_true", help="test a classifier"
    )
    argument_parser.add_argument(
        "--sequences_fasta",
        help="path of FASTA file with protein sequences to assign symbols to",
    )
    argument_parser.add_argument(
        "--scientific_name",
        help="scientific name of the species the protein sequences belong to",
    )
    argument_parser.add_argument(
        "--evaluate", action="store_true", help="evaluate a classifier"
    )
    argument_parser.add_argument(
        "--complete",
        action="store_true",
        help="run the evaluation for all genome assemblies in the Ensembl release",
    )
    argument_parser.add_argument(
        "--assignments_csv",
        help="assignments CSV file path",
    )
    argument_parser.add_argument(
        "--ensembl_database",
        help="genome assembly core database name on the public Ensembl MySQL server",
    )

    args = argument_parser.parse_args()

    # filter warning about number of dataloader workers
    warnings.filterwarnings(
        "ignore",
        ".*does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument.*",
    )

    # train a new classifier
    if args.train and args.configuration:
        # read the experiment configuration YAML file to a dictionary
        with open(args.configuration) as file:
            configuration = yaml.safe_load(file)

        configuration = AttributeDict(configuration)

        if args.datetime:
            configuration.datetime = args.datetime
        else:
            configuration.datetime = dt.datetime.now().isoformat(
                sep="_", timespec="seconds"
            )

        if "min_frequency" in configuration:
            assert (
                "num_symbols" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.min_frequency}_min_frequency"
        elif "num_symbols" in configuration:
            assert (
                "min_frequency" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.num_symbols}_num_symbols"
        else:
            raise KeyError(
                'missing configuration value: one of "min_frequency", "num_symbols" is required'
            )

        configuration.logging_version = f"{configuration.experiment_prefix}_{configuration.dataset_id}_{configuration.datetime}"

        # generate random seed if it doesn't exist
        # Using the range [1_000_000, 1_001_000] for the random seed. This range contains
        # numbers that have a good balance of 0 and 1 bits, as recommended by the PyTorch docs.
        # https://pytorch.org/docs/stable/generated/torch.Generator.html#torch.Generator.manual_seed
        configuration.random_seed = configuration.get(
            "random_seed", random.randint(1_000_000, 1_001_000)
        )

        configuration.feature_encoding = "label"

        configuration.experiment_directory = (
            f"{configuration.save_directory}/{configuration.logging_version}"
        )
        log_directory_path = pathlib.Path(configuration.experiment_directory)
        log_directory_path.mkdir(parents=True, exist_ok=True)

        log_file_path = log_directory_path / "experiment.log"
        add_log_file_handler(logger, log_file_path)

        log_pytorch_cuda_info()

        # get training, validation, and test dataloaders
        (
            training_dataloader,
            validation_dataloader,
            test_dataloader,
        ) = generate_dataloaders(configuration)

        if configuration.num_symbols < 1000:
            configuration.symbols_metadata = None
        else:
            # load symbols metadata
            symbols_metadata_filename = "symbols_metadata.json"
            symbols_metadata_path = data_directory / symbols_metadata_filename
            with open(symbols_metadata_path) as file:
                configuration.symbols_metadata = ConciseReprDict(json.load(file))

        # instantiate neural network
        network = GSC(**configuration)

        # don't use a per-experiment subdirectory
        logging_name = ""

        tensorboard_logger = pl.loggers.TensorBoardLogger(
            save_dir=configuration.save_directory,
            name=logging_name,
            version=configuration.logging_version,
            default_hp_metric=False,
        )

        early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
            monitor="validation_loss",
            min_delta=configuration.loss_delta,
            patience=configuration.patience,
            verbose=True,
        )

        if torch.cuda.is_available():
            num_gpus = args.num_gpus
        else:
            num_gpus = 0

        trainer = pl.Trainer(
            gpus=num_gpus,
            logger=tensorboard_logger,
            max_epochs=configuration.max_epochs,
            log_every_n_steps=1,
            callbacks=[early_stopping_callback],
            profiler=configuration.profiler,
        )

        trainer.fit(
            model=network,
            train_dataloaders=training_dataloader,
            val_dataloaders=validation_dataloader,
        )

        if configuration.test_size > 0:
            trainer.test(ckpt_path="best", dataloaders=test_dataloader)

    # test a classifier
    elif args.test and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = f"{checkpoint_path.parent}/experiment.log"
        add_log_file_handler(logger, log_file_path)

        network = GSC.load_from_checkpoint(args.checkpoint)

        _, _, test_dataloader = generate_dataloaders(network.hparams)

        if torch.cuda.is_available():
            num_gpus = args.num_gpus
        else:
            num_gpus = 0

        trainer = pl.Trainer(gpus=num_gpus)
        trainer.test(network, dataloaders=test_dataloader)

    # evaluate a classifier
    elif args.evaluate and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        evaluation_directory_path = (
            checkpoint_path.parent / f"{checkpoint_path.stem}_evaluation"
        )
        evaluation_directory_path.mkdir(exist_ok=True)
        log_file_path = (
            evaluation_directory_path / f"{checkpoint_path.stem}_evaluate.log"
        )
        add_log_file_handler(logger, log_file_path)

        network = GSC.load_from_checkpoint(checkpoint_path)

        evaluate_network(network, checkpoint_path, args.complete)

    # assign symbols to sequences
    elif args.sequences_fasta and args.scientific_name and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = f"{checkpoint_path.parent}/experiment.log"
        add_log_file_handler(logger, log_file_path)

        network = GSC.load_from_checkpoint(args.checkpoint)
        configuration = network.hparams

        logger.info("assigning symbols...")
        assign_symbols(
            network,
            args.sequences_fasta,
            scientific_name=args.scientific_name,
        )

    # compare assignments with the ones on the latest Ensembl release
    elif args.assignments_csv and args.ensembl_database and args.scientific_name:
        if args.checkpoint:
            network = GSC.load_from_checkpoint(args.checkpoint)
        else:
            network = None

        compare_assignments(
            assignments_csv=args.assignments_csv,
            ensembl_database=args.ensembl_database,
            scientific_name=args.scientific_name,
            network=network,
        )

    else:
        argument_parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
