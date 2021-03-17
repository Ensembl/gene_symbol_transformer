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
Fully connected neural network pipeline.
"""


# standard library imports
import argparse
import csv
import datetime as dt
import itertools
import math
import pathlib
import sys
import time

# third party imports
import numpy as np
import torch
import torch.nn as nn
import yaml

from Bio import SeqIO
from loguru import logger
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

# project imports
from pipeline_abstractions import (
    EarlyStopping,
    PrettySimpleNamespace,
    SequenceDataset,
    TrainingSession,
    experiments_directory,
    load_checkpoint,
    specify_device,
    transform_sequences,
)


LOGURU_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{message}</level>"

DEVICE = specify_device()


class FullyConnectedNetwork(nn.Module):
    """
    A fully connected neural network for gene name classification of protein sequences
    using the protein letters as features.
    """

    def __init__(
        self,
        sequence_length,
        num_protein_letters,
        num_symbols,
        num_connections,
        dropout_probability,
        gene_symbols,
    ):
        """
        Initialize the neural network.
        """
        super().__init__()

        self.sequence_length = sequence_length
        self.gene_symbols = gene_symbols

        input_size = self.sequence_length * num_protein_letters
        output_size = num_symbols

        self.input_layer = nn.Linear(in_features=input_size, out_features=num_connections)
        self.output_layer = nn.Linear(
            in_features=num_connections, out_features=output_size
        )
        self.dropout = nn.Dropout(dropout_probability)

        self.relu = nn.ReLU()
        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        """
        Perform a forward pass of the network.
        """
        # flatten sample values to one dimension
        # original x.shape: torch.Size([batch_size, sequence_length, num_protein_letters])
        # e.g. torch.Size([512, 1000, 27])
        # flattened x.shape: torch.Size([batch_size, sequence_length * num_protein_letters])
        # e.g. torch.Size([512, 27000])
        x = torch.flatten(x, start_dim=1)

        x = self.input_layer(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.output_layer(x)
        x = self.dropout(x)
        x = self.final_activation(x)

        return x

    def predict(self, sequences):
        """
        Get predictions of symbols for a list of protein sequences.
        """
        tensor_sequences = transform_sequences(sequences, self.sequence_length)
        tensor_sequences = tensor_sequences.to(DEVICE)

        # run inference
        with torch.no_grad():
            self.eval()
            output = self.forward(tensor_sequences)

        # get predicted labels from output
        predicted_probabilities = torch.exp(output)
        predictions = torch.argmax(predicted_probabilities, dim=1)

        predictions = self.gene_symbols.one_hot_encoding_to_symbol(predictions)
        predictions = predictions.tolist()

        return predictions


def train_network(
    network,
    training_session,
    training_loader,
    validation_loader,
    verbose=False,
):
    tensorboard_log_dir = (
        f"runs/{training_session.num_symbols}/{training_session.datetime}"
    )
    summary_writer = SummaryWriter(log_dir=tensorboard_log_dir)

    num_epochs = training_session.num_epochs
    criterion = training_session.criterion

    # optimization function
    training_session.optimizer = torch.optim.Adam(
        network.parameters(), lr=training_session.learning_rate
    )

    clip_max_norm = 5

    checkpoint_path = experiments_directory / training_session.checkpoint_filename
    stop_early = EarlyStopping(
        checkpoint_path, training_session.patience, training_session.loss_delta
    )
    logger.info(f"training started, session checkpoints saved at {checkpoint_path}")

    num_epochs_length = len(str(num_epochs))

    if training_session.drop_last:
        num_batches = int(training_session.training_size / training_session.batch_size)
    else:
        num_batches = math.ceil(
            training_session.training_size / training_session.batch_size
        )
    num_batches_length = len(str(num_batches))

    if not hasattr(training_session, "average_training_losses"):
        training_session.average_training_losses = []

    if not hasattr(training_session, "average_validation_losses"):
        training_session.average_validation_losses = []

    training_session.epoch = training_session.num_complete_epochs + 1
    for epoch in range(training_session.epoch, num_epochs + 1):
        training_session.epoch = epoch

        # training
        ########################################################################
        training_losses = []

        # set the network in training mode
        network.train()
        for batch_number, (inputs, labels) in enumerate(training_loader, start=1):
            epoch_end = batch_number == num_batches

            # inputs.shape: torch.Size([batch_size, sequence_length, num_protein_letters])
            # e.g. torch.Size([512, 1000, 27])
            # inputs[i].shape: torch.Size([sequence_length, num_protein_letters])
            # e.g. torch.Size([1000, 27])
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # zero accumulated gradients
            network.zero_grad()

            # get network output and hidden state
            output = network(inputs)

            with torch.no_grad():
                # get class indexes from the one-hot encoded labels
                labels = torch.argmax(labels, dim=1)

            # calculate the training loss
            training_loss = criterion(output, labels)
            training_losses.append(training_loss.item())
            summary_writer.add_scalar("loss/training", training_loss, epoch)

            # perform back propagation
            training_loss.backward()

            # prevent the exploding gradient problem
            nn.utils.clip_grad_norm_(network.parameters(), clip_max_norm)

            # perform an optimization step
            training_session.optimizer.step()

            if verbose and not epoch_end:
                average_training_loss = np.average(training_losses)

                training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, batch {batch_number:{num_batches_length}} of {num_batches} | average training loss: {average_training_loss:.4f}"
                logger.info(training_progress)

        training_session.num_complete_epochs += 1

        average_training_loss = np.average(training_losses)
        training_session.average_training_losses.append(average_training_loss)

        # validation
        ########################################################################
        validation_losses = []

        # disable gradient calculation
        with torch.no_grad():
            # set the network in evaluation mode
            network.eval()

            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

                output = network(inputs)
                labels = torch.argmax(labels, dim=1)
                validation_loss = criterion(output, labels)
                validation_losses.append(validation_loss.item())
                summary_writer.add_scalar("loss/validation", validation_loss, epoch)

        average_validation_loss = np.average(validation_losses)
        training_session.average_validation_losses.append(average_validation_loss)

        training_progress = f"epoch {epoch:{num_epochs_length}} of {num_epochs}, "
        if verbose:
            training_progress += (
                f"batch {batch_number:{num_batches_length}} of {num_batches} "
            )
        training_progress += f"| average training loss: {average_training_loss:.4f}, average validation loss: {average_validation_loss:.4f}"
        logger.info(training_progress)

        if stop_early(network, training_session, average_validation_loss):
            summary_writer.flush()
            summary_writer.close()

            break


def test_network(
    network, training_session, test_loader, log_file_path, print_sample_predictions=False
):
    """
    Calculate test loss and generate metrics.
    """
    criterion = training_session.criterion

    if training_session.drop_last:
        num_batches = int(training_session.test_size / training_session.batch_size)
    else:
        num_batches = math.ceil(training_session.test_size / training_session.batch_size)
    num_batches_length = len(str(num_batches))

    test_losses = []
    num_correct_predictions = 0

    with torch.no_grad():
        network.eval()

        num_samples = 0
        for batch_number, (inputs, labels) in enumerate(test_loader, start=1):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            output = network(inputs)

            # get predicted labels from output
            predicted_probabilities = torch.exp(output)
            predictions = torch.argmax(predicted_probabilities, dim=1)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

            # calculate test loss
            test_loss = criterion(output, labels)
            test_losses.append(test_loss.item())

            # predictions to ground truth comparison
            predictions_correctness = predictions.eq(labels)
            num_correct_predictions += torch.sum(predictions_correctness).item()

            num_samples += len(predictions)
            running_test_accuracy = num_correct_predictions / num_samples

            logger.info(
                f"batch {batch_number:{num_batches_length}} of {num_batches} | running test accuracy: {running_test_accuracy:.4f}"
            )

    # log statistics

    # reset logger, add raw messages format
    logger.remove()
    logger.add(sys.stderr, format="{message}")
    logger.add(log_file_path, format="{message}")

    logger.info("\n" + "average test loss: {:.4f}".format(np.mean(test_losses)))

    # test predictions accuracy
    test_accuracy = num_correct_predictions / num_samples
    logger.info("test accuracy: {:.3f}".format(test_accuracy) + "\n")

    if print_sample_predictions:
        # num_sample_predictions = 10
        # num_sample_predictions = 20
        num_sample_predictions = 100

        with torch.no_grad():
            network.eval()

            inputs, labels = next(iter(test_loader))
            # inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            with torch.random.fork_rng():
                torch.manual_seed(time.time() * 1000)
                permutation = torch.randperm(len(inputs))

            inputs = inputs[permutation[0:num_sample_predictions]]
            labels = labels[permutation[0:num_sample_predictions]]

            output = network(inputs)

            # get predicted labels from output
            predicted_probabilities = torch.exp(output)
            predictions = torch.argmax(predicted_probabilities, dim=1)

            # get class indexes from the one-hot encoded labels
            labels = torch.argmax(labels, dim=1)

        predictions = network.gene_symbols.one_hot_encoding_to_symbol(predictions)
        labels = network.gene_symbols.one_hot_encoding_to_symbol(labels)

        logger.info("sample predictions")
        logger.info("prediction | true label")
        logger.info("-----------------------")
        for prediction, label in zip(predictions, labels):
            if prediction == label:
                logger.info(f"{prediction:>10} | {label:>10}")
            else:
                logger.info(f"{prediction:>10} | {label:>10}  !!!")


def save_network_from_checkpoint(checkpoint_path):
    """
    Save a network residing inside a checkpoint file as a separate file.
    """
    checkpoint = load_checkpoint(checkpoint_path)
    network = checkpoint["network"]

    path = checkpoint_path
    network_path = pathlib.Path(f"{path.parent}/{path.stem}_network.pth")

    torch.save(network, network_path)

    return network_path


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
        "-ex",
        "--experiment_settings",
        help="path to the experiment settings configuration file",
    )
    argument_parser.add_argument(
        "--load_checkpoint",
        help="path to the training session checkpoint to load",
    )
    argument_parser.add_argument("--train", action="store_true", help="train a network")
    argument_parser.add_argument("--test", action="store_true", help="test a network")
    argument_parser.add_argument(
        "--predict_fasta",
        help="path of FASTA file with protein sequences to generate symbol predictions for",
    )
    argument_parser.add_argument(
        "--predictions_csv",
        help="path of CSV file to save the generated symbol predictions",
    )
    argument_parser.add_argument("--save_network")

    args = argument_parser.parse_args()

    # load experiment settings and generate the log file path
    if args.experiment_settings:
        with open(args.experiment_settings) as f:
            experiment = PrettySimpleNamespace(**yaml.safe_load(f))

        if args.datetime is None:
            datetime = dt.datetime.now().isoformat(sep="_", timespec="seconds")
        else:
            datetime = args.datetime

        log_file_path = (
            experiments_directory / f"n={experiment.num_symbols}_{datetime}.log"
        )
    elif args.load_checkpoint:
        log_file_path = pathlib.Path(args.load_checkpoint).with_suffix(".log")
    else:
        argument_parser.print_help()
        sys.exit()

    # set up logger
    logger.remove()
    logger.add(sys.stderr, format=LOGURU_FORMAT)
    logger.add(log_file_path, format=LOGURU_FORMAT)

    # save network
    if args.save_network:
        checkpoint_path = pathlib.Path(args.save_network)
        logger.info(f'Loading checkpoint "{checkpoint_path}" ...')
        network_path = save_network_from_checkpoint(checkpoint_path)
        logger.info(f'Saved network at "{network_path}"')
        return

    if args.predict_fasta:
        fasta_path = args.predict_fasta
        csv_path = args.predictions_csv

        checkpoint_path = pathlib.Path(args.load_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]

        logger.info("generating predictions...")

        # number of sequences in each chunk of the FASTA file read
        num_sequences_chunk = 1024
        # Count the number of sequences in the FASTA file up to the maximum
        # of the num_sequences_chunk chunk size. If the FASTA file has fewer entries
        # than num_sequences_chunk, re-assign the latter to that smaller value.
        with open(fasta_path) as fasta_file:
            num_entries_counter = 0
            for _ in SeqIO.FastaIO.SimpleFastaParser(fasta_file):
                num_entries_counter += 1
                if num_entries_counter == num_sequences_chunk:
                    break
            else:
                num_sequences_chunk = num_entries_counter
        logger.info(f"{num_sequences_chunk=}")

        # read the FASTA file in chunks and generate predictions
        with open(fasta_path) as fasta_file, open(csv_path, "w+") as csv_file:
            fasta_generator = SeqIO.FastaIO.SimpleFastaParser(fasta_file)
            args = [fasta_generator] * num_sequences_chunk
            fasta_chunks_iterator = itertools.zip_longest(*args)

            # generate a csv writer, create the CSV file with a header
            field_names = ["stable_id", "symbol"]
            csv_writer = csv.writer(csv_file, delimiter="\t")
            csv_writer.writerow(field_names)

            for fasta_entries in fasta_chunks_iterator:
                if fasta_entries[-1] is None:
                    fasta_entries = [
                        fasta_entry
                        for fasta_entry in fasta_entries if fasta_entry is not None
                    ]

                stable_ids = [
                    fasta_entry[0].split(" ")[0] for fasta_entry in fasta_entries
                ]
                sequences = [fasta_entry[1] for fasta_entry in fasta_entries]

                start = time.time()
                predictions = network.predict(sequences)
                end = time.time()
                inference_duration = end - start
                logger.debug(f"inference call took {inference_duration:.3f} seconds")

                # write predictions to the CSV file
                csv_writer.writerows(zip(stable_ids, predictions))
        logger.debug(f"predictions saved at {csv_path}")
        sys.exit()

    # log PyTorch version information
    logger.info(f"{torch.__version__=}")
    # specify GPU devices visible to CUDA applications
    # https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    logger.info(f"{DEVICE=}")

    logger.debug(f"{torch.version.cuda=}")
    logger.debug(f"{torch.backends.cudnn.enabled=}")
    logger.debug(f"{torch.cuda.is_available()=}")
    if torch.cuda.is_available():
        logger.debug(f"{torch.cuda.device_count()=}")
        logger.debug(f"{torch.cuda.get_device_properties(DEVICE)}")
        # logger.debug(f"{torch.cuda.memory_summary(DEVICE)}")

    # load training checkpoint or generate new training session
    if args.load_checkpoint:
        checkpoint_path = pathlib.Path(args.load_checkpoint)
        checkpoint = load_checkpoint(checkpoint_path)
        network = checkpoint["network"]
        training_session = checkpoint["training_session"]
    else:
        if experiment.num_symbols == 3:
            test_ratio = 0.2
            validation_ratio = 0.2
        elif experiment.num_symbols in {101, 1013}:
            test_ratio = 0.1
            validation_ratio = 0.1
        else:
            test_ratio = 0.05
            validation_ratio = 0.05

        # larger patience for short epochs and smaller patience for longer epochs
        if experiment.num_symbols in {3, 101, 1013}:
            patience = 11
        else:
            patience = 7

        training_session = TrainingSession(
            experiment.num_symbols,
            datetime,
            experiment.random_state,
            test_ratio,
            validation_ratio,
            experiment.sequence_length,
            experiment.batch_size,
            experiment.num_connections,
            experiment.dropout_probability,
            experiment.learning_rate,
            experiment.num_epochs,
            patience,
        )

        # loss function
        training_session.criterion = nn.NLLLoss()

    torch.manual_seed(training_session.random_state)

    # load data, generate datasets
    ############################################################################
    dataset = SequenceDataset(
        training_session.num_symbols, training_session.sequence_length
    )

    if not args.load_checkpoint:
        # neural network instantiation
        ############################################################################
        # num_protein_letters = len(dataset.protein_letters)
        num_protein_letters = 27

        network = FullyConnectedNetwork(
            training_session.sequence_length,
            num_protein_letters,
            training_session.num_symbols,
            training_session.num_connections,
            training_session.dropout_probability,
            dataset.gene_symbols,
        )
        ############################################################################
        training_session.device = DEVICE

        network.to(DEVICE)

    pandas_symbols_categories = (
        dataset.gene_symbols.symbol_categorical_datatype.categories
    )
    logger.info(
        "gene symbols:\n{}".format(
            pandas_symbols_categories.to_series(
                index=range(len(pandas_symbols_categories)), name="gene symbols"
            )
        )
    )

    # split dataset into train, validation, and test datasets
    validation_size = int(training_session.validation_ratio * len(dataset))
    test_size = int(training_session.test_ratio * len(dataset))
    training_session.training_size = len(dataset) - validation_size - test_size
    training_session.validation_size = validation_size
    training_session.test_size = test_size

    training_dataset, validation_dataset, test_dataset = random_split(
        dataset,
        lengths=(
            training_session.training_size,
            training_session.validation_size,
            training_session.test_size,
        ),
    )

    logger.info(
        f"dataset split to train ({training_session.training_size}), validation ({training_session.validation_size}), and test ({training_session.test_size}) datasets"
    )

    # set the batch size to the size of the smallest dataset if larger than that
    min_dataset_size = min(
        training_session.training_size,
        training_session.validation_size,
        training_session.test_size,
    )
    if training_session.batch_size > min_dataset_size:
        training_session.batch_size = min_dataset_size

    if training_session.num_symbols in {3, 101}:
        training_session.drop_last = False
    else:
        training_session.drop_last = True
    training_loader = DataLoader(
        training_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=training_session.batch_size,
        shuffle=True,
        drop_last=training_session.drop_last,
    )
    ############################################################################

    logger.info(f"network:\n{network}")
    logger.info(f"training_session:\n{training_session}")

    # train network
    if args.train:
        verbose = True

        train_network(
            network,
            training_session,
            training_loader,
            validation_loader,
            verbose,
        )

    # test trained network
    if args.test:
        if args.train:
            checkpoint_path = experiments_directory / training_session.checkpoint_filename
            checkpoint = load_checkpoint(checkpoint_path)
            network = checkpoint["network"]
            training_session = checkpoint["training_session"]
        test_network(
            network,
            training_session,
            test_loader,
            log_file_path,
            print_sample_predictions=True,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
        sys.exit()
