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
import sys
import time
import warnings

# third party imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
import yaml

from torch import nn

# project imports
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
    logging_formatter_message,
)


class ClassificationTransformer(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        embedding_dimension,
        num_heads,
        depth,
        feedforward_connections,
        sequence_length,
        num_tokens,
        dropout_probability,
        activation_function,
    ):
        super().__init__()

        self.num_classes = num_classes

        self.token_embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dimension
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=sequence_length, embedding_dim=embedding_dimension
        )

        transformer_blocks = [
            TransformerBlock(
                embedding_dimension=embedding_dimension,
                num_heads=num_heads,
                feedforward_connections=feedforward_connections,
                dropout_probability=dropout_probability,
                activation_function=activation_function,
            )
            for _ in range(depth)
        ]
        self.transformer_blocks = nn.Sequential(*transformer_blocks)

        self.fully_connected_layer = nn.Linear(embedding_dimension, num_classes)

        self.final_activation = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # generate token embeddings
        token_embeddings = self.token_embedding(x)

        b, t, k = token_embeddings.size()

        # generate position embeddings
        position_embeddings_init = torch.arange(t, device=self.device)
        position_embeddings = self.position_embedding(position_embeddings_init)[
            None, :, :
        ].expand(b, t, k)

        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)

        # average-pool over dimension t
        x = x.mean(dim=1)

        x = self.fully_connected_layer(x)

        x = self.final_activation(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, embedding_dimension, num_heads):
        super().__init__()

        assert (
            embedding_dimension % num_heads == 0
        ), f"embedding dimension must be divisible by number of heads, got {embedding_dimension=}, {num_heads=}"

        self.num_heads = num_heads

        k = embedding_dimension

        self.to_keys = nn.Linear(k, k * num_heads, bias=False)
        self.to_queries = nn.Linear(k, k * num_heads, bias=False)
        self.to_values = nn.Linear(k, k * num_heads, bias=False)

        self.unify_heads = nn.Linear(num_heads * k, k)

    def forward(self, x):
        b, t, k = x.size()
        h = self.num_heads

        keys = self.to_keys(x).view(b, t, h, k)
        queries = self.to_queries(x).view(b, t, h, k)
        values = self.to_values(x).view(b, t, h, k)

        # fold heads into the batch dimension
        keys = keys.transpose(1, 2).contiguous().view(b * h, t, k)
        queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
        values = values.transpose(1, 2).contiguous().view(b * h, t, k)

        # get dot product of queries and keys
        dot = torch.bmm(queries, keys.transpose(1, 2))
        # dot.shape: (b * h, t, t)

        # scale dot product
        dot = dot / (k ** (1 / 2))

        # get row-wise normalized weights
        dot = F.softmax(dot, dim=2)

        # apply the self attention to the values
        out = torch.bmm(dot, values).view(b, h, t, k)

        # swap h, t back
        out = out.transpose(1, 2).contiguous().view(b, t, h * k)

        return self.unify_heads(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        num_heads,
        feedforward_connections,
        activation_function,
        dropout_probability,
    ):
        super().__init__()

        assert (
            feedforward_connections > embedding_dimension
        ), f"feed forward subnet number of connections should be larger than the embedding dimension, got {feedforward_connections=}, {embedding_dimension=}"

        activation = {"relu": nn.ReLU, "gelu": nn.GELU}

        k = embedding_dimension

        self.attention = SelfAttention(k, num_heads=num_heads)

        self.layer_normalization_1 = nn.LayerNorm(k)
        self.layer_normalization_2 = nn.LayerNorm(k)

        self.feed_forward = nn.Sequential(
            nn.Linear(k, feedforward_connections),
            activation[activation_function](),
            nn.Linear(feedforward_connections, k),
            nn.Dropout(dropout_probability),
        )

    def forward(self, x):
        x = self.layer_normalization_1(self.attention(x) + x)
        x = self.layer_normalization_2(self.feed_forward(x) + x)

        return x


class GeneSymbolClassifier(ClassificationTransformer):
    """
    Neural network for gene symbol classification of protein coding sequences using
    the raw protein letters as features.
    """

    def __init__(self, **kwargs):
        self.save_hyperparameters()

        super().__init__(
            num_classes=self.hparams.num_symbols,
            embedding_dimension=self.hparams.embedding_dimension,
            num_heads=self.hparams.num_heads,
            depth=self.hparams.transformer_depth,
            feedforward_connections=self.hparams.feedforward_connections,
            sequence_length=self.hparams.sequence_length,
            num_tokens=self.hparams.num_protein_letters,
            dropout_probability=self.hparams.dropout_probability,
            activation_function=self.hparams.activation_function,
        )

        self.sequence_length = self.hparams.sequence_length
        self.padding_side = self.hparams.padding_side
        self.num_symbols = self.hparams.num_symbols

        self.protein_sequence_mapper = self.hparams.protein_sequence_mapper
        self.clade_mapper = self.hparams.clade_mapper
        self.symbol_mapper = self.hparams.symbol_mapper

        self.num_sample_predictions = self.hparams.num_sample_predictions

        self.best_validation_accuracy = 0

        self.torchmetrics_accuracy_average = "weighted"

    def on_pretrain_routine_end(self):
        logger.info("start network training")
        logger.info(f"configuration:\n{self.hparams}")

    def training_step(self, batch, batch_index):
        sequence_features, clade_features, labels = batch

        # forward pass
        output = self(sequence_features)

        # loss function
        training_loss = F.nll_loss(output, labels)
        self.log("training_loss", training_loss)

        # clip gradients to prevent the exploding gradient problem
        if self.hparams.clip_max_norm > 0:
            nn.utils.clip_grad_norm_(self.parameters(), self.hparams.clip_max_norm)

        return training_loss

    def on_validation_start(self):
        # https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metrics-and-devices
        self.validation_accuracy = torchmetrics.Accuracy(
            num_classes=self.num_symbols, average=self.torchmetrics_accuracy_average
        ).to(self.device)

    def validation_step(self, batch, batch_index):
        sequence_features, clade_features, labels = batch

        # forward pass
        output = self(sequence_features)

        # loss function
        validation_loss = F.nll_loss(output, labels)
        self.log("validation_loss", validation_loss)

        # get predicted label indexes from output
        predictions, _ = self.get_prediction_indexes_probabilities(output)

        self.validation_accuracy(predictions, labels)

        return validation_loss

    def on_validation_end(self):
        self.best_validation_accuracy = max(
            self.best_validation_accuracy,
            self.validation_accuracy.compute().item(),
        )

    def on_train_end(self):
        # NOTE: disabling saving network to TorchScript, seems buggy

        # workaround for a bug when saving network to TorchScript format
        # self.hparams.dropout_probability = float(self.hparams.dropout_probability)

        # save network to TorchScript format
        # experiment_directory_path = pathlib.Path(self.hparams.experiment_directory)
        # torchscript_path = experiment_directory_path / "torchscript_network.pt"
        # torchscript = self.to_torchscript()
        # torch.jit.save(torchscript, torchscript_path)
        pass

    def on_test_start(self):
        self.test_accuracy = torchmetrics.Accuracy(
            num_classes=self.num_symbols, average=self.torchmetrics_accuracy_average
        ).to(self.device)
        self.test_precision = torchmetrics.Precision(
            num_classes=self.num_symbols, average=self.torchmetrics_accuracy_average
        ).to(self.device)
        self.test_recall = torchmetrics.Recall(
            num_classes=self.num_symbols, average=self.torchmetrics_accuracy_average
        ).to(self.device)

        self.sample_labels = torch.empty(0).to(self.device)
        self.sample_predictions = torch.empty(0).to(self.device)

    def test_step(self, batch, batch_index):
        sequence_features, clade_features, labels = batch

        # forward pass
        output = self(sequence_features)

        # get predicted label indexes from output
        predictions, _ = self.get_prediction_indexes_probabilities(output)

        self.test_accuracy(predictions, labels)
        self.test_precision(predictions, labels)
        self.test_recall(predictions, labels)

        if self.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(len(labels))

            sample_labels = labels[permutation[0 : self.num_sample_predictions]]
            sample_predictions = predictions[permutation[0 : self.num_sample_predictions]]

            self.sample_labels = torch.cat((self.sample_labels, sample_labels))
            self.sample_predictions = torch.cat(
                (self.sample_predictions, sample_predictions)
            )

    def on_test_end(self):
        # log statistics
        accuracy = self.test_accuracy.compute()
        precision = self.test_precision.compute()
        recall = self.test_recall.compute()
        logger.info(
            f"test accuracy: {accuracy:.4f} | precision: {precision:.4f} | recall: {recall:.4f}"
        )
        logger.info(f"(best validation accuracy: {self.best_validation_accuracy:.4f})")

        if self.num_sample_predictions > 0:
            with torch.random.fork_rng():
                torch.manual_seed(int(time.time() * 1000))
                permutation = torch.randperm(len(self.sample_labels))

            self.sample_labels = self.sample_labels[
                permutation[0 : self.num_sample_predictions]
            ].tolist()
            self.sample_predictions = self.sample_predictions[
                permutation[0 : self.num_sample_predictions]
            ].tolist()

            # change logging format to raw messages
            for handler in logger.handlers:
                handler.setFormatter(logging_formatter_message)

            labels = [
                self.symbol_mapper.index_to_label(label) for label in self.sample_labels
            ]
            assignments = [
                self.symbol_mapper.index_to_label(prediction)
                for prediction in self.sample_predictions
            ]

            logger.info("\nsample assignments")
            logger.info("assignment | true label")
            logger.info("-----------------------")
            for assignment, label in zip(assignments, labels):
                if assignment == label:
                    logger.info(f"{assignment:>10} | {label:>10}")
                else:
                    logger.info(f"{assignment:>10} | {label:>10}  !!!")

    def configure_optimizers(self):
        # optimization function
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

    def predict_probabilities(self, sequences, clades):
        """
        Get symbol predictions for a list of protein sequences, along with
        the probabilities of predictions.
        """
        sequence_features, _clade_features = self.generate_features_tensor(
            sequences, clades
        )

        # run inference
        with torch.no_grad():
            self.eval()
            output = self.forward(sequence_features)

        prediction_indexes, probabilities = self.get_prediction_indexes_probabilities(
            output
        )

        predictions = [
            self.symbol_mapper.index_to_label(prediction.item())
            for prediction in prediction_indexes
        ]

        predictions_probabilities = [
            (prediction, probability.item())
            for prediction, probability in zip(predictions, probabilities)
        ]

        return predictions_probabilities

    @staticmethod
    def get_prediction_indexes_probabilities(output):
        """
        Get predicted labels from network's forward pass output, along with
        the probabilities of predictions.
        """
        predicted_probabilities = torch.exp(output)
        # get class indexes from the one-hot encoded labels
        predictions = torch.argmax(predicted_probabilities, dim=1)
        # get max probability
        probabilities, _indices = torch.max(predicted_probabilities, dim=1)
        return (predictions, probabilities)

    def generate_features_tensor(self, sequences, clades):
        """
        Convert lists of protein sequences and species clades to a label encoded sequence
        features tensor and an one-hot encoded clade features tensor.
        """
        padding_side_to_align = {"left": ">", "right": "<"}

        sequence_features_list = []
        clade_features_list = []
        for sequence, clade in zip(sequences, clades):
            # pad or truncate sequence to be exactly `self.sequence_length` letters long
            sequence = "{string:{align}{string_length}.{truncate_length}}".format(
                string=sequence,
                align=padding_side_to_align[self.padding_side],
                string_length=self.sequence_length,
                truncate_length=self.sequence_length,
            )

            label_encoded_sequence = (
                self.protein_sequence_mapper.sequence_to_label_encoding(sequence)
            )
            one_hot_clade = self.clade_mapper.label_to_one_hot(clade)

            sequence_features_list.append(label_encoded_sequence)
            clade_features_list.append(one_hot_clade)

        sequence_features = np.stack(sequence_features_list)
        clade_features = np.stack(clade_features_list)

        sequence_features_tensor = torch.from_numpy(sequence_features)
        clade_features_tensor = torch.from_numpy(clade_features)

        return (sequence_features_tensor, clade_features_tensor)


def get_item_sequence_label_clade_one_hot(self, index):
    """
    Modularized Dataset __getitem__ method.

    Generate a protein sequence label encoding and a species clade one-hot encoding and
    use as separate feature vectors.

    Args:
        self (Dataset): the Dataset object that will contain __getitem__
    Returns:
        tuple containing the features vector and symbol index
    """
    dataset_row = self.dataset.iloc[index].to_dict()

    sequence = dataset_row["sequence"]
    clade = dataset_row["clade"]
    symbol = dataset_row["symbol"]

    label_encoded_sequence = self.protein_sequence_mapper.sequence_to_label_encoding(
        sequence
    )
    # label_encoded_sequence.shape: (sequence_length,)

    one_hot_clade = self.clade_mapper.label_to_one_hot(clade)
    # one_hot_clade.shape: (num_clades,)

    symbol_index = self.symbol_mapper.label_to_index(symbol)

    item = (label_encoded_sequence, one_hot_clade, symbol_index)

    return item


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
        "--checkpoint",
        help="experiment checkpoint path",
    )
    argument_parser.add_argument(
        "--train", action="store_true", help="train a classifier"
    )
    argument_parser.add_argument("--test", action="store_true", help="test a classifier")
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

        if "num_symbols" in configuration:
            assert (
                "min_frequency" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.num_symbols}_num_symbols"
        elif "min_frequency" in configuration:
            assert (
                "num_symbols" not in configuration
            ), "num_symbols and min_frequency are mutually exclusive, provide only one of them in the configuration"
            configuration.dataset_id = f"{configuration.min_frequency}_min_frequency"
        else:
            raise KeyError(
                'missing configuration value: one of "num_symbols", "min_frequency" is required'
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
        ) = generate_dataloaders(configuration, get_item_sequence_label_clade_one_hot)

        if configuration.num_symbols < 1000:
            configuration.symbols_metadata = None
        else:
            # load symbols metadata
            symbols_metadata_filename = "symbols_metadata.json"
            symbols_metadata_path = data_directory / symbols_metadata_filename
            with open(symbols_metadata_path) as file:
                configuration.symbols_metadata = ConciseReprDict(json.load(file))

        # instantiate neural network
        network = GeneSymbolClassifier(**configuration)

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

        trainer = pl.Trainer(
            gpus=configuration.gpus,
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

        network = GeneSymbolClassifier.load_from_checkpoint(args.checkpoint)

        _, _, test_dataloader = generate_dataloaders(
            network.hparams, get_item_sequence_label_clade_one_hot
        )

        trainer = pl.Trainer()
        trainer.test(network, dataloaders=test_dataloader)

    # evaluate a classifier
    elif args.evaluate and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)
        evaluation_directory_path = (
            checkpoint_path.parent / f"{checkpoint_path.stem}_evaluation"
        )
        evaluation_directory_path.mkdir(exist_ok=True)
        log_file_path = evaluation_directory_path / f"{checkpoint_path.stem}_evaluate.log"
        add_log_file_handler(logger, log_file_path)

        network = GeneSymbolClassifier.load_from_checkpoint(checkpoint_path)

        evaluate_network(network, checkpoint_path, args.complete)

    # assign symbols to sequences
    elif args.sequences_fasta and args.scientific_name and args.checkpoint:
        checkpoint_path = pathlib.Path(args.checkpoint)

        log_file_path = f"{checkpoint_path.parent}/experiment.log"
        add_log_file_handler(logger, log_file_path)

        network = GeneSymbolClassifier.load_from_checkpoint(args.checkpoint)
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
            network = GeneSymbolClassifier.load_from_checkpoint(args.checkpoint)
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
        sys.exit()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interrupted with CTRL-C, exiting...")
