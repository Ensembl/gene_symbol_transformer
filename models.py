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
A Transformer encoder implementation.
"""

# standard library imports
import time

from typing import List, Optional

# third party imports
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics

from torch import nn

# project imports
from utils import logger, logging_formatter_message


class GSC(pl.LightningModule):
    """
    Neural network for gene symbol classification of protein coding sequences.
    """

    def __init__(self, **kwargs):
        self.save_hyperparameters()

        super().__init__()

        self.num_protein_letters = self.hparams.num_protein_letters
        self.sequence_length = self.hparams.sequence_length
        self.padding_side = self.hparams.padding_side
        self.num_symbols = self.hparams.num_symbols
        self.clade = self.hparams.clade
        self.num_clades = self.hparams.num_clades
        self.mlp_output_size = self.hparams.mlp_output_size
        self.activation_function = self.hparams.activation_function

        flat_one_hot_sequence_length = self.num_protein_letters * self.sequence_length
        self.mlp = MLP(
            input_dim=flat_one_hot_sequence_length,
            output_dim=self.mlp_output_size,
            activation=self.activation_function,
        )

        self.sequence_transformer = TransformerEncoder(
            embedding_dimension=self.hparams.embedding_dimension,
            num_heads=self.hparams.num_heads,
            depth=self.hparams.transformer_depth,
            feedforward_connections=self.hparams.feedforward_connections,
            sequence_length=self.sequence_length,
            num_tokens=self.num_protein_letters,
            dropout_probability=self.hparams.dropout_probability,
            activation_function=self.activation_function,
        )

        integration_input_size = (
            self.mlp_output_size + self.sequence_length + self.num_clades
        )
        self.integration_layer = nn.Linear(
            in_features=integration_input_size, out_features=self.num_symbols
        )

        self.final_activation = nn.LogSoftmax(dim=1)

        self.protein_sequence_mapper = self.hparams.protein_sequence_mapper
        self.clade_mapper = self.hparams.clade_mapper
        self.symbol_mapper = self.hparams.symbol_mapper

        self.num_sample_predictions = self.hparams.num_sample_predictions

        self.best_validation_accuracy = 0

        self.torchmetrics_accuracy_average = "weighted"

    def forward(self, features):
        label_encoded_sequence = features["label_encoded_sequence"]
        flat_one_hot_sequence = features["flat_one_hot_sequence"]
        clade_features = features["clade_features"]

        x_mlp = self.mlp(flat_one_hot_sequence)
        x_transformer = self.sequence_transformer(label_encoded_sequence)

        # concatenate the mlp and transformer outputs and clade_features tensors
        x = torch.cat((x_mlp, x_transformer, clade_features), dim=1)

        x = self.integration_layer(x)

        x = self.final_activation(x)

        return x

    def on_fit_start(self):
        logger.info("start network training")
        logger.info(f"configuration:\n{self.hparams}")

    def training_step(self, batch, batch_index):
        features, labels = batch

        # forward pass
        output = self(features)

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
        features, labels = batch

        # forward pass
        output = self(features)

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
        features, labels = batch

        # forward pass
        output = self(features)

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
            sample_predictions = predictions[
                permutation[0 : self.num_sample_predictions]
            ]

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
        for sequence, clade in zip(sequences, clades):
            features = self.generate_sequence_features(sequence)

            if self.clade:
                clade_features = self.clade_mapper.label_to_one_hot(clade)
            else:
                # generate a null clade features tensor
                clade_features = torch.zeros(self.num_clades)

            features["clade_features"] = clade_features

            for feature in features:
                features[feature] = torch.unsqueeze(features[feature], 0)

            # run inference
            with torch.no_grad():
                self.eval()
                # forward pass
                output = self(features)

            (
                prediction_indexes,
                probabilities,
            ) = self.get_prediction_indexes_probabilities(output)

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

    def generate_sequence_features(self, sequence: str):
        """
        Generate features for a protein sequence.

        Args:
            sequence: a protein sequence
        """
        padding_side_to_align = {"left": ">", "right": "<"}

        # pad or truncate sequence to be exactly `self.sequence_length` letters long
        sequence = "{string:{align}{string_length}.{truncate_length}}".format(
            string=sequence,
            align=padding_side_to_align[self.padding_side],
            string_length=self.sequence_length,
            truncate_length=self.sequence_length,
        )

        sequence_features = self._generate_sequence_features(sequence)

        return sequence_features

    def _generate_sequence_features(self, sequence: str):
        label_encoded_sequence = (
            self.protein_sequence_mapper.sequence_to_label_encoding(sequence)
        )
        # label_encoded_sequence.shape: (sequence_length,)

        one_hot_sequence = self.protein_sequence_mapper.sequence_to_one_hot(sequence)
        # one_hot_sequence.shape: (sequence_length, num_protein_letters)

        # flatten sequence matrix to a vector
        flat_one_hot_sequence = torch.flatten(one_hot_sequence)
        # flat_one_hot_sequence.shape: (sequence_length * num_protein_letters,)

        sequence_features = {
            "label_encoded_sequence": label_encoded_sequence,
            "flat_one_hot_sequence": flat_one_hot_sequence,
        }

        return sequence_features


class TransformerEncoder(pl.LightningModule):
    def __init__(
        self,
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

        self.token_embedding = nn.Embedding(
            num_embeddings=num_tokens, embedding_dim=embedding_dimension
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, embedding_dimension)
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

        self.final_layer = nn.Linear(embedding_dimension, sequence_length)

    def forward(self, x):
        token_embeddings = self.token_embedding(x)

        # generate token embeddings
        token_embeddings = self.token_embedding(x)

        b, t, k = token_embeddings.size()

        # generate position embeddings
        # each position maps to a (learnable) vector
        position_embeddings = self.position_embedding[:, :t, :]

        x = token_embeddings + position_embeddings

        x = self.transformer_blocks(x)

        # average-pool over dimension t
        x = x.mean(dim=1)

        x = self.final_layer(x)

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
        # former
        x = self.layer_normalization_1(self.attention(x) + x)
        x = self.layer_normalization_2(self.feed_forward(x) + x)

        # minGPT
        # x = x + self.attention(self.layer_normalization_1(x))
        # x = x + self.feed_forward(self.layer_normalization_2(x))

        # ws
        # x = x + self.layer_normalization_1(self.attention(x))
        # x = x + self.layer_normalization_2(self.feed_forward(x))

        return x


class MLP(nn.Module):
    """
    Parametrizable multi-layer perceptron.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dimensions: Optional[List[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()

        if hidden_dimensions is None:
            hidden_dimensions = []

        self.layers = nn.ModuleList(
            nn.Linear(in_dim, out_dim)
            for in_dim, out_dim in zip(
                [input_dim] + hidden_dimensions, hidden_dimensions + [output_dim]
            )
        )

        self.activation_function = {"relu": F.relu, "gelu": F.gelu}[activation]

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = activation_function(layer(x))

        x = self.layers[-1](x)

        return x
