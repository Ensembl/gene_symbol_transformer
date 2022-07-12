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

# third party imports
import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from torch import nn


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
        # x = self.layer_normalization_1(self.attention(x) + x)
        # x = self.layer_normalization_2(self.feed_forward(x) + x)

        # minGPT
        # x = x + self.attention(self.layer_normalization_1(x))
        # x = x + self.feed_forward(self.layer_normalization_2(x))

        # ws
        x = x + self.layer_normalization_1(self.attention(x))
        x = x + self.layer_normalization_2(self.feed_forward(x))

        return x
