from typing import List, Tuple

import tinygrad.nn as nn
from tinygrad import Tensor
from transformer import Transformer

from mlp import MLP


# Better way to do this?
def split_features(dataset: Tensor, indices: List[int]) -> Tuple[Tensor, Tensor]:
    continuous_indexes = [i for i in list(range(dataset.shape[1])) if i not in indices]

    categorical_features = dataset[:, indices]
    continuous_features = dataset[:, continuous_indexes]

    # return the features in the order (categorical, continuous) so callers can
    # simply unpack as ``x_cat, x_cont = split_features(...)``
    return categorical_features, continuous_features


class TabTransformer:
    def __init__(self, n_cat: int, n_cont: int) -> None:
        self.n_cat = n_cat
        self.n_cont = n_cont
        self.transformer = Transformer(
            syms=65, maxlen=n_cat, layers=6, embed_dim=32, num_heads=8, ff_dim=128
        )
        self.mlp = MLP(l=n_cat * 32 + n_cont)
        self.cont_layer_norm = nn.LayerNorm2d(n_cont)
        # use the same embedding dimension as the transformer for compatibility
        self.cat_embed = nn.Embedding(vocab_size=65, embed_size=32)

    def forward(self, x: Tensor, indices: List[int]) -> Tensor:
        """
        1. Split the data based on categorical/continuous
            1.1 LayerNorm the Continuous Features
            1.2 Column Embed the Categorical Features
        2. Pass the Column Embeddings through the Transformer Blocks
        3. Concatenate Continuous and Column Embeddings
        4. Pass through the MLP
        """
        x_cat, x_cont = split_features(x, indices)

        x_cont_ln = self.cont_layer_norm(x_cont)
        x_cat_ce = self.cat_embed(x_cat)

        transformer_outputs = self.transformer.forward(x_cat_ce)

        transformer_flattened = transformer_outputs.reshape(
            (transformer_outputs.shape[0], -1)
        )

        concat_features = transformer_flattened.cat(x_cont_ln)

        return self.mlp(concat_features)
