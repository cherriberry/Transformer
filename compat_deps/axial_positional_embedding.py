"""Import compatibility for configurations with axial embeddings disabled."""

import torch.nn as nn


class AxialPositionalEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        raise RuntimeError("Axial positional embeddings are disabled in this benchmark")
