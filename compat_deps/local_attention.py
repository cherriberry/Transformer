"""Import compatibility for configurations with local attention disabled."""

import torch.nn as nn


class LocalAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, *args, **kwargs):
        raise RuntimeError("Local attention is disabled in this benchmark configuration")
