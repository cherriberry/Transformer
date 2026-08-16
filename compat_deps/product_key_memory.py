"""Import compatibility for configurations with product-key memory disabled."""

import torch.nn as nn


class PKM(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, **kwargs):
        raise RuntimeError("Product-key memory layers are disabled in this benchmark")
