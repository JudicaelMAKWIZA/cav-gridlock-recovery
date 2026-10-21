"""TCN causal à convolutions dilatées ; aucun état futur ni oracle en entrée."""

import torch
from torch import nn
from torch.nn import functional as F


class CausalBlock(nn.Module):
    def __init__(self, channels, dilation, kernel_size=3):
        super().__init__()
        self.left = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x, mask):
        transformed = F.gelu(self.convolution(F.pad(x, (self.left, 0))))
        return (x + transformed) * mask


class TemporalClassifier(nn.Module):
    """Le logit final estime une confirmation disponible maintenant, sans la prouver."""

    def __init__(
        self,
        features,
        hidden=8,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        kernel_size=3,
        architecture="tcn",
    ):
        super().__init__()
        self.architecture = architecture
        self.projection = nn.Conv1d(features + 1, hidden, 1)
        self.blocks = (
            nn.ModuleList(CausalBlock(hidden, d, kernel_size) for d in dilations)
            if architecture == "tcn"
            else nn.ModuleList()
        )
        self.output = nn.Conv1d(hidden, 1, 1)

    def sequence_logits(self, x, mask):
        x = F.gelu(self.projection(torch.cat([x, mask], dim=1))) * mask
        for block in self.blocks:
            x = block(x, mask)
        return self.output(x).squeeze(1)

    def forward(self, x, mask):
        return self.sequence_logits(x, mask)[:, -1]


def create_model(config, features):
    return TemporalClassifier(
        features,
        hidden=config["hidden"],
        dilations=config["dilations"],
        kernel_size=config["kernel_size"],
        architecture=config.get("architecture", "tcn"),
    )
