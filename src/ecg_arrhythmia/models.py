"""1D CNN beat classifier: waveform conv branch + RR-interval branch."""

from __future__ import annotations

import torch
from torch import nn

from . import config


class BeatCNN(nn.Module):
    """Convolutional branch over the beat waveform, concatenated with an
    RR-interval branch, followed by a small classification head."""

    def __init__(self, n_classes: int = len(config.AAMI_CLASSES),
                 n_rr_feats: int = config.N_RR_FEATURES,
                 dropout: float = config.DROPOUT) -> None:
        super().__init__()
        self.waveform = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )
        self.rr_branch = nn.Sequential(
            nn.Linear(n_rr_feats, 32), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 32, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

    def forward(self, wave: torch.Tensor, rr: torch.Tensor) -> torch.Tensor:
        """wave: (batch, BEAT_LEN); rr: (batch, N_RR_FEATURES)."""
        f = self.waveform(wave.unsqueeze(1))
        g = self.rr_branch(rr)
        return self.head(torch.cat([f, g], dim=1))
