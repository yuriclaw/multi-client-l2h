"""Rejector networks for the L2H deferral decision.

Each client has its own rejector that maps input x → [0, 1] indicating
the probability of deferring to the server (REMOTE) vs. keeping locally (LOCAL).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict


class MLPRejector(nn.Module):
    """2-layer MLP rejector: input → hidden → sigmoid scalar."""

    def __init__(self, input_dim: int = 3072, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return deferral probability (B,)."""
        x = x.view(x.size(0), -1)
        return torch.sigmoid(self.net(x)).squeeze(-1)


class RejectorBank:
    """Manages per-client rejectors.

    Usage:
        bank = RejectorBank(input_dim=3072, hidden_dim=128, device='cuda')
        bank.add_client('client_0')
        r = bank['client_0']          # get the rejector module
        probs = r(x)                   # deferral probabilities
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 128,
        device: str = "cpu",
    ) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self._rejectors: Dict[str, MLPRejector] = {}

    def add_client(self, client_id: str) -> MLPRejector:
        """Create and register a rejector for a client."""
        if client_id not in self._rejectors:
            r = MLPRejector(self.input_dim, self.hidden_dim).to(self.device)
            self._rejectors[client_id] = r
        return self._rejectors[client_id]

    def __getitem__(self, client_id: str) -> MLPRejector:
        return self._rejectors[client_id]

    def __contains__(self, client_id: str) -> bool:
        return client_id in self._rejectors

    def parameters(self, client_id: str):
        """Return trainable parameters for a specific client's rejector."""
        return self._rejectors[client_id].parameters()

    def all_client_ids(self) -> list[str]:
        return list(self._rejectors.keys())
