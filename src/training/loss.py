"""Stage-switching surrogate loss for Learning to Help.

The total L2H surrogate loss decomposes into two stages:
  L_S = L1 (adapter loss) + L2 (rejector loss)

Stage 1 — L1: Cross-entropy on server predictions for deferred samples,
  training the LoRA adapter to be accurate on samples the rejector sends.

Stage 2 — L2: Cost-sensitive BCE for the rejector, balancing:
  - c_e: cost of a wrong prediction (by either client or server)
  - c_1: cost of deferral (communication / latency)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdapterLoss(nn.Module):
    """L1: Cross-entropy on server predictions for deferred samples.

    Only samples with r(x) ≈ 1 (deferred) contribute.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        server_logits: torch.Tensor,  # (B, C)
        labels: torch.Tensor,          # (B,)
        defer_probs: torch.Tensor,     # (B,)  from rejector
    ) -> torch.Tensor:
        """Weighted CE where weight = deferral probability."""
        ce = F.cross_entropy(server_logits, labels, reduction="none")  # (B,)
        # Weight by deferral probability (soft routing)
        loss = (defer_probs * ce).mean()
        return loss


class RejectorLoss(nn.Module):
    """L2: Cost-sensitive BCE for the rejector.

    The rejector outputs r(x) ∈ [0,1]:
      r(x) → 1 means DEFER (send to server)
      r(x) → 0 means LOCAL (keep at client)

    Target:
      defer is beneficial when server is correct and client is wrong,
      weighted by the cost structure.
    """

    def __init__(self, c_e: float = 1.0, c_1: float = 0.1) -> None:
        super().__init__()
        self.c_e = c_e
        self.c_1 = c_1

    def forward(
        self,
        defer_probs: torch.Tensor,       # (B,)
        client_logits: torch.Tensor,      # (B, C)
        server_logits: torch.Tensor,      # (B, C)
        labels: torch.Tensor,             # (B,)
    ) -> torch.Tensor:
        """Cost-sensitive surrogate loss for the rejector."""
        client_correct = (client_logits.argmax(dim=1) == labels).float()
        server_correct = (server_logits.argmax(dim=1) == labels).float()

        # Cost of local processing: c_e if client is wrong, 0 if correct
        cost_local = self.c_e * (1.0 - client_correct)
        # Cost of deferral: c_1 + c_e if server is also wrong
        cost_defer = self.c_1 + self.c_e * (1.0 - server_correct)

        # Target: defer when cost_defer < cost_local
        # Surrogate: weighted BCE
        # r(x) * cost_defer + (1 - r(x)) * cost_local
        loss = defer_probs * cost_defer + (1.0 - defer_probs) * cost_local
        return loss.mean()


class L2HSurrogateLoss(nn.Module):
    """Combined L2H surrogate: L_S = L1 + λ * L2."""

    def __init__(self, c_e: float = 1.0, c_1: float = 0.1, lam: float = 1.0) -> None:
        super().__init__()
        self.adapter_loss = AdapterLoss()
        self.rejector_loss = RejectorLoss(c_e=c_e, c_1=c_1)
        self.lam = lam

    def forward(
        self,
        server_logits: torch.Tensor,
        client_logits: torch.Tensor,
        labels: torch.Tensor,
        defer_probs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (total_loss, L1, L2)."""
        l1 = self.adapter_loss(server_logits, labels, defer_probs)
        l2 = self.rejector_loss(defer_probs, client_logits, server_logits, labels)
        total = l1 + self.lam * l2
        return total, l1, l2
