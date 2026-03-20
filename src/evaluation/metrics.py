"""Evaluation metrics for the L2H system.

Key metrics:
- System accuracy: accuracy of the combined client + server system with deferral
- Deferral rate: fraction of inputs sent to the server
- Per-client accuracy: individual client/server accuracy
- Personalization gain: improvement from LoRA personalization
"""

from __future__ import annotations

from typing import Dict

import torch


def compute_client_metrics(
    labels: torch.Tensor,         # (N,)
    client_preds: torch.Tensor,   # (N,)
    server_preds: torch.Tensor,   # (N,)
    defer_probs: torch.Tensor,    # (N,) ∈ [0,1]
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics for a single client.

    Args:
        labels: Ground-truth labels.
        client_preds: Predicted classes from the client model.
        server_preds: Predicted classes from the server model.
        defer_probs: Deferral probabilities from the rejector.
        threshold: Decision threshold for hard deferral.

    Returns:
        Dict of metric name → value.
    """
    N = labels.size(0)
    defer_mask = defer_probs >= threshold       # True = deferred to server
    local_mask = ~defer_mask                     # True = kept at client

    # System prediction: use server for deferred, client otherwise
    system_preds = torch.where(defer_mask, server_preds, client_preds)

    system_correct = (system_preds == labels).float()
    client_correct = (client_preds == labels).float()
    server_correct = (server_preds == labels).float()

    deferral_rate = defer_mask.float().mean().item()

    return {
        "system_accuracy": system_correct.mean().item(),
        "client_accuracy": client_correct.mean().item(),
        "server_accuracy": server_correct.mean().item(),
        "deferral_rate": deferral_rate,
        "local_accuracy": (
            client_correct[local_mask].mean().item() if local_mask.any() else 0.0
        ),
        "deferred_accuracy": (
            server_correct[defer_mask].mean().item() if defer_mask.any() else 0.0
        ),
    }


def aggregate_metrics(
    per_client: Dict[str, Dict[str, float]],
    weights: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """Weighted average of per-client metrics.

    Args:
        per_client: {client_id: {metric: value}}.
        weights: {client_id: weight}. Uniform if None.
    """
    clients = list(per_client.keys())
    if weights is None:
        weights = {c: 1.0 / len(clients) for c in clients}

    # Normalize
    total_w = sum(weights.values())
    weights = {c: w / total_w for c, w in weights.items()}

    metric_names = list(next(iter(per_client.values())).keys())
    agg = {}
    for m in metric_names:
        agg[f"avg_{m}"] = sum(weights[c] * per_client[c][m] for c in clients)
    return agg


def personalization_gain(
    personalized: Dict[str, Dict[str, float]],
    baseline: Dict[str, Dict[str, float]],
    metric: str = "system_accuracy",
) -> Dict[str, float]:
    """Per-client improvement from personalization.

    Args:
        personalized: Metrics with per-client LoRA adapters.
        baseline: Metrics with a single shared adapter.
        metric: Which metric to compare.

    Returns:
        {client_id: gain}.
    """
    return {
        cid: personalized[cid][metric] - baseline[cid][metric]
        for cid in personalized
    }
