"""Main training loop for multi-client L2H with LoRA personalization.

Alternating optimization:
  Stage 1: Fix rejector, update LoRA adapter (minimize L1)
  Stage 2: Fix adapter, update rejector (minimize L2)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    import wandb
except ImportError:
    wandb = None

from src.models.server import ServerWithLoRA
from src.models.client import build_client, freeze_model
from src.models.rejector import RejectorBank
from src.training.loss import AdapterLoss, RejectorLoss
from src.evaluation.metrics import compute_client_metrics


class L2HTrainer:
    """Orchestrates per-client alternating training."""

    def __init__(
        self,
        server: ServerWithLoRA,
        client_models: Dict[str, nn.Module],
        rejector_bank: RejectorBank,
        train_loaders: Dict[str, DataLoader],
        val_loaders: Dict[str, DataLoader],
        cfg: dict,
        device: str = "cuda",
    ) -> None:
        self.server = server.to(device)
        self.client_models = {k: v.to(device) for k, v in client_models.items()}
        self.rejector_bank = rejector_bank
        self.train_loaders = train_loaders
        self.val_loaders = val_loaders
        self.cfg = cfg
        self.device = device

        self.adapter_loss_fn = AdapterLoss()
        self.rejector_loss_fn = RejectorLoss(
            c_e=cfg["costs"]["c_e"], c_1=cfg["costs"]["c_1"]
        )

        # Per-client optimizers
        self.adapter_optimizers: Dict[str, Adam] = {}
        self.rejector_optimizers: Dict[str, Adam] = {}
        for cid in client_models:
            self.adapter_optimizers[cid] = Adam(
                self.server.adapter_parameters(cid),
                lr=cfg["training"]["adapter_lr"],
                weight_decay=cfg["training"]["weight_decay"],
            )
            self.rejector_optimizers[cid] = Adam(
                self.rejector_bank.parameters(cid),
                lr=cfg["training"]["rejector_lr"],
            )

    # ------------------------------------------------------------------
    # Training stages
    # ------------------------------------------------------------------

    def _train_adapter_one_epoch(self, client_id: str, loader: DataLoader) -> float:
        """Stage 1: Update LoRA adapter for one client."""
        self.server.train()
        self.rejector_bank[client_id].eval()
        opt = self.adapter_optimizers[client_id]
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                defer_probs = self.rejector_bank[client_id](x)
            server_logits = self.server(x, client_id)
            loss = self.adapter_loss_fn(server_logits, y, defer_probs)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    def _train_rejector_one_epoch(self, client_id: str, loader: DataLoader) -> float:
        """Stage 2: Update rejector for one client."""
        self.server.eval()
        self.rejector_bank[client_id].train()
        client_model = self.client_models[client_id]
        client_model.eval()
        opt = self.rejector_optimizers[client_id]
        total_loss = 0.0

        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            with torch.no_grad():
                server_logits = self.server(x, client_id)
                client_logits = client_model(x)
            defer_probs = self.rejector_bank[client_id](x)
            loss = self.rejector_loss_fn(defer_probs, client_logits, server_logits, y)

            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)

        return total_loss / len(loader.dataset)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def train(self) -> None:
        """Run the full training procedure."""
        num_rounds = self.cfg["training"]["num_rounds"]
        adapter_epochs = self.cfg["training"]["adapter_epochs"]
        rejector_epochs = self.cfg["training"]["rejector_epochs"]
        eval_every = self.cfg["evaluation"]["eval_every"]

        for rnd in range(1, num_rounds + 1):
            print(f"\n=== Round {rnd}/{num_rounds} ===")

            for cid in self.client_models:
                loader = self.train_loaders[cid]

                # Stage 1: adapter
                for ep in range(adapter_epochs):
                    a_loss = self._train_adapter_one_epoch(cid, loader)
                # Stage 2: rejector
                for ep in range(rejector_epochs):
                    r_loss = self._train_rejector_one_epoch(cid, loader)

                print(f"  {cid}: adapter_loss={a_loss:.4f}, rejector_loss={r_loss:.4f}")

                if wandb and wandb.run:
                    wandb.log({
                        f"{cid}/adapter_loss": a_loss,
                        f"{cid}/rejector_loss": r_loss,
                        "round": rnd,
                    })

            # Evaluate
            if rnd % eval_every == 0:
                self.evaluate(rnd)

    @torch.no_grad()
    def evaluate(self, rnd: int) -> Dict[str, dict]:
        """Evaluate all clients and log metrics."""
        self.server.eval()
        all_metrics = {}

        for cid in self.client_models:
            loader = self.val_loaders[cid]
            client_model = self.client_models[cid]
            client_model.eval()
            rejector = self.rejector_bank[cid]
            rejector.eval()

            all_labels, all_client_preds, all_server_preds, all_defer = [], [], [], []

            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                client_logits = client_model(x)
                server_logits = self.server(x, cid)
                defer_probs = rejector(x)

                all_labels.append(y)
                all_client_preds.append(client_logits.argmax(1))
                all_server_preds.append(server_logits.argmax(1))
                all_defer.append(defer_probs)

            metrics = compute_client_metrics(
                labels=torch.cat(all_labels),
                client_preds=torch.cat(all_client_preds),
                server_preds=torch.cat(all_server_preds),
                defer_probs=torch.cat(all_defer),
            )
            all_metrics[cid] = metrics
            print(f"  [{cid}] acc={metrics['system_accuracy']:.4f}, "
                  f"defer={metrics['deferral_rate']:.4f}")

            if wandb and wandb.run:
                wandb.log({f"{cid}/{k}": v for k, v in metrics.items()} | {"round": rnd})

        return all_metrics
