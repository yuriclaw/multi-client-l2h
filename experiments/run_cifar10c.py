"""Main experiment: Multi-Client L2H with LoRA on CIFAR-10-C.

Usage:
    python experiments/run_cifar10c.py --config configs/default.yaml
    python experiments/run_cifar10c.py --config configs/default.yaml --num_clients 3
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, random_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.client import build_client, freeze_model
from src.models.server import ServerWithLoRA
from src.models.rejector import RejectorBank
from src.data.cifar_c import build_client_datasets
from src.training.trainer import L2HTrainer

try:
    import wandb
except ImportError:
    wandb = None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="L2H + LoRA on CIFAR-10-C")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--num_clients", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_clients:
        cfg["clients"]["num_clients"] = args.num_clients
    device = args.device or cfg.get("device", "cuda")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    set_seed(cfg["seed"])

    # ------------------------------------------------------------------
    # W&B
    # ------------------------------------------------------------------
    if wandb and not args.no_wandb:
        wandb.init(
            project=cfg["logging"]["wandb_project"],
            entity=cfg["logging"].get("wandb_entity"),
            config=cfg,
        )

    # ------------------------------------------------------------------
    # Data: per-client corrupted datasets
    # ------------------------------------------------------------------
    num_clients = cfg["clients"]["num_clients"]
    corruptions = cfg["dataset"]["corruptions"][:num_clients]
    cifar_version = 10 if "10" in cfg["dataset"]["name"] else 100
    num_classes = cfg["dataset"]["num_classes"]

    print(f"Setting up {num_clients} clients with corruptions: {corruptions}")

    datasets = build_client_datasets(
        data_dir=cfg["dataset"]["data_dir"],
        corruptions=corruptions,
        severity=cfg["dataset"]["severity"],
        cifar_version=cifar_version,
    )

    # Split each into train/val (80/20)
    train_loaders, val_loaders = {}, {}
    for cid, ds in datasets.items():
        n_train = int(0.8 * len(ds))
        n_val = len(ds) - n_train
        train_ds, val_ds = random_split(ds, [n_train, n_val])
        train_loaders[cid] = DataLoader(
            train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True, num_workers=2
        )
        val_loaders[cid] = DataLoader(
            val_ds, batch_size=cfg["training"]["batch_size"], shuffle=False, num_workers=2
        )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    # Client models (frozen)
    client_models = {}
    for cid in datasets:
        model = build_client(cfg["clients"]["model"], num_classes)
        if cfg["clients"]["freeze"]:
            freeze_model(model)
        client_models[cid] = model

    # Server with per-client LoRA
    server_cfg = cfg["server"]
    server = ServerWithLoRA(
        backbone_name=server_cfg["backbone"],
        num_classes=num_classes,
        lora_rank=server_cfg["lora"]["rank"],
        lora_alpha=server_cfg["lora"]["alpha"],
        lora_dropout=server_cfg["lora"]["dropout"],
        target_modules=server_cfg["lora"]["target_modules"],
    )
    for cid in datasets:
        server.add_adapter(cid)

    # Rejector bank
    rejector_bank = RejectorBank(
        input_dim=cfg["rejector"]["input_dim"],
        hidden_dim=cfg["rejector"]["hidden_dim"],
        device=device,
    )
    for cid in datasets:
        rejector_bank.add_client(cid)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    trainer = L2HTrainer(
        server=server,
        client_models=client_models,
        rejector_bank=rejector_bank,
        train_loaders=train_loaders,
        val_loaders=val_loaders,
        cfg=cfg,
        device=device,
    )

    print("\nStarting training...")
    trainer.train()

    # Final evaluation
    print("\n=== Final Evaluation ===")
    final_metrics = trainer.evaluate(rnd=cfg["training"]["num_rounds"])
    
    from src.evaluation.metrics import aggregate_metrics
    agg = aggregate_metrics(final_metrics)
    print(f"\nAggregated: {agg}")

    if wandb and wandb.run:
        wandb.log(agg)
        wandb.finish()

    print("\nDone.")


if __name__ == "__main__":
    main()
