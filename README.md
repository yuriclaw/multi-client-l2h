# Multi-Client Learning to Help with LoRA Personalization

Experiment framework for multi-client Learning to Help (L2H) with per-client LoRA adapter personalization on the server model.

## Overview

In the L2H framework, lightweight **client models** handle easy inputs locally while deferring hard inputs to a powerful **server model**. A **rejector network** per client learns the deferral policy. This codebase extends L2H with **LoRA adapters** on the server, giving each client a personalized server without full fine-tuning.

## Structure

```
configs/          — YAML experiment configs
src/models/       — Client, server (with LoRA), and rejector models
src/data/         — Data loaders (CIFAR-10-C, CIFAR-100-C)
src/training/     — Loss functions and training loop
src/evaluation/   — Metrics (accuracy, deferral rate, system accuracy)
experiments/      — Runnable experiment scripts
```

## Quick Start

```bash
pip install -r requirements.txt
python experiments/run_cifar10c.py --config configs/default.yaml
```

## Key Ideas

- **Per-client LoRA adapters**: Each client gets a lightweight adapter on the frozen server backbone (ViT/ResNet), enabling personalization at minimal cost.
- **Alternating optimization**: Stage 1 updates the LoRA adapter (cross-entropy), Stage 2 updates the rejector (cost-sensitive BCE).
- **Corruption-based heterogeneity**: Clients receive different CIFAR-C corruption types to simulate distribution shift.
