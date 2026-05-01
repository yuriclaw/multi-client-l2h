# Multi-Client L2H: Collaborative Inference with Heterogeneous Clients

## Overview

This project studies **collaborative inference** between heterogeneous edge clients and a powerful cloud server. Each client runs a lightweight model locally and uses a **deferral policy** to decide which inputs to send to the server for better predictions.

## Current Setup (v7.6 — Baseline)

### Architecture

```
Client (edge):   MobileNetV2 (ImageNet pretrained, full finetune, 3.4M params)
                 Input: 224×224 → Hidden: 1280d → Classifier: Linear(1280, K)

Server (cloud):  DINOv2-B/14 (frozen, 86M params) + 2-layer MLP head
                 Input: 224×224 → Features: 768d → MLP(768→512→K)

Rejector (L2H):  AlexNet-style CNN on 32×32 downsampled image + client hidden
                 Input: 32×32 image + 1280d hidden → 2-class (keep/defer)
```

### Datasets (5 heterogeneous clients)

| Client | Dataset | Classes | Train | Val | Test |
|--------|---------|---------|-------|-----|------|
| 0 | CIFAR-100 | 100 | 5000 | 500 | 500 |
| 1 | Flowers102 | 102 | 1020 | 500 | 500 |
| 2 | OxfordPet | 37 | 3680 | 500 | 500 |
| 3 | Food101 | 101 | 5000 | 500 | 500 |
| 4 | DTD | 47 | 1880 | 500 | 500 |

Val and Test are split from the original test set. SEED=42 for reproducibility.

### Training Pipeline (Two-Stage)

**Stage 1: Independent Training**
1. Client: MobileNetV2 full finetune (30 epochs, backbone LR=1e-4, head LR=1e-3)
2. Server: Extract DINOv2 features → Train MLP head (200 epochs, Adam LR=1e-3)
   - Per-client head: one head per client dataset
   - Universal head: one head trained on all clients' data (387 unified classes)

**Stage 2: Joint Deferral Training (100 epochs)**
- L2H: Train AlexNet rejector + server head jointly (L2H loss + server CE)
- Gatekeeper: Finetune client classifier + server head (GK loss + server CE)
- ConfThresh: No training needed (use client softmax confidence directly)

### Evaluation

- **Threshold-based deferral**: Find score threshold on validation set → apply on test set
- Target deferral rates: 10%, 30%, 50%, 70%, 90%
- Actual test deferral rate may differ from target (realistic evaluation)

### Deferral Methods

| Method | Modifies Client? | Extra Model? | Training? | Deferral Signal |
|--------|-----------------|-------------|-----------|-----------------|
| L2H (Ours) | No | AlexNet rejector | Yes (Stage 2) | Rejector output |
| Gatekeeper | Yes (classifier) | No | Yes (Stage 2) | Client softmax conf |
| ConfThresh | No | No | No | Client softmax conf |
| Random | No | No | No | Random selection |

### Key Results (v7.6)

**Client & Server Accuracy:**

| Dataset | Client | Server(PC) | Server(Uni) |
|---------|--------|-----------|------------|
| CIFAR-100 | 62.6% | 84.2% | 84.0% |
| Flowers102 | 89.8% | 99.8% | 99.8% |
| OxfordPet | 90.8% | 96.0% | 96.0% |
| Food101 | 60.2% | 86.4% | 85.2% |
| DTD | 70.6% | 82.0% | 80.0% |

**System Accuracy (avg 5 clients, threshold-based):**

| Rate | L2H | GK | ConfTh | Random | Server-only |
|------|-----|-----|--------|--------|-------------|
| 10% | .761 | .784 | **.794** | .763 | .897 |
| 30% | .785 | .834 | **.846** | .792 | .897 |
| 50% | .814 | .872 | **.872** | .822 | .897 |
| 70% | .850 | **.891** | .890 | .852 | .897 |
| 90% | .882 | .894 | **.897** | .881 | .897 |

**Finding: ConfThresh (no training) ≈ GK >> L2H > Random**

### Conclusions from v7 series

1. DINOv2 is the best server backbone (>> ResNet-50, CLIP-ViT)
2. Per-client ≈ Universal server head (gap < 1.5pp)
3. 2-layer MLP head slightly better than linear head (~0.5pp)
4. ConfThresh is nearly optimal — no-training baseline matches trained methods
5. MobileNetV2 full finetune gives reasonable client accuracy (60-91%)

## File Structure

```
experiments/
  test_l2h_v7_backbones.py          — v7: AlexNet client, 3 backbones × 5 datasets
  test_l2h_v7.1_resnet18client.py   — v7.1: ResNet-18 CIFAR-style client
  test_l2h_v7.2_mobilenet_client.py — v7.2: MobileNetV2 linear probe client
  test_l2h_v7.3_mobilenet_finetune.py — v7.3: MobileNetV2 full finetune client
  test_l2h_v7.4_dinov2_mlphead.py   — v7.4: DINOv2 only + MLP head
  test_l2h_v7.4b_universal.py       — v7.4b: Per-client vs Universal head comparison
  test_l2h_v7.5_threshold.py        — v7.5: Threshold-based deferral (val→test)
  test_l2h_v7.6_alexnet_rejector.py — v7.6: AlexNet rejector restored (CURRENT BASELINE)
```

## Environment

- Server: NVIDIA RTX 4000 Ada (20GB VRAM), 125GB RAM
- Python 3.12, PyTorch, torchvision, timm
- Conda env: `l2h-b`

## Next Directions

### Direction 1: Privacy-aware Universal vs Per-Client

Three privacy scenarios:
- 1.1: Clients share only partial data (10%/30%/50%) for universal head
- 1.2: Clients share noisy data (differential privacy) for universal head
- 1.3: Clients share NO data; universal head uses public dataset only

### Direction 2: Same Dataset, Different Client Architectures

- All clients use CIFAR-100 but with different models (AlexNet, ResNet-18, MobileNetV2, etc.)
- Compare universal rejector + universal head vs per-client rejector + per-client head
