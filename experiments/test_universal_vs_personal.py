"""Ablation: Universal LoRA (all clients pooled) vs Per-client LoRA.

Question: Does per-client personalization actually help, or is a single
universal LoRA trained on all corruption data equally good?
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm
from peft import LoraConfig, get_peft_model

NUM_CLASSES = 10
BATCH_SIZE = 128
LORA_RANK = 16
LORA_LR = 1e-4
LORA_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))
SEVERITY = 3

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "fog", "motion_blur",
    "brightness", "contrast", "frost", "snow",
]

LORA_TARGETS = ["layer3.0.conv2", "layer3.1.conv2", "layer3.2.conv2",
                "layer3.3.conv2", "layer3.4.conv2", "layer3.5.conv2",
                "layer4.0.conv2", "layer4.1.conv2", "layer4.2.conv2"]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_cifar10c(data_dir, corruption, severity):
    base = os.path.join(data_dir, "CIFAR-10-C")
    imgs_all = np.load(os.path.join(base, f"{corruption}.npy"))
    labels_all = np.load(os.path.join(base, "labels.npy"))
    start = (severity - 1) * 10000
    imgs = imgs_all[start:start+10000]
    labels = labels_all[start:start+10000]
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    imgs = (imgs / 255.0 - mean) / std
    imgs = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

def batched_eval(model, upsample, x, bs=128):
    preds = []
    for i in range(0, len(x), bs):
        xb = x[i:i+bs].to(DEVICE)
        with torch.no_grad():
            preds.append(model(upsample(xb)).argmax(1).cpu())
    return torch.cat(preds)

def train_lora(server_state, upsample, train_loader, tag=""):
    """Build fresh LoRA model, train, return it."""
    server = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
    for p in server.parameters():
        p.requires_grad = False
    server.load_state_dict(server_state)
    
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGETS, lora_dropout=0.05, bias="none",
    )
    model = get_peft_model(server, lora_config).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LORA_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LORA_EPOCHS)
    
    model.train()
    for ep in range(LORA_EPOCHS):
        correct = total = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = model(upsample(xb))
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()
        if (ep+1) % 5 == 0:
            print(f"    [{tag}] Epoch {ep+1}/{LORA_EPOCHS}: train_acc={correct/total:.4f}")
    
    model.eval()
    return model

def main():
    set_seed(42)
    print(f"Device: {DEVICE}")
    
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(DEVICE)
    
    # ── Phase 1: Build & probe server ──
    print("\n=== Phase 1: Linear probe on clean CIFAR-10 ===")
    server = timm.create_model("resnet50", pretrained=True, num_classes=NUM_CLASSES)
    for p in server.parameters():
        p.requires_grad = False
    server = server.to(DEVICE)
    
    import torchvision, torchvision.transforms as T
    probe_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    probe_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=probe_transform)
    probe_loader = DataLoader(probe_ds, batch_size=256, shuffle=True, num_workers=2)
    
    for name, p in server.named_parameters():
        if 'fc' in name: p.requires_grad = True
    probe_opt = Adam([p for p in server.parameters() if p.requires_grad], lr=1e-3)
    server.train()
    for ep in range(8):
        correct = total = 0
        for xb, yb in probe_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = server(upsample(xb))
            loss = F.cross_entropy(logits, yb)
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if (ep+1) % 2 == 0:
            print(f"  Probe epoch {ep+1}/8: acc={correct/total:.4f}")
    for p in server.parameters():
        p.requires_grad = False
    server.eval()
    probed_state = {k: v.cpu().clone() for k, v in server.state_dict().items()}
    
    # ── Phase 2: Baselines ──
    print("\n=== Phase 2: Server baseline ===")
    baselines = {}
    for corr in CORRUPTIONS:
        imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
        preds = batched_eval(server, upsample, imgs[-2000:])
        baselines[corr] = (preds == labels[-2000:]).float().mean().item()
        print(f"  {corr:20s}: {baselines[corr]:.4f}")
    
    # ── Phase 3: Load all corruption data ──
    print("\n=== Phase 3: Loading corruption data ===")
    train_data = {}  # per-corruption train sets
    test_data = {}   # per-corruption test sets
    all_train_datasets = []
    
    for corr in CORRUPTIONS:
        imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
        train_data[corr] = TensorDataset(imgs[:8000], labels[:8000])
        test_data[corr] = (imgs[-2000:], labels[-2000:])
        all_train_datasets.append(train_data[corr])
    
    universal_ds = ConcatDataset(all_train_datasets)
    print(f"  Per-client: 8000 samples each")
    print(f"  Universal:  {len(universal_ds)} samples total")
    
    # ── Phase 4: Train Universal LoRA ──
    print("\n=== Phase 4: Universal LoRA (all corruptions pooled) ===")
    universal_loader = DataLoader(universal_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    universal_model = train_lora(probed_state, upsample, universal_loader, tag="universal")
    
    # Eval universal on each corruption
    universal_acc = {}
    for corr in CORRUPTIONS:
        x_test, y_test = test_data[corr]
        preds = batched_eval(universal_model, upsample, x_test)
        universal_acc[corr] = (preds == y_test).float().mean().item()
    
    # Cleanup
    del universal_model
    torch.cuda.empty_cache()
    
    # ── Phase 5: Train Per-client LoRA ──
    print("\n=== Phase 5: Per-client LoRA (one per corruption) ===")
    personal_acc = {}  # personal_acc[corr] = acc on own corruption
    personal_cross = {}  # personal_cross[trained_on][tested_on] = acc
    
    for corr in CORRUPTIONS:
        print(f"\n  Training LoRA for {corr}...")
        loader = DataLoader(train_data[corr], batch_size=BATCH_SIZE, shuffle=True)
        model = train_lora(probed_state, upsample, loader, tag=corr)
        
        personal_cross[corr] = {}
        for test_corr in CORRUPTIONS:
            x_test, y_test = test_data[test_corr]
            preds = batched_eval(model, upsample, x_test)
            acc = (preds == y_test).float().mean().item()
            personal_cross[corr][test_corr] = acc
            if test_corr == corr:
                personal_acc[corr] = acc
        
        del model
        torch.cuda.empty_cache()
    
    # ── Summary ──
    print("\n" + "="*80)
    print("COMPARISON: Baseline vs Universal LoRA vs Per-client LoRA")
    print("="*80)
    print(f"{'Corruption':20s} {'Baseline':>10s} {'Universal':>10s} {'Personal':>10s} {'U-delta':>10s} {'P-delta':>10s} {'P-U':>10s} {'Winner':>10s}")
    print("-"*90)
    
    u_total = p_total = 0
    for corr in CORRUPTIONS:
        b = baselines[corr]
        u = universal_acc[corr]
        p = personal_acc[corr]
        ud = u - b
        pd = p - b
        pu = p - u
        winner = "Personal" if p > u + 0.005 else ("Universal" if u > p + 0.005 else "Tie")
        print(f"{corr:20s} {b:10.4f} {u:10.4f} {p:10.4f} {ud:+10.4f} {pd:+10.4f} {pu:+10.4f} {winner:>10s}")
        u_total += ud
        p_total += pd
    
    print("-"*90)
    print(f"{'AVERAGE':20s} {'':10s} {'':10s} {'':10s} {u_total/len(CORRUPTIONS):+10.4f} {p_total/len(CORRUPTIONS):+10.4f}")
    
    print("\n✅ Universal vs Personal LoRA test complete!")

if __name__ == "__main__":
    main()
