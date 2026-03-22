"""Fair comparison: Per-client head vs Universal head with EQUAL data budget.

Each client gets N samples. Universal also gets N samples total (N/5 from each client).
This isolates personalization value from data quantity advantage.

Also test: varying data budgets (1k, 2k, 4k, 8k, 16k, 32k) to see scaling.
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm
import torchvision
import torchvision.transforms as T

BATCH_SIZE = 256
HEAD_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))

DATA_BUDGETS = [1000, 2000, 4000, 8000, 16000, 32000]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_transform(grayscale=False):
    transforms = []
    if grayscale:
        transforms.append(T.Grayscale(num_output_channels=3))
    transforms.extend([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return T.Compose(transforms)

def load_full_client_data(client_id, data_dir):
    """Load full datasets (no subsampling)."""
    if client_id == 0:
        tf = get_transform()
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        nc, name = 10, "CIFAR-10"
    elif client_id == 1:
        tf = get_transform()
        train = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=tf)
        nc, name = 10, "SVHN"
    elif client_id == 2:
        tf = get_transform(grayscale=True)
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        nc, name = 10, "FashionMNIST"
    elif client_id == 3:
        tf = get_transform()
        train = torchvision.datasets.STL10(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.STL10(data_dir, split='test', download=True, transform=tf)
        nc, name = 10, "STL-10"
    elif client_id == 4:
        tf = get_transform()
        train = torchvision.datasets.GTSRB(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.GTSRB(data_dir, split='test', download=True, transform=tf)
        nc, name = 43, "GTSRB"
    
    # Subsample test to 2000
    if len(test) > 2000:
        test = Subset(test, random.sample(range(len(test)), 2000))
    
    return train, test, nc, name

def extract_features(backbone, upsample, loader, device):
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            f = backbone(upsample(xb)).cpu()
            feats.append(f)
            labels.append(yb)
    return torch.cat(feats), torch.cat(labels)

def train_head(in_dim, num_classes, train_feats, train_labels, epochs, device):
    head = nn.Linear(in_dim, num_classes).to(device)
    ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    head.train()
    for ep in range(epochs):
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            loss = F.cross_entropy(head(fb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        scheduler.step()
    head.eval()
    return head

def eval_head(head, feats, labels, device):
    with torch.no_grad():
        logits = head(feats.to(device))
        return (logits.argmax(1) == labels.to(device)).float().mean().item()

def main():
    set_seed(42)
    print(f"Device: {DEVICE}")
    
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(DEVICE)
    
    # Build frozen backbone
    print("\n=== Building server (ResNet-50, frozen) ===")
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(DEVICE).eval()
    feat_dim = 2048
    
    # Load all data & extract features
    print("\n=== Loading & extracting features ===")
    clients = {}
    for cid in range(5):
        train_ds, test_ds, nc, name = load_full_client_data(cid, DATA_DIR)
        print(f"  Client {cid} ({name}): train={len(train_ds)}, test={len(test_ds)}, classes={nc}")
        
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=128, num_workers=2)
        
        train_feats, train_labels = extract_features(backbone, upsample, train_loader, DEVICE)
        test_feats, test_labels = extract_features(backbone, upsample, test_loader, DEVICE)
        
        clients[cid] = {
            "name": name, "nc": nc,
            "train_feats": train_feats, "train_labels": train_labels,
            "test_feats": test_feats, "test_labels": test_labels,
            "full_train_size": len(train_ds),
        }
    
    # For each data budget, compare per-client vs universal
    print("\n=== Running experiments across data budgets ===")
    
    all_results = {}  # budget -> {cid -> {"personal": acc, "universal": acc}}
    
    for budget in DATA_BUDGETS:
        print(f"\n{'='*60}")
        print(f"  DATA BUDGET: {budget} samples")
        print(f"{'='*60}")
        
        results = {}
        
        # Per-client: each client uses `budget` samples from its own data
        print(f"\n  --- Per-client heads (each uses {budget} own samples) ---")
        for cid in range(5):
            c = clients[cid]
            n = min(budget, len(c["train_feats"]))
            indices = random.sample(range(len(c["train_feats"])), n)
            feats = c["train_feats"][indices]
            labels = c["train_labels"][indices]
            
            head = train_head(feat_dim, c["nc"], feats, labels, HEAD_EPOCHS, DEVICE)
            acc = eval_head(head, c["test_feats"], c["test_labels"], DEVICE)
            results[cid] = {"personal": acc}
            print(f"    Client {cid} ({c['name']}): {acc:.4f} (used {n} samples)")
        
        # Universal: total budget = `budget` samples, split equally across clients
        per_client_share = budget // 5
        print(f"\n  --- Universal head ({budget} total = {per_client_share} per client) ---")
        
        total_classes = 0
        uni_feats, uni_labels = [], []
        client_offsets = {}
        
        for cid in range(5):
            c = clients[cid]
            client_offsets[cid] = total_classes
            n = min(per_client_share, len(c["train_feats"]))
            indices = random.sample(range(len(c["train_feats"])), n)
            uni_feats.append(c["train_feats"][indices])
            uni_labels.append(c["train_labels"][indices] + total_classes)
            total_classes += c["nc"]
        
        uni_feats = torch.cat(uni_feats)
        uni_labels = torch.cat(uni_labels)
        
        uni_head = train_head(feat_dim, total_classes, uni_feats, uni_labels, HEAD_EPOCHS, DEVICE)
        
        for cid in range(5):
            c = clients[cid]
            shifted = c["test_labels"] + client_offsets[cid]
            acc = eval_head(uni_head, c["test_feats"], shifted, DEVICE)
            results[cid]["universal"] = acc
            print(f"    Client {cid} ({c['name']}): {acc:.4f}")
        
        all_results[budget] = results
    
    # ── Grand Summary ──
    print("\n" + "="*100)
    print("GRAND SUMMARY: Per-client vs Universal at equal data budget")
    print("="*100)
    
    # Per client
    for cid in range(5):
        c = clients[cid]
        print(f"\n  Client {cid} ({c['name']}):")
        print(f"    {'Budget':>8s} {'Personal':>10s} {'Universal':>10s} {'P-U':>10s} {'Winner':>10s}")
        print(f"    {'-'*50}")
        for budget in DATA_BUDGETS:
            r = all_results[budget][cid]
            p, u = r["personal"], r["universal"]
            pu = p - u
            winner = "Personal" if p > u + 0.005 else ("Universal" if u > p + 0.005 else "Tie")
            print(f"    {budget:>8d} {p:10.4f} {u:10.4f} {pu:+10.4f} {winner:>10s}")
    
    # Average across clients
    print(f"\n  AVERAGE across all clients:")
    print(f"    {'Budget':>8s} {'Personal':>10s} {'Universal':>10s} {'P-U':>10s}")
    print(f"    {'-'*40}")
    for budget in DATA_BUDGETS:
        avg_p = np.mean([all_results[budget][cid]["personal"] for cid in range(5)])
        avg_u = np.mean([all_results[budget][cid]["universal"] for cid in range(5)])
        print(f"    {budget:>8d} {avg_p:10.4f} {avg_u:10.4f} {avg_p-avg_u:+10.4f}")
    
    print("\n✅ Fair comparison test complete!")

if __name__ == "__main__":
    main()
