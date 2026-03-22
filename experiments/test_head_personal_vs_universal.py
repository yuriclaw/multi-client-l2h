"""Per-client head vs Universal head on frozen backbone.

No LoRA — pure head-level personalization comparison.
Server: ResNet-50 pretrained, backbone frozen, extract 2048-dim features.

Compare:
  1. Per-client head: each client trains its own linear head
  2. Universal head: all client data pooled, one shared head (unified label space)
  3. Universal head eval per-client: accuracy on each client's test set
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

def load_client_data(client_id, data_dir, max_train=8000, max_test=2000):
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
    
    if len(train) > max_train:
        train = Subset(train, random.sample(range(len(train)), max_train))
    if len(test) > max_test:
        test = Subset(test, random.sample(range(len(test)), max_test))
    
    print(f"  Client {client_id} ({name}): train={len(train)}, test={len(test)}, classes={nc}")
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

def train_head(in_dim, num_classes, train_feats, train_labels, epochs, device, tag=""):
    head = nn.Linear(in_dim, num_classes).to(device)
    ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    opt = Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    head.train()
    for ep in range(epochs):
        correct = total = 0
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            logits = head(fb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()
        if (ep+1) % 5 == 0:
            print(f"    [{tag}] Epoch {ep+1}/{epochs}: train_acc={correct/total:.4f}")
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
    
    # ── Build frozen backbone ──
    print("\n=== Building server (ResNet-50, frozen, no head) ===")
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(DEVICE).eval()
    feat_dim = 2048
    
    # ── Load data & extract features ──
    print("\n=== Loading client data ===")
    clients = {}
    for cid in range(5):
        train_ds, test_ds, nc, name = load_client_data(cid, DATA_DIR)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=128, num_workers=2)
        
        print(f"  Extracting features for {name}...")
        train_feats, train_labels = extract_features(backbone, upsample, train_loader, DEVICE)
        test_feats, test_labels = extract_features(backbone, upsample, test_loader, DEVICE)
        
        clients[cid] = {
            "name": name, "nc": nc,
            "train_feats": train_feats, "train_labels": train_labels,
            "test_feats": test_feats, "test_labels": test_labels,
        }
    
    # ── Phase 1: Per-client heads ──
    print("\n=== Phase 1: Per-client heads ===")
    personal_acc = {}
    for cid in range(5):
        c = clients[cid]
        print(f"\n  Client {cid} ({c['name']}):")
        head = train_head(feat_dim, c["nc"], c["train_feats"], c["train_labels"],
                         HEAD_EPOCHS, DEVICE, tag=c["name"])
        acc = eval_head(head, c["test_feats"], c["test_labels"], DEVICE)
        personal_acc[cid] = acc
        print(f"    Test acc: {acc:.4f}")
    
    # ── Phase 2: Universal head ──
    print("\n=== Phase 2: Universal head (all clients pooled) ===")
    
    # Build unified label space
    total_classes = 0
    client_offsets = {}
    all_train_feats, all_train_labels = [], []
    
    for cid in range(5):
        c = clients[cid]
        client_offsets[cid] = total_classes
        all_train_feats.append(c["train_feats"])
        all_train_labels.append(c["train_labels"] + total_classes)
        total_classes += c["nc"]
    
    all_train_feats = torch.cat(all_train_feats)
    all_train_labels = torch.cat(all_train_labels)
    print(f"  Total unified classes: {total_classes}, total samples: {len(all_train_feats)}")
    
    universal_head = train_head(feat_dim, total_classes, all_train_feats, all_train_labels,
                               HEAD_EPOCHS, DEVICE, tag="universal")
    
    # Eval universal on each client
    universal_acc = {}
    for cid in range(5):
        c = clients[cid]
        offset = client_offsets[cid]
        shifted_labels = c["test_labels"] + offset
        acc = eval_head(universal_head, c["test_feats"], shifted_labels, DEVICE)
        universal_acc[cid] = acc
        print(f"    Client {cid} ({c['name']}): {acc:.4f}")
    
    # ── Phase 3: "Seen some" universal — train on N-1 clients, test on held-out ──
    print("\n=== Phase 3: Leave-one-out universal (train on 4, test on held-out) ===")
    loo_acc = {}
    for held_out in range(5):
        # Train on all except held_out
        loo_feats, loo_labels = [], []
        loo_classes = 0
        loo_offsets = {}
        for cid in range(5):
            if cid == held_out:
                loo_offsets[cid] = loo_classes
                loo_classes += clients[cid]["nc"]
                continue
            c = clients[cid]
            loo_offsets[cid] = loo_classes
            loo_feats.append(c["train_feats"])
            loo_labels.append(c["train_labels"] + loo_classes)
            loo_classes += c["nc"]
        
        loo_feats = torch.cat(loo_feats)
        loo_labels = torch.cat(loo_labels)
        
        # Train head on the 4 clients
        loo_head = train_head(feat_dim, loo_classes, loo_feats, loo_labels,
                             HEAD_EPOCHS, DEVICE, tag=f"LOO-{held_out}")
        
        # Eval on held-out client (using its offset in the unified space)
        c = clients[held_out]
        shifted = c["test_labels"] + loo_offsets[held_out]
        acc = eval_head(loo_head, c["test_feats"], shifted, DEVICE)
        loo_acc[held_out] = acc
        print(f"    Held-out client {held_out} ({c['name']}): {acc:.4f}")
    
    # ── Summary ──
    print("\n" + "="*80)
    print("COMPARISON: Per-client Head vs Universal Head")
    print("="*80)
    print(f"{'Client':>8s} {'Dataset':>15s} {'Per-client':>12s} {'Universal':>12s} {'LOO':>12s} {'P-U':>10s} {'Winner':>10s}")
    print("-"*80)
    
    for cid in range(5):
        c = clients[cid]
        p = personal_acc[cid]
        u = universal_acc[cid]
        l = loo_acc[cid]
        pu = p - u
        winner = "Personal" if p > u + 0.005 else ("Universal" if u > p + 0.005 else "Tie")
        print(f"{cid:>8d} {c['name']:>15s} {p:12.4f} {u:12.4f} {l:12.4f} {pu:+10.4f} {winner:>10s}")
    
    avg_p = np.mean([personal_acc[i] for i in range(5)])
    avg_u = np.mean([universal_acc[i] for i in range(5)])
    avg_l = np.mean([loo_acc[i] for i in range(5)])
    print("-"*80)
    print(f"{'AVG':>8s} {'':>15s} {avg_p:12.4f} {avg_u:12.4f} {avg_l:12.4f} {avg_p-avg_u:+10.4f}")
    
    print("\n✅ Head personalization test complete!")

if __name__ == "__main__":
    main()
