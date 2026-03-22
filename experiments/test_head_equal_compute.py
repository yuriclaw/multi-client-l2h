"""Equal compute comparison: Per-client vs Universal with same total training effort.

Universal: all 5 clients' data pooled (5N samples), E epochs
Per-client: own data only (N samples), 5E epochs
Total gradient updates ≈ same (5N*E / bs ≈ N*5E / bs)

Test at multiple base epoch counts: E = 4, 8, 16, 32
With N = 8000 samples per client.
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm
import torchvision
import torchvision.transforms as T

BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))
N_SAMPLES = 8000  # per client
BASE_EPOCHS = [4, 8, 16, 32]  # Universal epochs; Per-client gets 5x

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def get_transform(grayscale=False):
    transforms = []
    if grayscale:
        transforms.append(T.Grayscale(num_output_channels=3))
    transforms.extend([
        T.Resize((32, 32)), T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return T.Compose(transforms)

def load_client(cid, data_dir, max_train=8000, max_test=2000):
    datasets = {
        0: ("CIFAR10", False, 10, "CIFAR-10"),
        1: ("SVHN", False, 10, "SVHN"),
        2: ("FashionMNIST", True, 10, "FashionMNIST"),
        3: ("STL10", False, 10, "STL-10"),
        4: ("GTSRB", False, 43, "GTSRB"),
    }
    ds_name, gray, nc, name = datasets[cid]
    tf = get_transform(grayscale=gray)
    
    if ds_name == "SVHN":
        train = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=tf)
    elif ds_name == "STL10":
        train = torchvision.datasets.STL10(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.STL10(data_dir, split='test', download=True, transform=tf)
    elif ds_name == "GTSRB":
        train = torchvision.datasets.GTSRB(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.GTSRB(data_dir, split='test', download=True, transform=tf)
    else:
        cls = getattr(torchvision.datasets, ds_name)
        train = cls(data_dir, train=True, download=True, transform=tf)
        test = cls(data_dir, train=False, download=True, transform=tf)
    
    if len(train) > max_train:
        train = Subset(train, random.sample(range(len(train)), max_train))
    if len(test) > max_test:
        test = Subset(test, random.sample(range(len(test)), max_test))
    return train, test, nc, name

def extract_features(backbone, upsample, loader, device):
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for xb, yb in loader:
            f = backbone(upsample(xb.to(device))).cpu()
            feats.append(f); labels.append(yb)
    return torch.cat(feats), torch.cat(labels)

def train_head(in_dim, nc, feats, labels, epochs, device):
    head = nn.Linear(in_dim, nc).to(device)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=BATCH_SIZE, shuffle=True)
    opt = Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    head.train()
    for ep in range(epochs):
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            loss = F.cross_entropy(head(fb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
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
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)
    for p in backbone.parameters(): p.requires_grad = False
    backbone = backbone.to(DEVICE).eval()
    feat_dim = 2048
    
    # Load & extract
    print("\n=== Loading data & extracting features ===")
    clients = {}
    for cid in range(5):
        train_ds, test_ds, nc, name = load_client(cid, DATA_DIR, max_train=N_SAMPLES)
        train_loader = DataLoader(train_ds, batch_size=128, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_ds, batch_size=128, num_workers=2)
        tf, tl = extract_features(backbone, upsample, train_loader, DEVICE)
        ef, el = extract_features(backbone, upsample, test_loader, DEVICE)
        clients[cid] = {"name": name, "nc": nc, "tf": tf, "tl": tl, "ef": ef, "el": el}
        print(f"  Client {cid} ({name}): train={len(tf)}, test={len(ef)}")
    
    # Build universal features (unified label space)
    total_classes = 0
    offsets = {}
    uni_feats, uni_labels = [], []
    for cid in range(5):
        c = clients[cid]
        offsets[cid] = total_classes
        uni_feats.append(c["tf"])
        uni_labels.append(c["tl"] + total_classes)
        total_classes += c["nc"]
    uni_feats = torch.cat(uni_feats)
    uni_labels = torch.cat(uni_labels)
    print(f"\n  Universal: {len(uni_feats)} samples, {total_classes} classes")
    
    # Run experiments
    all_results = {}
    
    for base_ep in BASE_EPOCHS:
        uni_epochs = base_ep
        personal_epochs = base_ep * 5
        
        uni_steps = len(uni_feats) * uni_epochs // BATCH_SIZE
        per_steps = N_SAMPLES * personal_epochs // BATCH_SIZE
        
        print(f"\n{'='*70}")
        print(f"  Universal: {len(uni_feats)} samples × {uni_epochs} epochs ≈ {uni_steps} steps")
        print(f"  Personal:  {N_SAMPLES} samples × {personal_epochs} epochs ≈ {per_steps} steps")
        print(f"{'='*70}")
        
        results = {}
        
        # Per-client
        for cid in range(5):
            c = clients[cid]
            head = train_head(feat_dim, c["nc"], c["tf"], c["tl"], personal_epochs, DEVICE)
            acc = eval_head(head, c["ef"], c["el"], DEVICE)
            results[cid] = {"personal": acc}
        
        # Universal
        uni_head = train_head(feat_dim, total_classes, uni_feats, uni_labels, uni_epochs, DEVICE)
        for cid in range(5):
            c = clients[cid]
            shifted = c["el"] + offsets[cid]
            acc = eval_head(uni_head, c["ef"], shifted, DEVICE)
            results[cid]["universal"] = acc
        
        all_results[base_ep] = results
        
        # Print
        for cid in range(5):
            c = clients[cid]
            p, u = results[cid]["personal"], results[cid]["universal"]
            print(f"    Client {cid} ({c['name']:>12s}): Personal={p:.4f}  Universal={u:.4f}  P-U={p-u:+.4f}")
    
    # Grand summary
    print("\n" + "="*90)
    print("GRAND SUMMARY: Equal Compute (Universal E epochs on 5N data vs Personal 5E epochs on N data)")
    print("="*90)
    
    for cid in range(5):
        c = clients[cid]
        print(f"\n  Client {cid} ({c['name']}):")
        print(f"    {'Uni Epochs':>12s} {'Per Epochs':>12s} {'Personal':>10s} {'Universal':>10s} {'P-U':>10s} {'Winner':>10s}")
        print(f"    {'-'*65}")
        for be in BASE_EPOCHS:
            r = all_results[be][cid]
            p, u = r["personal"], r["universal"]
            w = "Personal" if p > u + 0.005 else ("Universal" if u > p + 0.005 else "Tie")
            print(f"    {be:>12d} {be*5:>12d} {p:10.4f} {u:10.4f} {p-u:+10.4f} {w:>10s}")
    
    print(f"\n  AVERAGE:")
    print(f"    {'Uni Epochs':>12s} {'Per Epochs':>12s} {'Personal':>10s} {'Universal':>10s} {'P-U':>10s}")
    print(f"    {'-'*55}")
    for be in BASE_EPOCHS:
        ap = np.mean([all_results[be][cid]["personal"] for cid in range(5)])
        au = np.mean([all_results[be][cid]["universal"] for cid in range(5)])
        print(f"    {be:>12d} {be*5:>12d} {ap:10.4f} {au:10.4f} {ap-au:+10.4f}")
    
    print("\n✅ Equal compute test complete!")

if __name__ == "__main__":
    main()
