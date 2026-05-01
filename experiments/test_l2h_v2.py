"""
Two-Stage L2H System v2 — Gatekeeper-inspired
==============================================
- Client = AlexNet-style CNN (trained per-client, then frozen for L2H)
- Server = Pretrained ResNet-50 (frozen) + per-client/universal linear head
- Rejector = Gatekeeper-style: predict "client incorrect" → defer

3 methods at fixed deferral rates:
1. Ours: Per-client server head + Per-client rejector
2. Universal: Universal server head + Universal rejector
3. ConfThresh: Per-client server head + client max-softmax threshold
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import timm
import random
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_TRAIN = 8000
N_TEST = 2000
DATA_DIR = '/tmp/data'
HEAD_EPOCHS = 500
REJ_EPOCHS = 100
BATCH_SIZE = 128

# ============================================================
# Data — two transforms: one for 32x32 client, one for 224x224 backbone
# ============================================================
INET_MEAN = (0.485, 0.456, 0.406)
INET_STD = (0.229, 0.224, 0.225)

def get_transform_32(grayscale=False):
    """For AlexNet client: 32x32, CIFAR norm."""
    t = []
    if grayscale:
        t.append(transforms.Grayscale(3))
    t.extend([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return transforms.Compose(t)

def get_transform_224(grayscale=False):
    """For pretrained ResNet-50 server: 224x224, ImageNet norm."""
    t = []
    if grayscale:
        t.append(transforms.Grayscale(3))
    t.extend([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(INET_MEAN, INET_STD),
    ])
    return transforms.Compose(t)

CLIENT_CONFIGS = {
    0: {'ds': 'CIFAR10', 'name': 'CIFAR-10', 'gray': False, 'n_classes': 10},
    1: {'ds': 'SVHN', 'name': 'SVHN', 'gray': False, 'n_classes': 10},
    2: {'ds': 'FashionMNIST', 'name': 'FashionMNIST', 'gray': True, 'n_classes': 10},
    3: {'ds': 'STL10', 'name': 'STL-10', 'gray': False, 'n_classes': 10},
    4: {'ds': 'GTSRB', 'name': 'GTSRB', 'gray': False, 'n_classes': 43},
}

def load_dataset(name, split='train', transform=None):
    if name == 'CIFAR10':
        return torchvision.datasets.CIFAR10(DATA_DIR, train=(split=='train'), download=True, transform=transform)
    elif name == 'SVHN':
        return torchvision.datasets.SVHN(DATA_DIR, split=split, download=True, transform=transform)
    elif name == 'FashionMNIST':
        return torchvision.datasets.FashionMNIST(DATA_DIR, train=(split=='train'), download=True, transform=transform)
    elif name == 'STL10':
        return torchvision.datasets.STL10(DATA_DIR, split=split, download=True, transform=transform)
    elif name == 'GTSRB':
        return torchvision.datasets.GTSRB(DATA_DIR, split=split, download=True, transform=transform)

def get_client_data():
    """Load data with BOTH 32x32 (client) and 224x224 (server) transforms, same indices."""
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tf32 = get_transform_32(cfg['gray'])
        tf224 = get_transform_224(cfg['gray'])
        
        train_ds_32 = load_dataset(cfg['ds'], 'train', tf32)
        test_ds_32 = load_dataset(cfg['ds'], 'test', tf32)
        train_ds_224 = load_dataset(cfg['ds'], 'train', tf224)
        test_ds_224 = load_dataset(cfg['ds'], 'test', tf224)
        
        # Same random indices for both resolutions
        train_idx = random.sample(range(len(train_ds_32)), min(N_TRAIN, len(train_ds_32)))
        test_idx = random.sample(range(len(test_ds_32)), min(N_TEST, len(test_ds_32)))
        
        clients[cid] = {
            'name': cfg['name'], 'n_classes': cfg['n_classes'],
            'train_32': Subset(train_ds_32, train_idx),
            'test_32': Subset(test_ds_32, test_idx),
            'train_224': Subset(train_ds_224, train_idx),
            'test_224': Subset(test_ds_224, test_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {len(train_idx)} train, {len(test_idx)} test")
    return clients

# ============================================================
# Client Model: AlexNet-style CNN (trained per-client, then frozen)
# ============================================================
CLIENT_EPOCHS = 50
CLIENT_FEAT_DIM = 256

class ClientAlexNet(nn.Module):
    """Lightweight AlexNet-style CNN for 32x32 input.
    Weaker than ResNet-50 server but non-trivial."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),   # 16x16
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),            # 8x8
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),                             # 8x8
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),                             # 8x8
            nn.Conv2d(256, CLIENT_FEAT_DIM, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),# 4x4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),                                     # 256
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(CLIENT_FEAT_DIM, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, n_classes),
        )
    
    def forward(self, x):
        return self.classifier(self.features(x))
    
    def get_features(self, x):
        return self.features(x)

def train_client_model(n_classes, train_ds, test_ds, epochs=CLIENT_EPOCHS):
    """Train AlexNet client from scratch, return frozen model."""
    model = ClientAlexNet(n_classes).to(device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
    # Freeze
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    # Eval
    correct, total = 0, 0
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)
    with torch.no_grad():
        for x, y in test_loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return model, correct / total

# ============================================================
# Server Backbone: pretrained ResNet-50 (frozen)
# ============================================================
def make_server_backbone():
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone

def extract_server_features(backbone, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

# ============================================================
# Linear Head (for both client probe and server head)
# ============================================================
def train_head(feats, labels, n_classes, epochs=HEAD_EPOCHS):
    head = nn.Linear(feats.shape[1], n_classes).to(device)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=256, shuffle=True)
    opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            opt.zero_grad(); crit(head(f), y).backward(); opt.step()
        sched.step()
    head.eval()
    return head

def eval_head(head, feats, labels):
    head.eval()
    with torch.no_grad():
        return (head(feats.to(device)).argmax(1).cpu() == labels).float().mean().item()

# ============================================================
# Rejector (Gatekeeper-inspired)
# ============================================================
class Rejector(nn.Module):
    """LeNet-style rejector that takes raw 32x32 images as input.
    Learns to predict whether the client model will be incorrect on this sample."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),       # 28x28
            nn.ReLU(),
            nn.MaxPool2d(2),           # 14x14
            nn.Conv2d(6, 16, 5),       # 10x10
            nn.ReLU(),
            nn.MaxPool2d(2),           # 5x5
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, 1),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def build_rejector_labels_gatekeeper(client_preds, server_preds, labels):
    """Gatekeeper-style: label = 1 when client is INCORRECT (should defer).
    This gives a much better training signal than "server correct & client wrong".
    """
    return (client_preds != labels).long()

def train_rejector_on_images(dataset_32, rej_labels, epochs=REJ_EPOCHS):
    """Train rejector on raw 32x32 images with binary labels."""
    rej = Rejector().to(device)
    
    # Build dataset: (image, rejector_label)
    images = []
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    for x, _ in loader:
        images.append(x)
    images = torch.cat(images)
    
    rej_dataset = TensorDataset(images, rej_labels.float())
    rej_loader = DataLoader(rej_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    n_pos = rej_labels.sum().item(); n_neg = len(rej_labels) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    rej.train()
    for _ in range(epochs):
        for imgs, y in rej_loader:
            imgs, y = imgs.to(device), y.to(device)
            opt.zero_grad(); crit(rej(imgs).squeeze(), y).backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def get_rejector_scores(rejector, dataset_32):
    """Get rejector scores on raw images. Higher = more likely to defer."""
    rejector.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            scores.append(rejector(x.to(device)).squeeze().cpu())
    return torch.cat(scores)

# ============================================================
# System evaluation
# ============================================================
def system_acc_at_rate(scores, target_rate, client_preds, server_preds, labels):
    """Higher score = more likely to defer. Uses topk for exact count."""
    n = len(scores)
    k = max(1, int(n * target_rate))
    _, top_idx = torch.topk(scores, k)
    defer = torch.zeros(n, dtype=torch.bool)
    defer[top_idx] = True
    system_preds = torch.where(defer, server_preds, client_preds)
    acc = (system_preds == labels).float().mean().item()
    return acc, k / n

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("Two-Stage L2H System v2 (Gatekeeper-inspired)")
    print("Client: AlexNet-style CNN (trained per-client, then frozen)")
    print("Server: Pretrained ResNet-50 (frozen + per-client/universal head)")
    print("=" * 80)
    
    # --- Load data (224x224 for both models) ---
    print("\n[1/8] Loading datasets...")
    clients = get_client_data()
    
    # --- Train client models (AlexNet per client, then freeze) ---
    print(f"\n[2/8] Training client AlexNet models ({CLIENT_EPOCHS} epochs each)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client_model(info['n_classes'], info['train_32'], info['test_32'])
        client_models[cid] = model
        print(f"  Client {cid} ({info['name']}): acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    # --- Server backbone ---
    print("\n[3/8] Loading pretrained ResNet-50 (server backbone, frozen)...")
    server_backbone = make_server_backbone()
    
    # --- Extract server features (224x224) ---
    print("\n[4/8] Extracting server features...")
    server_tr_feats, server_tr_labels = {}, {}
    server_te_feats, server_te_labels = {}, {}
    for cid, info in clients.items():
        server_tr_feats[cid], server_tr_labels[cid] = extract_server_features(server_backbone, info['train_224'])
        server_te_feats[cid], server_te_labels[cid] = extract_server_features(server_backbone, info['test_224'])
        print(f"  Client {cid} ({info['name']}): train {server_tr_feats[cid].shape}")
    
    # --- No separate client feature extraction needed — we use client_models directly ---
    # Client logits will be computed end-to-end on 32x32 data
    print("\n[5/8] (Client models already trained and frozen)")
    
    # --- Train server heads ---
    print(f"\n[6/8] Training server heads ({HEAD_EPOCHS} epochs)...")
    
    # Per-client server heads
    print("  Per-client heads:")
    per_server_heads = {}
    for cid, info in clients.items():
        per_server_heads[cid] = train_head(server_tr_feats[cid], server_tr_labels[cid], info['n_classes'])
        acc = eval_head(per_server_heads[cid], server_te_feats[cid], server_te_labels[cid])
        print(f"    {info['name']}: acc={acc:.4f}")
    
    # Universal server head
    print("  Universal head:")
    offsets = {}; off = 0
    for cid, info in clients.items():
        offsets[cid] = off; off += info['n_classes']
    total_classes = off
    uni_feats = torch.cat([server_tr_feats[cid] for cid in clients])
    uni_labels = torch.cat([server_tr_labels[cid] + offsets[cid] for cid in clients])
    uni_server_head = train_head(uni_feats, uni_labels, total_classes)
    for cid, info in clients.items():
        acc = eval_head(uni_server_head, server_te_feats[cid], server_te_labels[cid] + offsets[cid])
        print(f"    {info['name']}: acc={acc:.4f}")
    
    # --- Get logits for rejector features ---
    print("\n[7/8] Computing logits for rejector training...")
    
    def get_client_logits(client_model, dataset_32):
        """Get client logits from AlexNet on 32x32 data."""
        client_model.eval()
        loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        logits_list, labels_list = [], []
        with torch.no_grad():
            for x, y in loader:
                logits_list.append(client_model(x.to(device)).cpu())
                labels_list.append(y)
        return torch.cat(logits_list), torch.cat(labels_list)
    
    def get_server_logits(server_head, s_feats):
        server_head.eval()
        with torch.no_grad():
            return server_head(s_feats.to(device)).cpu()
    
    # Per-client server: get logits
    per_client_logits_tr, per_server_logits_tr = {}, {}
    per_client_logits_te, per_server_logits_te = {}, {}
    per_client_preds_tr, per_server_preds_tr = {}, {}
    per_client_preds_te, per_server_preds_te = {}, {}
    client_tr_labels, client_te_labels = {}, {}
    
    for cid, info in clients.items():
        cl, client_tr_labels[cid] = get_client_logits(client_models[cid], info['train_32'])
        per_client_logits_tr[cid] = cl
        per_server_logits_tr[cid] = get_server_logits(per_server_heads[cid], server_tr_feats[cid])
        per_client_preds_tr[cid] = cl.argmax(1)
        per_server_preds_tr[cid] = per_server_logits_tr[cid].argmax(1)
        
        cl, client_te_labels[cid] = get_client_logits(client_models[cid], info['test_32'])
        per_client_logits_te[cid] = cl
        per_server_logits_te[cid] = get_server_logits(per_server_heads[cid], server_te_feats[cid])
        per_client_preds_te[cid] = cl.argmax(1)
        per_server_preds_te[cid] = per_server_logits_te[cid].argmax(1)
    
    # Universal server: get logits
    uni_server_logits_tr, uni_server_logits_te = {}, {}
    uni_server_preds_tr, uni_server_preds_te = {}, {}
    
    for cid, info in clients.items():
        nc = info['n_classes']
        off = offsets[cid]
        
        sl = get_server_logits(uni_server_head, server_tr_feats[cid])
        uni_server_logits_tr[cid] = sl
        uni_server_preds_tr[cid] = sl[:, off:off+nc].argmax(1)
        
        sl = get_server_logits(uni_server_head, server_te_feats[cid])
        uni_server_logits_te[cid] = sl
        uni_server_preds_te[cid] = sl[:, off:off+nc].argmax(1)
    
    # --- Train rejectors on raw images ---
    print(f"\n[8/8] Training rejectors on raw images ({REJ_EPOCHS} epochs)...")
    
    # Per-client rejectors
    print("  Per-client rejectors:")
    per_rejectors = {}
    for cid, info in clients.items():
        t0 = time.time()
        rej_labels = build_rejector_labels_gatekeeper(per_client_preds_tr[cid], per_server_preds_tr[cid], client_tr_labels[cid])
        per_rejectors[cid] = train_rejector_on_images(info['train_32'], rej_labels)
        print(f"    Client {cid} ({info['name']}): defer_rate={rej_labels.float().mean():.3f} ({time.time()-t0:.0f}s)")
    
    # Universal rejector (pooled data from all clients)
    print("  Universal rejector:")
    t0 = time.time()
    all_images, all_labels_rej = [], []
    for cid in clients:
        loader = DataLoader(clients[cid]['train_32'], batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        for x, _ in loader:
            all_images.append(x)
        rej_labels = build_rejector_labels_gatekeeper(per_client_preds_tr[cid], uni_server_preds_tr[cid], client_tr_labels[cid])
        all_labels_rej.append(rej_labels)
    all_images = torch.cat(all_images)
    all_labels_rej = torch.cat(all_labels_rej)
    
    uni_rej_dataset = TensorDataset(all_images, all_labels_rej.float())
    uni_rej_loader = DataLoader(uni_rej_dataset, batch_size=BATCH_SIZE, shuffle=True)
    uni_rejector = Rejector().to(device)
    n_pos = all_labels_rej.sum().item(); n_neg = len(all_labels_rej) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt_uni = optim.Adam(uni_rejector.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_uni = optim.lr_scheduler.CosineAnnealingLR(opt_uni, T_max=REJ_EPOCHS)
    uni_rejector.train()
    for _ in range(REJ_EPOCHS):
        for imgs, y in uni_rej_loader:
            imgs, y = imgs.to(device), y.to(device)
            opt_uni.zero_grad(); crit(uni_rejector(imgs).squeeze(), y).backward(); opt_uni.step()
        sched_uni.step()
    uni_rejector.eval()
    print(f"    Pooled defer_rate={all_labels_rej.float().mean():.3f} ({time.time()-t0:.0f}s)")
    
    # ============================================================
    # Results
    # ============================================================
    target_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Baselines
    print(f"\n{'='*75}")
    print("BASELINES (no deferral)")
    print(f"{'='*75}")
    print(f"{'Client':<13} {'Client':>8} {'PerServ':>8} {'UniServ':>8} {'Orc(Per)':>9} {'Orc(Uni)':>9}")
    print("-" * 75)
    for cid, info in clients.items():
        labels = client_te_labels[cid]  # all in original label space
        co = (per_client_preds_te[cid] == labels).float().mean().item()
        ps = (per_server_preds_te[cid] == labels).float().mean().item()
        us = (uni_server_preds_te[cid] == labels).float().mean().item()
        op = ((per_client_preds_te[cid] == labels) | (per_server_preds_te[cid] == labels)).float().mean().item()
        ou = ((per_client_preds_te[cid] == labels) | (uni_server_preds_te[cid] == labels)).float().mean().item()
        print(f"{info['name']:<13} {co:>8.4f} {ps:>8.4f} {us:>8.4f} {op:>9.4f} {ou:>9.4f}")
    
    # Fixed deferral rate comparison
    for rate in target_rates:
        print(f"\n{'='*55}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*55}")
        print(f"{'Client':<13} {'Ours':>8} {'Universal':>10} {'ConfThresh':>10}")
        print("-" * 45)
        
        avgs = {'ours': 0, 'uni': 0, 'conf': 0}
        
        for cid, info in clients.items():
            # --- Ours: Per-client server + Per-client rejector ---
            labels = client_te_labels[cid]
            
            # --- Ours: Per-client server + Per-client rejector (raw images) ---
            scores = get_rejector_scores(per_rejectors[cid], info['test_32'])
            ours_acc, _ = system_acc_at_rate(
                scores, rate, per_client_preds_te[cid], per_server_preds_te[cid], labels)
            
            # --- Universal: Universal server + Universal rejector (raw images) ---
            scores_u = get_rejector_scores(uni_rejector, info['test_32'])
            uni_acc, _ = system_acc_at_rate(
                scores_u, rate, per_client_preds_te[cid], uni_server_preds_te[cid], labels)
            
            # --- ConfThresh: Per-client server + client confidence ---
            conf_scores = -torch.softmax(per_client_logits_te[cid], 1).max(1).values
            conf_acc, _ = system_acc_at_rate(
                conf_scores, rate, per_client_preds_te[cid], per_server_preds_te[cid], labels)
            
            print(f"{info['name']:<13} {ours_acc:>8.4f} {uni_acc:>10.4f} {conf_acc:>10.4f}")
            avgs['ours'] += ours_acc; avgs['uni'] += uni_acc; avgs['conf'] += conf_acc
        
        n = len(clients)
        print("-" * 45)
        print(f"{'Average':<13} {avgs['ours']/n:>8.4f} {avgs['uni']/n:>10.4f} {avgs['conf']/n:>10.4f}")
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
