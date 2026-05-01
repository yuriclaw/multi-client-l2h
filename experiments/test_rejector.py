"""
Stage 2: Per-client vs Universal Rejector Experiment
=====================================================
Two-stage L2H:
  Stage 1: Per-client head on frozen server backbone (done, validated)
  Stage 2: Train rejector to decide client vs server for each sample

Setup:
  - 5 heterogeneous clients: CIFAR-10, SVHN, FashionMNIST, STL-10, GTSRB
  - Client model: small 3-layer CNN per client
  - Server: frozen ResNet-50 + per-client linear head
  - Rejector input: [client_softmax, server_softmax, client_confidence, server_confidence]
  - Rejector output: 0 = use client, 1 = defer to server
  - Label: 1 if server correct & client wrong, 0 if client correct, 
           and for ambiguous cases (both right/both wrong) use server_confidence > client_confidence

Comparison:
  - Per-client rejector: trained on each client's own data
  - Universal rejector: trained on pooled data from all clients
  - Baseline: no deferral (client only), always defer (server only), confidence threshold
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np
import timm
from collections import OrderedDict
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ============================================================
# 1. Data Loading (same as previous experiments)
# ============================================================

def get_client_datasets(n_train=8000, n_test=2000):
    """Load 5 heterogeneous client datasets."""
    
    transform_32 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    clients = {}
    
    # Client 0: CIFAR-10
    ds = torchvision.datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transform_32)
    train_sub = Subset(ds, range(n_train))
    ds_test = torchvision.datasets.CIFAR10('/tmp/data', train=False, download=True, transform=transform_32)
    test_sub = Subset(ds_test, range(n_test))
    clients[0] = {'name': 'CIFAR-10', 'train': train_sub, 'test': test_sub, 'n_classes': 10}
    
    # Client 1: SVHN
    ds = torchvision.datasets.SVHN('/tmp/data', split='train', download=True, transform=transform_32)
    train_sub = Subset(ds, range(n_train))
    ds_test = torchvision.datasets.SVHN('/tmp/data', split='test', download=True, transform=transform_32)
    test_sub = Subset(ds_test, range(n_test))
    clients[1] = {'name': 'SVHN', 'train': train_sub, 'test': test_sub, 'n_classes': 10}
    
    # Client 2: FashionMNIST
    ds = torchvision.datasets.FashionMNIST('/tmp/data', train=True, download=True, transform=transform_gray)
    train_sub = Subset(ds, range(n_train))
    ds_test = torchvision.datasets.FashionMNIST('/tmp/data', train=False, download=True, transform=transform_gray)
    test_sub = Subset(ds_test, range(n_test))
    clients[2] = {'name': 'FashionMNIST', 'train': train_sub, 'test': test_sub, 'n_classes': 10}
    
    # Client 3: STL-10
    ds = torchvision.datasets.STL10('/tmp/data', split='train', download=True, transform=transform_32)
    train_sub = Subset(ds, range(min(n_train, len(ds))))
    ds_test = torchvision.datasets.STL10('/tmp/data', split='test', download=True, transform=transform_32)
    test_sub = Subset(ds_test, range(n_test))
    clients[3] = {'name': 'STL-10', 'train': train_sub, 'test': test_sub, 'n_classes': 10}
    
    # Client 4: GTSRB
    ds = torchvision.datasets.GTSRB('/tmp/data', split='train', download=True, transform=transform_32)
    train_sub = Subset(ds, range(n_train))
    ds_test = torchvision.datasets.GTSRB('/tmp/data', split='test', download=True, transform=transform_32)
    test_sub = Subset(ds_test, range(n_test))
    clients[4] = {'name': 'GTSRB', 'train': train_sub, 'test': test_sub, 'n_classes': 43}
    
    return clients


# ============================================================
# 2. Client Model: Small CNN
# ============================================================

class SmallCNN(nn.Module):
    """Simple 3-conv-layer CNN for 32x32 input."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, n_classes)
    
    def forward(self, x):
        feat = self.features(x).flatten(1)
        return self.classifier(feat)
    
    def get_features(self, x):
        return self.features(x).flatten(1)


# ============================================================
# 3. Rejector Model
# ============================================================

class Rejector(nn.Module):
    """Binary classifier: 0=use client, 1=defer to server.
    Input: concatenation of client softmax + server softmax + scalar features.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, x):
        return self.net(x)


# ============================================================
# 4. Training Helpers
# ============================================================

def train_client_model(client_data, n_classes, epochs=50, lr=0.01):
    """Train a small CNN on client data."""
    model = SmallCNN(n_classes).to(device)
    loader = DataLoader(client_data, batch_size=128, shuffle=True, num_workers=2)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return model


def train_server_head(server_backbone, client_data, n_classes, epochs=100, lr=0.01):
    """Train a linear head on frozen server features."""
    # Extract features first
    server_backbone.eval()
    all_feats, all_labels = [], []
    loader = DataLoader(client_data, batch_size=128, shuffle=False, num_workers=2)
    
    transform_up = transforms.Resize((224, 224), antialias=True)
    
    with torch.no_grad():
        for x, y in loader:
            x = transform_up(x).to(device)
            feat = server_backbone(x)
            all_feats.append(feat.cpu())
            all_labels.append(y)
    
    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)
    
    # Train linear head
    head = nn.Linear(feats.shape[1], n_classes).to(device)
    feat_dataset = TensorDataset(feats, labels)
    loader = DataLoader(feat_dataset, batch_size=256, shuffle=True)
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    head.train()
    for epoch in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(head(f), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    
    return head, feats.shape[1]


def get_predictions(client_model, server_backbone, server_head, data, n_classes_client):
    """Get client and server predictions + softmax for all samples."""
    client_model.eval()
    server_head.eval()
    server_backbone.eval()
    
    transform_up = transforms.Resize((224, 224), antialias=True)
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=2)
    
    all_client_softmax = []
    all_server_softmax = []
    all_client_preds = []
    all_server_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            
            # Client predictions
            client_logits = client_model(x)
            client_sm = torch.softmax(client_logits, dim=1)
            client_pred = client_logits.argmax(dim=1)
            
            # Server predictions
            x_up = transform_up(x)
            server_feat = server_backbone(x_up)
            server_logits = server_head(server_feat)
            server_sm = torch.softmax(server_logits, dim=1)
            server_pred = server_logits.argmax(dim=1)
            
            all_client_softmax.append(client_sm.cpu())
            all_server_softmax.append(server_sm.cpu())
            all_client_preds.append(client_pred.cpu())
            all_server_preds.append(server_pred.cpu())
            all_labels.append(y)
    
    return {
        'client_softmax': torch.cat(all_client_softmax),
        'server_softmax': torch.cat(all_server_softmax),
        'client_preds': torch.cat(all_client_preds),
        'server_preds': torch.cat(all_server_preds),
        'labels': torch.cat(all_labels),
    }


def build_rejector_features(preds_dict, mode='full'):
    """Build rejector input features from predictions.
    mode='full': [client_softmax, server_softmax, summary_stats] (per-client only, variable dim)
    mode='summary': [top-k confidences, entropy, etc.] (fixed dim, for universal rejector)
    """
    cs = preds_dict['client_softmax']  # (N, C_client)
    ss = preds_dict['server_softmax']  # (N, C_server)
    
    client_conf = cs.max(dim=1, keepdim=True).values
    server_conf = ss.max(dim=1, keepdim=True).values
    conf_diff = server_conf - client_conf
    client_entropy = -(cs * (cs + 1e-8).log()).sum(dim=1, keepdim=True)
    server_entropy = -(ss * (ss + 1e-8).log()).sum(dim=1, keepdim=True)
    
    # Top-3 client and server confidences (fixed-size features)
    k = min(3, cs.shape[1], ss.shape[1])
    client_topk = cs.topk(k, dim=1).values  # (N, k)
    server_topk = ss.topk(k, dim=1).values  # (N, k)
    
    # Client and server margin (top1 - top2)
    if cs.shape[1] >= 2:
        client_top2 = cs.topk(2, dim=1).values
        client_margin = (client_top2[:, 0] - client_top2[:, 1]).unsqueeze(1)
    else:
        client_margin = client_conf
    if ss.shape[1] >= 2:
        server_top2 = ss.topk(2, dim=1).values
        server_margin = (server_top2[:, 0] - server_top2[:, 1]).unsqueeze(1)
    else:
        server_margin = server_conf
    
    if mode == 'full':
        features = torch.cat([cs, ss, client_conf, server_conf, conf_diff, client_entropy, server_entropy], dim=1)
    else:  # 'summary' - fixed dimension regardless of n_classes
        features = torch.cat([
            client_topk, server_topk,
            client_conf, server_conf, conf_diff,
            client_entropy, server_entropy,
            client_margin, server_margin
        ], dim=1)  # dim = 2*k + 7 = 13
    
    return features


def build_rejector_labels(preds_dict):
    """Build rejector labels: 1=defer to server is better, 0=client is fine.
    Logic:
      - client correct → 0 (no need to defer)
      - client wrong, server correct → 1 (defer helps)
      - both wrong → 0 (defer doesn't help)
      - both correct → 0 (no need to defer)
    """
    client_correct = (preds_dict['client_preds'] == preds_dict['labels'])
    server_correct = (preds_dict['server_preds'] == preds_dict['labels'])
    
    # Defer only when server is correct and client is wrong
    labels = (~client_correct & server_correct).long()
    return labels


def train_rejector(features, labels, epochs=100, lr=0.001):
    """Train a rejector on given features/labels."""
    input_dim = features.shape[1]
    rejector = Rejector(input_dim).to(device)
    
    dataset = TensorDataset(features, labels.float())
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Handle class imbalance
    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos]).to(device)
    else:
        pos_weight = torch.tensor([1.0]).to(device)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(rejector.parameters(), lr=lr, weight_decay=1e-4)
    
    rejector.train()
    for epoch in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(rejector(f).squeeze(), y)
            loss.backward()
            optimizer.step()
    
    return rejector


def evaluate_system(rejector, test_features, test_preds_dict):
    """Evaluate the full L2H system with a rejector.
    Returns: system_acc, deferral_rate, client_only_acc, server_only_acc
    """
    rejector.eval()
    with torch.no_grad():
        logits = rejector(test_features.to(device)).squeeze().cpu()
        defer = (logits > 0).long()  # 1 = defer to server
    
    labels = test_preds_dict['labels']
    client_preds = test_preds_dict['client_preds']
    server_preds = test_preds_dict['server_preds']
    
    # System prediction: use server when defer=1, client when defer=0
    system_preds = torch.where(defer == 1, server_preds, client_preds)
    
    system_acc = (system_preds == labels).float().mean().item()
    deferral_rate = defer.float().mean().item()
    client_only_acc = (client_preds == labels).float().mean().item()
    server_only_acc = (server_preds == labels).float().mean().item()
    
    return system_acc, deferral_rate, client_only_acc, server_only_acc


def confidence_threshold_baseline(test_preds_dict, threshold=0.5):
    """Baseline: defer when client confidence < threshold."""
    client_conf = test_preds_dict['client_softmax'].max(dim=1).values
    defer = (client_conf < threshold).long()
    
    labels = test_preds_dict['labels']
    client_preds = test_preds_dict['client_preds']
    server_preds = test_preds_dict['server_preds']
    
    system_preds = torch.where(defer == 1, server_preds, client_preds)
    system_acc = (system_preds == labels).float().mean().item()
    deferral_rate = defer.float().mean().item()
    
    return system_acc, deferral_rate


# ============================================================
# 5. Main Experiment
# ============================================================

def main():
    print("=" * 70)
    print("Stage 2: Per-client vs Universal Rejector")
    print("=" * 70)
    
    # Load data
    print("\n[1/6] Loading datasets...")
    clients = get_client_datasets()
    
    # Load frozen server backbone
    print("[2/6] Loading frozen ResNet-50 backbone...")
    backbone_full = timm.create_model('resnet50', pretrained=True).to(device)
    backbone_full.eval()
    # Remove classification head to get features
    server_backbone = nn.Sequential(OrderedDict([
        (name, module) for name, module in backbone_full.named_children()
        if name != 'fc'
    ]))
    server_backbone.add_module('flatten', nn.Flatten())
    server_backbone.eval()
    for p in server_backbone.parameters():
        p.requires_grad = False
    
    # Stage 1: Train client models + server heads
    print("[3/6] Training client models (SmallCNN, 50 epochs each)...")
    client_models = {}
    for i, info in clients.items():
        print(f"  Client {i} ({info['name']})...", end=' ', flush=True)
        t0 = time.time()
        client_models[i] = train_client_model(info['train'], info['n_classes'], epochs=50)
        # Quick eval
        client_models[i].eval()
        correct = 0
        total = 0
        for x, y in DataLoader(info['test'], batch_size=128):
            x = x.to(device)
            pred = client_models[i](x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
        print(f"acc={correct/total:.4f} ({time.time()-t0:.1f}s)")
    
    print("\n[4/6] Training per-client server heads (100 epochs each)...")
    server_heads = {}
    feat_dim = None
    for i, info in clients.items():
        print(f"  Client {i} ({info['name']})...", end=' ', flush=True)
        t0 = time.time()
        head, fd = train_server_head(server_backbone, info['train'], info['n_classes'], epochs=100)
        server_heads[i] = head
        feat_dim = fd
        # Quick eval
        head.eval()
        transform_up = transforms.Resize((224, 224), antialias=True)
        correct = 0
        total = 0
        for x, y in DataLoader(info['test'], batch_size=128):
            x_up = transform_up(x.to(device))
            with torch.no_grad():
                feat = server_backbone(x_up)
                pred = head(feat).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += len(y)
        print(f"acc={correct/total:.4f} ({time.time()-t0:.1f}s)")
    
    # Stage 2: Get predictions and train rejectors
    print("\n[5/6] Computing predictions for all clients...")
    train_preds = {}
    test_preds = {}
    train_features = {}
    test_features = {}
    train_labels_rej = {}
    test_labels_rej = {}
    
    for i, info in clients.items():
        print(f"  Client {i} ({info['name']})...")
        train_preds[i] = get_predictions(client_models[i], server_backbone, server_heads[i], info['train'], info['n_classes'])
        test_preds[i] = get_predictions(client_models[i], server_backbone, server_heads[i], info['test'], info['n_classes'])
        
        train_features[i] = build_rejector_features(train_preds[i], mode='full')
        test_features[i] = build_rejector_features(test_preds[i], mode='full')
        train_labels_rej[i] = build_rejector_labels(train_preds[i])
        test_labels_rej[i] = build_rejector_labels(test_preds[i])
        
        n_defer = train_labels_rej[i].sum().item()
        print(f"    Train: {n_defer}/{len(train_labels_rej[i])} samples should defer ({100*n_defer/len(train_labels_rej[i]):.1f}%)")
    
    print("\n[6/6] Training rejectors...")
    
    # --- Per-client rejectors ---
    print("\n  Per-client rejectors:")
    per_client_rejectors = {}
    for i, info in clients.items():
        per_client_rejectors[i] = train_rejector(train_features[i], train_labels_rej[i], epochs=100)
    
    # --- Build summary features for universal rejector (fixed dim across clients) ---
    train_features_summary = {}
    test_features_summary = {}
    for i in clients:
        train_features_summary[i] = build_rejector_features(train_preds[i], mode='summary')
        test_features_summary[i] = build_rejector_features(test_preds[i], mode='summary')
    
    # --- Universal rejector ---
    print("  Universal rejector (pooled data):")
    all_train_feats = []
    all_train_labels = []
    for i in clients:
        all_train_feats.append(train_features_summary[i])
        all_train_labels.append(train_labels_rej[i])
    uni_feats = torch.cat(all_train_feats)
    uni_labels = torch.cat(all_train_labels)
    universal_rejector = train_rejector(uni_feats, uni_labels, epochs=100)
    
    # ============================================================
    # 6. Evaluation
    # ============================================================
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Client':<15} {'Client-Only':>11} {'Server-Only':>11} {'ConfThresh':>10} {'Per-Rej':>10} {'Uni-Rej':>10} {'Per-Defer%':>10} {'Uni-Defer%':>10}")
    print("-" * 95)
    
    avg = {'client': 0, 'server': 0, 'conf': 0, 'per': 0, 'uni': 0, 'per_def': 0, 'uni_def': 0}
    
    for i, info in clients.items():
        # Per-client rejector
        per_acc, per_def, client_acc, server_acc = evaluate_system(
            per_client_rejectors[i], test_features[i], test_preds[i])
        
        # Universal rejector (uses summary features)
        uni_acc, uni_def, _, _ = evaluate_system(
            universal_rejector, test_features_summary[i], test_preds[i])
        
        # Confidence threshold baseline (sweep for best)
        best_conf_acc = 0
        for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            ca, _ = confidence_threshold_baseline(test_preds[i], thr)
            if ca > best_conf_acc:
                best_conf_acc = ca
        
        print(f"{info['name']:<15} {client_acc:>11.4f} {server_acc:>11.4f} {best_conf_acc:>10.4f} {per_acc:>10.4f} {uni_acc:>10.4f} {per_def:>10.1%} {uni_def:>10.1%}")
        
        avg['client'] += client_acc
        avg['server'] += server_acc
        avg['conf'] += best_conf_acc
        avg['per'] += per_acc
        avg['uni'] += uni_acc
        avg['per_def'] += per_def
        avg['uni_def'] += uni_def
    
    n = len(clients)
    print("-" * 95)
    print(f"{'Average':<15} {avg['client']/n:>11.4f} {avg['server']/n:>11.4f} {avg['conf']/n:>10.4f} {avg['per']/n:>10.4f} {avg['uni']/n:>10.4f} {avg['per_def']/n:>10.1%} {avg['uni_def']/n:>10.1%}")
    
    print(f"\nPer-client rejector vs Universal: {(avg['per']-avg['uni'])/n:+.4f} avg accuracy")
    print(f"Per-client rejector vs Client-only: {(avg['per']-avg['client'])/n:+.4f} avg accuracy")
    print(f"Per-client rejector vs Server-only: {(avg['per']-avg['server'])/n:+.4f} avg accuracy")
    
    # Oracle upper bound
    print("\n--- Oracle (always pick the correct one) ---")
    for i, info in clients.items():
        client_correct = (test_preds[i]['client_preds'] == test_preds[i]['labels'])
        server_correct = (test_preds[i]['server_preds'] == test_preds[i]['labels'])
        oracle_acc = (client_correct | server_correct).float().mean().item()
        print(f"  {info['name']}: {oracle_acc:.4f}")


if __name__ == '__main__':
    main()
