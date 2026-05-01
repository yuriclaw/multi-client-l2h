"""
Fixed Deferral Rate Comparison
==============================
For each method, adjust threshold so that deferral rate = target (e.g. 10%, 30%, 50%, 70%, 90%).
Then compare system accuracy at the same deferral rate.

Methods:
1. Per-client server + Per-client rejector
2. Per-client server + Universal rejector  
3. Universal server + Per-client rejector
4. Universal server + Universal rejector
5. Per-client server + Confidence threshold
6. Universal server + Confidence threshold
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
# Data Loading
# ============================================================
def get_client_datasets(n_train=8000, n_test=2000):
    transform_32 = transforms.Compose([
        transforms.Resize((32, 32)), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)), transforms.Grayscale(3), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    clients = {}
    ds = torchvision.datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transform_32)
    ds_t = torchvision.datasets.CIFAR10('/tmp/data', train=False, download=True, transform=transform_32)
    clients[0] = {'name': 'CIFAR-10', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_t, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.SVHN('/tmp/data', split='train', download=True, transform=transform_32)
    ds_t = torchvision.datasets.SVHN('/tmp/data', split='test', download=True, transform=transform_32)
    clients[1] = {'name': 'SVHN', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_t, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.FashionMNIST('/tmp/data', train=True, download=True, transform=transform_gray)
    ds_t = torchvision.datasets.FashionMNIST('/tmp/data', train=False, download=True, transform=transform_gray)
    clients[2] = {'name': 'FashionMNIST', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_t, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.STL10('/tmp/data', split='train', download=True, transform=transform_32)
    ds_t = torchvision.datasets.STL10('/tmp/data', split='test', download=True, transform=transform_32)
    clients[3] = {'name': 'STL-10', 'train': Subset(ds, range(min(n_train, len(ds)))), 'test': Subset(ds_t, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.GTSRB('/tmp/data', split='train', download=True, transform=transform_32)
    ds_t = torchvision.datasets.GTSRB('/tmp/data', split='test', download=True, transform=transform_32)
    clients[4] = {'name': 'GTSRB', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_t, range(n_test)), 'n_classes': 43}
    return clients

# ============================================================
# Models
# ============================================================
class SmallCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, n_classes)
    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))

class Rejector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x)

# ============================================================
# Training helpers
# ============================================================
def train_client_model(data, n_classes, epochs=50):
    model = SmallCNN(n_classes).to(device)
    loader = DataLoader(data, batch_size=128, shuffle=True, num_workers=2)
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
    return model

def extract_features(backbone, data, batch_size=128):
    backbone.eval()
    up = transforms.Resize((224, 224), antialias=True)
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(up(x.to(device))).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

def train_head(feats, labels, n_classes, epochs=200):
    head = nn.Linear(feats.shape[1], n_classes).to(device)
    loader = DataLoader(TensorDataset(feats, labels), batch_size=256, shuffle=True)
    opt = optim.SGD(head.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    head.train()
    for _ in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            opt.zero_grad(); crit(head(f), y).backward(); opt.step()
        sched.step()
    return head

def get_predictions(client_model, backbone, server_head, data):
    client_model.eval(); server_head.eval(); backbone.eval()
    up = transforms.Resize((224, 224), antialias=True)
    loader = DataLoader(data, batch_size=128, shuffle=False, num_workers=2)
    cs_list, ss_list, cp_list, sp_list, lb_list = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            cl = client_model(x)
            sl = server_head(backbone(up(x)))
            cs_list.append(torch.softmax(cl, 1).cpu())
            ss_list.append(torch.softmax(sl, 1).cpu())
            cp_list.append(cl.argmax(1).cpu())
            sp_list.append(sl.argmax(1).cpu())
            lb_list.append(y)
    return {
        'client_softmax': torch.cat(cs_list), 'server_softmax': torch.cat(ss_list),
        'client_preds': torch.cat(cp_list), 'server_preds': torch.cat(sp_list),
        'labels': torch.cat(lb_list),
    }

def build_features(preds, mode='full'):
    cs, ss = preds['client_softmax'], preds['server_softmax']
    cc = cs.max(1, keepdim=True).values
    sc = ss.max(1, keepdim=True).values
    cd = sc - cc
    ce = -(cs * (cs + 1e-8).log()).sum(1, keepdim=True)
    se = -(ss * (ss + 1e-8).log()).sum(1, keepdim=True)
    k = min(3, cs.shape[1], ss.shape[1])
    ctk = cs.topk(k, 1).values
    stk = ss.topk(k, 1).values
    cm = (cs.topk(2, 1).values[:, 0] - cs.topk(2, 1).values[:, 1]).unsqueeze(1) if cs.shape[1] >= 2 else cc
    sm = (ss.topk(2, 1).values[:, 0] - ss.topk(2, 1).values[:, 1]).unsqueeze(1) if ss.shape[1] >= 2 else sc
    if mode == 'full':
        return torch.cat([cs, ss, cc, sc, cd, ce, se], 1)
    else:
        return torch.cat([ctk, stk, cc, sc, cd, ce, se, cm, sm], 1)

def build_labels(preds):
    return (~(preds['client_preds'] == preds['labels']) & (preds['server_preds'] == preds['labels'])).long()

def train_rejector(features, labels, epochs=100):
    rej = Rejector(features.shape[1]).to(device)
    loader = DataLoader(TensorDataset(features, labels.float()), batch_size=256, shuffle=True)
    n_pos = labels.sum().item(); n_neg = len(labels) - n_pos
    pw = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(rej.parameters(), lr=0.001, weight_decay=1e-4)
    rej.train()
    for _ in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            opt.zero_grad(); crit(rej(f).squeeze(), y).backward(); opt.step()
    return rej

def system_acc_at_rate(scores, target_rate, client_preds, server_preds, labels):
    """Given continuous scores (higher = more likely to defer),
    find threshold so that deferral_rate ~ target_rate, return system accuracy."""
    sorted_scores = torch.sort(scores).values
    n = len(scores)
    # Find threshold: top target_rate fraction defers
    idx = max(0, int(n * (1 - target_rate)))
    threshold = sorted_scores[idx].item()
    defer = (scores >= threshold).long()
    actual_rate = defer.float().mean().item()
    system_preds = torch.where(defer == 1, server_preds, client_preds)
    acc = (system_preds == labels).float().mean().item()
    return acc, actual_rate

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 80)
    print("Fixed Deferral Rate Comparison")
    print("=" * 80)
    
    clients = get_client_datasets()
    
    # Backbone
    print("\nLoading frozen ResNet-50...")
    bfull = timm.create_model('resnet50', pretrained=True).to(device)
    backbone = nn.Sequential(OrderedDict([(n, m) for n, m in bfull.named_children() if n != 'fc']))
    backbone.add_module('flatten', nn.Flatten()); backbone.eval()
    for p in backbone.parameters(): p.requires_grad = False
    
    # Train client models
    print("\n[1] Training client models...")
    client_models = {}
    for i, info in clients.items():
        t0 = time.time()
        client_models[i] = train_client_model(info['train'], info['n_classes'])
        client_models[i].eval()
        correct = sum((client_models[i](x.to(device)).argmax(1).cpu() == y).sum().item()
                      for x, y in DataLoader(info['test'], batch_size=128))
        print(f"  Client {i} ({info['name']}): acc={correct/len(info['test']):.4f} ({time.time()-t0:.0f}s)")
    
    # Extract features
    print("\n[2] Extracting server features...")
    tr_feats, tr_labels, te_feats, te_labels = {}, {}, {}, {}
    for i, info in clients.items():
        tr_feats[i], tr_labels[i] = extract_features(backbone, info['train'])
        te_feats[i], te_labels[i] = extract_features(backbone, info['test'])
    
    # Train per-client server heads
    print("\n[3] Training per-client server heads (200 epochs)...")
    per_heads = {}
    for i, info in clients.items():
        per_heads[i] = train_head(tr_feats[i], tr_labels[i], info['n_classes'])
        acc = ((per_heads[i](te_feats[i].to(device)).argmax(1).cpu() == te_labels[i]).float().mean().item())
        print(f"  {info['name']}: {acc:.4f}")
    
    # Train universal server head
    print("\n[4] Training universal server head (200 epochs)...")
    offsets = {}; off = 0
    for i, info in clients.items(): offsets[i] = off; off += info['n_classes']
    total_cls = off
    uni_f = torch.cat([tr_feats[i] for i in clients])
    uni_l = torch.cat([tr_labels[i] + offsets[i] for i in clients])
    uni_head = train_head(uni_f, uni_l, total_cls)
    for i, info in clients.items():
        acc = ((uni_head(te_feats[i].to(device)).argmax(1).cpu() == te_labels[i] + offsets[i]).float().mean().item())
        print(f"  {info['name']}: {acc:.4f}")
    
    # Get predictions for all combinations
    print("\n[5] Getting predictions...")
    # Per server predictions
    per_preds_tr, per_preds_te = {}, {}
    uni_preds_tr, uni_preds_te = {}, {}
    for i, info in clients.items():
        per_preds_tr[i] = get_predictions(client_models[i], backbone, per_heads[i], info['train'])
        per_preds_te[i] = get_predictions(client_models[i], backbone, per_heads[i], info['test'])
        uni_preds_tr[i] = get_predictions(client_models[i], backbone, uni_head, info['train'])
        uni_preds_te[i] = get_predictions(client_models[i], backbone, uni_head, info['test'])
        # Fix universal server preds (offset labels)
        uni_preds_tr[i]['labels'] = uni_preds_tr[i]['labels'] + offsets[i]
        uni_preds_te[i]['labels'] = uni_preds_te[i]['labels'] + offsets[i]
    
    # Train rejectors
    print("\n[6] Training rejectors...")
    # Per-client rejector with per-client server
    per_rej_per = {}
    for i in clients:
        f = build_features(per_preds_tr[i], 'full')
        l = build_labels(per_preds_tr[i])
        per_rej_per[i] = train_rejector(f, l)
    
    # Universal rejector with per-client server (summary features)
    all_f = torch.cat([build_features(per_preds_tr[i], 'summary') for i in clients])
    all_l = torch.cat([build_labels(per_preds_tr[i]) for i in clients])
    uni_rej_per = train_rejector(all_f, all_l)
    
    # Per-client rejector with universal server
    per_rej_uni = {}
    for i in clients:
        f = build_features(uni_preds_tr[i], 'full')
        l = build_labels(uni_preds_tr[i])
        per_rej_uni[i] = train_rejector(f, l)
    
    # Universal rejector with universal server
    all_f2 = torch.cat([build_features(uni_preds_tr[i], 'summary') for i in clients])
    all_l2 = torch.cat([build_labels(uni_preds_tr[i]) for i in clients])
    uni_rej_uni = train_rejector(all_f2, all_l2)
    
    # ============================================================
    # Evaluate at fixed deferral rates
    # ============================================================
    target_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for rate in target_rates:
        print(f"\n{'='*80}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*80}")
        print(f"{'Client':<13} {'PerS+PerR':>9} {'PerS+UniR':>9} {'PerS+Conf':>9} {'UniS+PerR':>9} {'UniS+UniR':>9} {'UniS+Conf':>9}")
        print("-" * 75)
        
        avgs = {k: 0 for k in ['pp', 'pu', 'pc', 'up', 'uu', 'uc']}
        
        for i, info in clients.items():
            results = {}
            
            # --- Per-client server methods ---
            pte = per_preds_te[i]
            cp, sp, lb = pte['client_preds'], pte['server_preds'], pte['labels']
            
            # Per-client rejector score (logit)
            f_full = build_features(pte, 'full')
            per_rej_per[i].eval()
            with torch.no_grad():
                scores = per_rej_per[i](f_full.to(device)).squeeze().cpu()
            results['pp'], _ = system_acc_at_rate(scores, rate, cp, sp, lb)
            
            # Universal rejector score
            f_sum = build_features(pte, 'summary')
            uni_rej_per.eval()
            with torch.no_grad():
                scores = uni_rej_per(f_sum.to(device)).squeeze().cpu()
            results['pu'], _ = system_acc_at_rate(scores, rate, cp, sp, lb)
            
            # Confidence threshold (use negative client confidence as score → high score = defer)
            conf_scores = -pte['client_softmax'].max(1).values
            results['pc'], _ = system_acc_at_rate(conf_scores, rate, cp, sp, lb)
            
            # --- Universal server methods ---
            ute = uni_preds_te[i]
            cp2, sp2, lb2 = ute['client_preds'], ute['server_preds'], ute['labels']
            
            f_full2 = build_features(ute, 'full')
            per_rej_uni[i].eval()
            with torch.no_grad():
                scores = per_rej_uni[i](f_full2.to(device)).squeeze().cpu()
            results['up'], _ = system_acc_at_rate(scores, rate, cp2, sp2, lb2)
            
            f_sum2 = build_features(ute, 'summary')
            uni_rej_uni.eval()
            with torch.no_grad():
                scores = uni_rej_uni(f_sum2.to(device)).squeeze().cpu()
            results['uu'], _ = system_acc_at_rate(scores, rate, cp2, sp2, lb2)
            
            conf_scores2 = -ute['client_softmax'].max(1).values
            results['uc'], _ = system_acc_at_rate(conf_scores2, rate, cp2, sp2, lb2)
            
            print(f"{info['name']:<13} {results['pp']:>9.4f} {results['pu']:>9.4f} {results['pc']:>9.4f} {results['up']:>9.4f} {results['uu']:>9.4f} {results['uc']:>9.4f}")
            
            for k in avgs: avgs[k] += results[k]
        
        n = len(clients)
        print("-" * 75)
        print(f"{'Average':<13} {avgs['pp']/n:>9.4f} {avgs['pu']/n:>9.4f} {avgs['pc']/n:>9.4f} {avgs['up']/n:>9.4f} {avgs['uu']/n:>9.4f} {avgs['uc']/n:>9.4f}")
    
    # Also print client-only and server-only baselines
    print(f"\n{'='*80}")
    print("BASELINES (no deferral)")
    print(f"{'='*80}")
    print(f"{'Client':<13} {'Client-Only':>11} {'Per-Server':>11} {'Uni-Server':>11} {'Oracle(Per)':>11} {'Oracle(Uni)':>11}")
    print("-" * 70)
    for i, info in clients.items():
        pte = per_preds_te[i]
        ute = uni_preds_te[i]
        co = (pte['client_preds'] == pte['labels']).float().mean().item()
        ps = (pte['server_preds'] == pte['labels']).float().mean().item()
        us = (ute['server_preds'] == ute['labels']).float().mean().item()
        orc_p = ((pte['client_preds'] == pte['labels']) | (pte['server_preds'] == pte['labels'])).float().mean().item()
        orc_u = ((ute['client_preds'] == ute['labels']) | (ute['server_preds'] == ute['labels'])).float().mean().item()
        print(f"{info['name']:<13} {co:>11.4f} {ps:>11.4f} {us:>11.4f} {orc_p:>11.4f} {orc_u:>11.4f}")

if __name__ == '__main__':
    main()
