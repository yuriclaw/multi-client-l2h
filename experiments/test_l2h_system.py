"""
Two-Stage L2H System: Fixed Deferral Rate Comparison
=====================================================
3 methods compared at fixed deferral rates:

1. Our Method: Per-client server head + Per-client rejector
2. Universal:  Universal server head + Universal rejector  
3. Conf Threshold: Per-client server head + confidence-based deferral

Client model: ResNet-18 (random init, NOT pretrained)
Server model: ResNet-50 (pretrained, frozen)

Fixes from audit:
- Random subsetting (not sequential) for all datasets
- timm num_classes=0 for backbone feature extraction
- Adam optimizer for all heads
- Consistent normalization (ImageNet stats for backbone)
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_TRAIN = 8000
N_TEST = 2000
DATA_DIR = '/tmp/data'
CLIENT_EPOCHS = 50
HEAD_EPOCHS = 200
REJ_EPOCHS = 100
BATCH_SIZE = 128

# ============================================================
# Data
# ============================================================
def get_transform(grayscale=False):
    t = []
    if grayscale:
        t.append(transforms.Grayscale(3))
    t.extend([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return transforms.Compose(t)

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

CLIENT_CONFIGS = {
    0: {'ds': 'CIFAR10', 'name': 'CIFAR-10', 'gray': False, 'n_classes': 10},
    1: {'ds': 'SVHN', 'name': 'SVHN', 'gray': False, 'n_classes': 10},
    2: {'ds': 'FashionMNIST', 'name': 'FashionMNIST', 'gray': True, 'n_classes': 10},
    3: {'ds': 'STL10', 'name': 'STL-10', 'gray': False, 'n_classes': 10},
    4: {'ds': 'GTSRB', 'name': 'GTSRB', 'gray': False, 'n_classes': 43},
}

def get_client_data():
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tf = get_transform(cfg['gray'])
        train_ds = load_dataset(cfg['ds'], 'train', tf)
        test_ds = load_dataset(cfg['ds'], 'test', tf)
        # Random subsetting (NOT sequential!)
        train_idx = random.sample(range(len(train_ds)), min(N_TRAIN, len(train_ds)))
        test_idx = random.sample(range(len(test_ds)), min(N_TEST, len(test_ds)))
        clients[cid] = {
            'name': cfg['name'],
            'n_classes': cfg['n_classes'],
            'train': Subset(train_ds, train_idx),
            'test': Subset(test_ds, test_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {len(train_idx)} train, {len(test_idx)} test, {cfg['n_classes']} classes")
    return clients

# ============================================================
# Client Model: ResNet-18 (random init, NOT pretrained)
# ============================================================
def make_client_model(n_classes):
    """ResNet-18 random init, adapted for 32x32 input."""
    model = torchvision.models.resnet18(weights=None)
    # Adapt for 32x32: smaller first conv, no maxpool
    model.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, n_classes)
    return model

def train_client_model(model, train_ds, epochs=CLIENT_EPOCHS):
    model = model.to(device)
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
    model.eval()
    return model

def eval_model(model, test_ds):
    model.eval()
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2)
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
            total += len(y)
    return correct / total

# ============================================================
# Server Backbone (frozen ResNet-50 pretrained)
# ============================================================
def make_backbone():
    backbone = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    return backbone

def renormalize_for_backbone(x):
    """Convert from CIFAR normalization to ImageNet normalization."""
    # Undo CIFAR norm
    cifar_mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(x.device)
    cifar_std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1).to(x.device)
    x = x * cifar_std + cifar_mean
    # Apply ImageNet norm
    inet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x.device)
    inet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x.device)
    return (x - inet_mean) / inet_std

def extract_features(backbone, dataset):
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x_up = upsample(renormalize_for_backbone(x.to(device)))
            f = backbone(x_up).cpu()
            feats.append(f); labels.append(y)
    return torch.cat(feats), torch.cat(labels)

# ============================================================
# Server Head (linear)
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
# Rejector
# ============================================================
class Rejector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        return self.net(x)

def build_rejector_features(client_softmax, server_softmax):
    """Fixed-dim summary features for both per-client and universal rejectors."""
    cc = client_softmax.max(1, keepdim=True).values
    sc = server_softmax.max(1, keepdim=True).values
    cd = sc - cc  # confidence diff
    ce = -(client_softmax * (client_softmax + 1e-8).log()).sum(1, keepdim=True)
    se = -(server_softmax * (server_softmax + 1e-8).log()).sum(1, keepdim=True)
    # Top-3 confidences
    k = min(3, client_softmax.shape[1], server_softmax.shape[1])
    ctk = client_softmax.topk(k, 1).values
    stk = server_softmax.topk(k, 1).values
    # Margins
    cm = (client_softmax.topk(min(2, client_softmax.shape[1]), 1).values)
    cm = (cm[:, 0] - cm[:, 1]).unsqueeze(1) if cm.shape[1] >= 2 else cc
    sm = (server_softmax.topk(min(2, server_softmax.shape[1]), 1).values)
    sm = (sm[:, 0] - sm[:, 1]).unsqueeze(1) if sm.shape[1] >= 2 else sc
    return torch.cat([ctk, stk, cc, sc, cd, ce, se, cm, sm], 1)  # dim = 2*k + 7

def build_rejector_labels(client_preds, server_preds, labels):
    """1 = should defer (server correct & client wrong), else 0"""
    return ((client_preds != labels) & (server_preds == labels)).long()

def train_rejector(features, labels, epochs=REJ_EPOCHS):
    rej = Rejector(features.shape[1]).to(device)
    loader = DataLoader(TensorDataset(features, labels.float()), batch_size=256, shuffle=True)
    n_pos = labels.sum().item(); n_neg = len(labels) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    rej.train()
    for _ in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            opt.zero_grad(); crit(rej(f).squeeze(), y).backward(); opt.step()
    rej.eval()
    return rej

# ============================================================
# System evaluation at fixed deferral rate
# ============================================================
def system_acc_at_rate(scores, target_rate, client_preds, server_preds, labels):
    """Higher score = more likely to defer. Uses topk for exact deferral count."""
    n = len(scores)
    k = max(1, int(n * target_rate))  # exact number to defer
    _, top_idx = torch.topk(scores, k)
    defer = torch.zeros(n, dtype=torch.bool)
    defer[top_idx] = True
    system_preds = torch.where(defer, server_preds, client_preds)
    acc = (system_preds == labels).float().mean().item()
    actual_rate = k / n
    return acc, actual_rate

def get_softmax_preds(client_model, backbone, server_head, dataset):
    """Get client and server softmax + predictions for a dataset."""
    client_model.eval(); server_head.eval()
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(device)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    cs_l, ss_l, cp_l, sp_l, lb_l = [], [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            x_d = x.to(device)
            cl = client_model(x_d)  # client uses CIFAR-normalized input
            x_inet = upsample(renormalize_for_backbone(x_d))  # backbone uses ImageNet norm
            sl = server_head(backbone(x_inet))
            cs_l.append(torch.softmax(cl, 1).cpu())
            ss_l.append(torch.softmax(sl, 1).cpu())
            cp_l.append(cl.argmax(1).cpu())
            sp_l.append(sl.argmax(1).cpu())
            lb_l.append(y)
    return {
        'client_sm': torch.cat(cs_l), 'server_sm': torch.cat(ss_l),
        'client_pred': torch.cat(cp_l), 'server_pred': torch.cat(sp_l),
        'labels': torch.cat(lb_l),
    }

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 80)
    print("Two-Stage L2H System Comparison")
    print("=" * 80)
    
    # --- Load data ---
    print("\n[1/7] Loading datasets...")
    clients = get_client_data()
    
    # --- Server backbone ---
    print("\n[2/7] Loading frozen ResNet-50 backbone...")
    backbone = make_backbone()
    feat_dim = 2048  # ResNet-50
    
    # --- Train client models (ResNet-18 random init) ---
    print(f"\n[3/7] Training client models (ResNet-18 random init, {CLIENT_EPOCHS} epochs)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model = make_client_model(info['n_classes'])
        client_models[cid] = train_client_model(model, info['train'])
        acc = eval_model(client_models[cid], info['test'])
        print(f"  Client {cid} ({info['name']}): acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    # --- Extract server features ---
    print("\n[4/7] Extracting server features...")
    tr_feats, tr_labels, te_feats, te_labels = {}, {}, {}, {}
    for cid, info in clients.items():
        tr_feats[cid], tr_labels[cid] = extract_features(backbone, info['train'])
        te_feats[cid], te_labels[cid] = extract_features(backbone, info['test'])
        print(f"  Client {cid} ({info['name']}): train {tr_feats[cid].shape}, test {te_feats[cid].shape}")
    
    # ============================================================
    # Stage 1: Server head training
    # ============================================================
    
    # --- Per-client server heads ---
    print(f"\n[5/7] Stage 1: Training server heads ({HEAD_EPOCHS} epochs)...")
    print("  Per-client heads:")
    per_heads = {}
    for cid, info in clients.items():
        per_heads[cid] = train_head(tr_feats[cid], tr_labels[cid], info['n_classes'])
        acc = eval_head(per_heads[cid], te_feats[cid], te_labels[cid])
        print(f"    {info['name']}: acc={acc:.4f}")
    
    # --- Universal server head (pooled data, unified label space) ---
    print("  Universal head:")
    offsets = {}; off = 0
    for cid, info in clients.items():
        offsets[cid] = off; off += info['n_classes']
    total_classes = off
    uni_feats = torch.cat([tr_feats[cid] for cid in clients])
    uni_labels = torch.cat([tr_labels[cid] + offsets[cid] for cid in clients])
    uni_head = train_head(uni_feats, uni_labels, total_classes)
    for cid, info in clients.items():
        acc = eval_head(uni_head, te_feats[cid], te_labels[cid] + offsets[cid])
        print(f"    {info['name']}: acc={acc:.4f}")
    
    # ============================================================
    # Get predictions for all method combinations
    # ============================================================
    print("\n[6/7] Computing predictions...")
    
    # Per-client server predictions
    per_train_preds, per_test_preds = {}, {}
    for cid, info in clients.items():
        per_train_preds[cid] = get_softmax_preds(client_models[cid], backbone, per_heads[cid], info['train'])
        per_test_preds[cid] = get_softmax_preds(client_models[cid], backbone, per_heads[cid], info['test'])
    
    # Universal server predictions — de-offset server_pred to original label space
    uni_train_preds, uni_test_preds = {}, {}
    for cid, info in clients.items():
        uni_train_preds[cid] = get_softmax_preds(client_models[cid], backbone, uni_head, info['train'])
        uni_test_preds[cid] = get_softmax_preds(client_models[cid], backbone, uni_head, info['test'])
        # De-offset server predictions so all preds & labels are in original space
        uni_train_preds[cid]['server_pred'] = uni_train_preds[cid]['server_pred'] - offsets[cid]
        uni_test_preds[cid]['server_pred'] = uni_test_preds[cid]['server_pred'] - offsets[cid]
        # Labels stay in original space (no offset)
    
    # ============================================================
    # Stage 2: Rejector training
    # ============================================================
    print("\n[7/7] Stage 2: Training rejectors...")
    
    # Per-client rejectors (on per-client server)
    print("  Per-client rejectors:")
    per_rejectors = {}
    for cid in clients:
        p = per_train_preds[cid]
        feats = build_rejector_features(p['client_sm'], p['server_sm'])
        labels = build_rejector_labels(p['client_pred'], p['server_pred'], p['labels'])
        per_rejectors[cid] = train_rejector(feats, labels)
        print(f"    Client {cid}: defer_rate_train={labels.float().mean():.3f}")
    
    # Universal rejector (on universal server)
    print("  Universal rejector:")
    all_feats = []
    all_labels = []
    for cid in clients:
        p = uni_train_preds[cid]
        feats = build_rejector_features(p['client_sm'], p['server_sm'])
        labels = build_rejector_labels(p['client_pred'], p['server_pred'], p['labels'])
        all_feats.append(feats)
        all_labels.append(labels)
    uni_rej_feats = torch.cat(all_feats)
    uni_rej_labels = torch.cat(all_labels)
    uni_rejector = train_rejector(uni_rej_feats, uni_rej_labels)
    print(f"    Pooled defer_rate_train={uni_rej_labels.float().mean():.3f}")
    
    # ============================================================
    # Evaluate at fixed deferral rates
    # ============================================================
    target_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Print baselines first
    print(f"\n{'='*70}")
    print("BASELINES (no deferral)")
    print(f"{'='*70}")
    print(f"{'Client':<13} {'Client':>8} {'PerServ':>8} {'UniServ':>8} {'Orc(Per)':>8} {'Orc(Uni)':>8}")
    print("-" * 70)
    for cid, info in clients.items():
        p = per_test_preds[cid]; u = uni_test_preds[cid]
        co = (p['client_pred'] == p['labels']).float().mean().item()
        ps = (p['server_pred'] == p['labels']).float().mean().item()
        # For universal: server_pred is de-offset, labels are original — both in same space
        us = (u['server_pred'] == u['labels']).float().mean().item()
        op = ((p['client_pred'] == p['labels']) | (p['server_pred'] == p['labels'])).float().mean().item()
        ou = ((u['client_pred'] == u['labels']) | (u['server_pred'] == u['labels'])).float().mean().item()
        # Note: de-offset server_pred may be negative or out of range for wrong-client predictions
        # Those will just count as incorrect, which is correct behavior
        print(f"{info['name']:<13} {co:>8.4f} {ps:>8.4f} {us:>8.4f} {op:>8.4f} {ou:>8.4f}")
    
    # Fixed rate comparison
    for rate in target_rates:
        print(f"\n{'='*70}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*70}")
        print(f"{'Client':<13} {'Ours':>8} {'Universal':>10} {'ConfThresh':>10}")
        print("-" * 45)
        
        avgs = {'ours': 0, 'uni': 0, 'conf': 0}
        
        for cid, info in clients.items():
            # --- Our Method: Per-client server + Per-client rejector ---
            p = per_test_preds[cid]
            feats = build_rejector_features(p['client_sm'], p['server_sm'])
            with torch.no_grad():
                scores = per_rejectors[cid](feats.to(device)).squeeze().cpu()
            ours_acc, _ = system_acc_at_rate(scores, rate, p['client_pred'], p['server_pred'], p['labels'])
            
            # --- Universal: Universal server + Universal rejector ---
            u = uni_test_preds[cid]
            feats_u = build_rejector_features(u['client_sm'], u['server_sm'])
            with torch.no_grad():
                scores_u = uni_rejector(feats_u.to(device)).squeeze().cpu()
            uni_acc, _ = system_acc_at_rate(scores_u, rate, u['client_pred'], u['server_pred'], u['labels'])
            
            # --- Conf Threshold: Per-client server + client confidence ---
            conf_scores = -p['client_sm'].max(1).values  # negative confidence → high score = defer
            conf_acc, _ = system_acc_at_rate(conf_scores, rate, p['client_pred'], p['server_pred'], p['labels'])
            
            print(f"{info['name']:<13} {ours_acc:>8.4f} {uni_acc:>10.4f} {conf_acc:>10.4f}")
            avgs['ours'] += ours_acc; avgs['uni'] += uni_acc; avgs['conf'] += conf_acc
        
        n = len(clients)
        print("-" * 45)
        print(f"{'Average':<13} {avgs['ours']/n:>8.4f} {avgs['uni']/n:>10.4f} {avgs['conf']/n:>10.4f}")
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
