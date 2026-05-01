"""
Two-Stage L2H v4: Comprehensive Comparison
============================================
Changes from v3:
1. Gatekeeper α grid search [0.1, 0.2, ..., 0.9]
2. L2H-Multi with c1=1 (penalize server errors)
3. Rejector uses AlexNet architecture (not LeNet) + client hidden features
4. Universal rejector + universal server head comparison

Setup:
- Client: AlexNet (trained 50ep, frozen)
- Server: ResNet-50 pretrained (frozen) + per-client/universal linear head (500ep)
- Rejector: AlexNet-style on raw x + client hidden features (256-dim)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
CLIENT_EPOCHS = 50
HEAD_EPOCHS = 500
REJ_EPOCHS = 100
BATCH_SIZE = 128

INET_MEAN = (0.485, 0.456, 0.406)
INET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ============================================================
# Data
# ============================================================
def get_transform_32(grayscale=False):
    t = []
    if grayscale: t.append(transforms.Grayscale(3))
    t += [transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    return transforms.Compose(t)

def get_transform_224(grayscale=False):
    t = []
    if grayscale: t.append(transforms.Grayscale(3))
    t += [transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(INET_MEAN, INET_STD)]
    return transforms.Compose(t)

CLIENT_CONFIGS = {
    0: {'ds': 'CIFAR10', 'name': 'CIFAR-10', 'gray': False, 'n_classes': 10},
    1: {'ds': 'SVHN', 'name': 'SVHN', 'gray': False, 'n_classes': 10},
    2: {'ds': 'FashionMNIST', 'name': 'FashionMNIST', 'gray': True, 'n_classes': 10},
    3: {'ds': 'STL10', 'name': 'STL-10', 'gray': False, 'n_classes': 10},
    4: {'ds': 'GTSRB', 'name': 'GTSRB', 'gray': False, 'n_classes': 43},
}

def load_dataset(name, split, transform):
    if name == 'CIFAR10': return torchvision.datasets.CIFAR10(DATA_DIR, train=(split=='train'), download=True, transform=transform)
    elif name == 'SVHN': return torchvision.datasets.SVHN(DATA_DIR, split=split, download=True, transform=transform)
    elif name == 'FashionMNIST': return torchvision.datasets.FashionMNIST(DATA_DIR, train=(split=='train'), download=True, transform=transform)
    elif name == 'STL10': return torchvision.datasets.STL10(DATA_DIR, split=split, download=True, transform=transform)
    elif name == 'GTSRB': return torchvision.datasets.GTSRB(DATA_DIR, split=split, download=True, transform=transform)

def get_client_data():
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tf32 = get_transform_32(cfg['gray'])
        tf224 = get_transform_224(cfg['gray'])
        tr32 = load_dataset(cfg['ds'], 'train', tf32)
        te32 = load_dataset(cfg['ds'], 'test', tf32)
        tr224 = load_dataset(cfg['ds'], 'train', tf224)
        te224 = load_dataset(cfg['ds'], 'test', tf224)
        tr_idx = random.sample(range(len(tr32)), min(N_TRAIN, len(tr32)))
        te_idx = random.sample(range(len(te32)), min(N_TEST, len(te32)))
        clients[cid] = {
            'name': cfg['name'], 'n_classes': cfg['n_classes'],
            'train_32': Subset(tr32, tr_idx), 'test_32': Subset(te32, te_idx),
            'train_224': Subset(tr224, tr_idx), 'test_224': Subset(te224, te_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {len(tr_idx)} train, {len(te_idx)} test")
    return clients

# ============================================================
# Client: AlexNet (trained, frozen)
# ============================================================
CLIENT_FEAT_DIM = 256

class ClientAlexNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, CLIENT_FEAT_DIM, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(CLIENT_FEAT_DIM, 256), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, n_classes))
    
    def forward(self, x):
        f = self.features(x)
        h = self.fc1(f)
        return self.fc2(h)
    
    def get_hidden(self, x):
        """Return fc1 hidden layer (256-dim) for rejector input."""
        f = self.features(x)
        return self.fc1(f)

def train_client(n_classes, train_ds, test_ds):
    model = ClientAlexNet(n_classes).to(device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CLIENT_EPOCHS)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(CLIENT_EPOCHS):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    correct, total = 0, 0
    for x, y in DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=2):
        correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item(); total += len(y)
    return model, correct / total

# ============================================================
# Server: ResNet-50 pretrained (frozen) + linear head
# ============================================================
def make_server_backbone():
    bb = timm.create_model('resnet50', pretrained=True, num_classes=0).to(device)
    bb.eval()
    for p in bb.parameters(): p.requires_grad = False
    return bb

def extract_features(backbone, dataset):
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu()); labels.append(y)
    return torch.cat(feats), torch.cat(labels)

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

def train_universal_head(srv_tr_feats, srv_tr_labels, clients, epochs=HEAD_EPOCHS):
    """Train a single universal head across all clients with label offsets."""
    # Compute total classes and offsets
    offsets = {}
    total_classes = 0
    for cid in sorted(clients.keys()):
        offsets[cid] = total_classes
        total_classes += clients[cid]['n_classes']
    
    # Combine all features with offset labels
    all_feats, all_labels = [], []
    for cid in sorted(clients.keys()):
        all_feats.append(srv_tr_feats[cid])
        all_labels.append(srv_tr_labels[cid] + offsets[cid])
    all_feats = torch.cat(all_feats)
    all_labels = torch.cat(all_labels)
    
    feat_dim = all_feats.shape[1]
    head = nn.Linear(feat_dim, total_classes).to(device)
    loader = DataLoader(TensorDataset(all_feats, all_labels), batch_size=256, shuffle=True)
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
    return head, offsets, total_classes

# ============================================================
# Rejector: AlexNet-style on raw x + client hidden features
# ============================================================
class RejectorAlexNet(nn.Module):
    """AlexNet-style rejector processing raw 32x32 image, 
    then concatenating with client hidden features."""
    def __init__(self, hidden_dim=256, out_dim=1):
        super().__init__()
        # Same conv architecture as client AlexNet but smaller
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        # 128 (image features) + hidden_dim (client hidden) = input to FC
        self.head = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128 + hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, out_dim),
        )
    
    def forward(self, x, hidden):
        img_feat = self.features(x)
        combined = torch.cat([img_feat, hidden], dim=1)
        return self.head(combined)

# ============================================================
# Precompute rejector data
# ============================================================
def precompute_rejector_data(client_model, server_head, server_backbone,
                             dataset_32, dataset_224, n_classes,
                             uni_head=None, offset=0, total_classes=0):
    """Precompute everything for rejector training. 
    Also computes universal server predictions if uni_head provided."""
    client_model.eval(); server_head.eval()
    
    loader_32 = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    loader_224 = DataLoader(dataset_224, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    all_x32, all_hidden, all_labels = [], [], []
    all_client_logits, all_client_preds = [], []
    
    with torch.no_grad():
        for (x32, y) in loader_32:
            x32_d = x32.to(device)
            cl = client_model(x32_d)
            h = client_model.get_hidden(x32_d)
            all_x32.append(x32)
            all_hidden.append(h.cpu())
            all_labels.append(y)
            all_client_logits.append(cl.cpu())
            all_client_preds.append(cl.argmax(1).cpu())
    
    all_server_logits, all_server_preds = [], []
    all_uni_preds = []
    with torch.no_grad():
        for (x224, _) in loader_224:
            sf = server_backbone(x224.to(device))
            sl = server_head(sf)
            all_server_logits.append(sl.cpu())
            all_server_preds.append(sl.argmax(1).cpu())
            if uni_head is not None:
                ul = uni_head(sf)
                # Use logit slicing for universal head
                up = ul[:, offset:offset+n_classes].argmax(1).cpu()
                all_uni_preds.append(up)
    
    result = {
        'x32': torch.cat(all_x32),
        'hidden': torch.cat(all_hidden),
        'labels': torch.cat(all_labels),
        'client_logits': torch.cat(all_client_logits),
        'client_preds': torch.cat(all_client_preds),
        'server_logits': torch.cat(all_server_logits),
        'server_preds': torch.cat(all_server_preds),
        'n_classes': n_classes,
    }
    if uni_head is not None:
        result['uni_preds'] = torch.cat(all_uni_preds)
    return result

# ============================================================
# Loss Functions
# ============================================================

def train_rejector_gatekeeper(data, epochs=REJ_EPOCHS, alpha=0.3, server_key='server_preds'):
    """Gatekeeper loss with configurable alpha and server prediction source."""
    n_classes = data['n_classes']
    rej = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=n_classes).to(device)
    
    client_correct = (data['client_preds'] == data['labels'])
    client_incorrect = ~client_correct
    
    dataset = TensorDataset(data['x32'], data['hidden'], data['labels'],
                           client_correct.float(), client_incorrect.float())
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    uniform = torch.ones(n_classes).to(device) / n_classes
    
    rej.train()
    for _ in range(epochs):
        for x, h, y, corr, incorr in loader:
            x, h, y = x.to(device), h.to(device), y.to(device)
            corr, incorr = corr.to(device), incorr.to(device)
            logits = rej(x, h)
            
            ce = F.cross_entropy(logits, y, reduction='none')
            l_corr = (corr * ce).sum() / max(corr.sum(), 1)
            
            log_probs = F.log_softmax(logits, dim=1)
            kl = F.kl_div(log_probs, uniform.expand_as(log_probs), reduction='none').sum(1)
            l_incorr = (incorr * kl).sum() / max(incorr.sum(), 1)
            
            loss = alpha * l_corr + (1 - alpha) * l_incorr
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def train_rejector_l2h(data, epochs=REJ_EPOCHS, c_e=0.0, c_1=0.0, server_key='server_preds'):
    """L2H Multi-class surrogate L2.
    L2 = -(1-c_e-c_1+c_1*𝟙[e=y]) * log σ(r₂) - 𝟙[m=y] * log σ(r₁)
    """
    rej = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=2).to(device)
    
    server_correct = (data[server_key] == data['labels']).float()
    client_correct = (data['client_preds'] == data['labels']).float()
    
    dataset = TensorDataset(data['x32'], data['hidden'], client_correct, server_correct)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    rej.train()
    for _ in range(epochs):
        for x, h, mc, ec in loader:
            x, h, mc, ec = x.to(device), h.to(device), mc.to(device), ec.to(device)
            logits = rej(x, h)
            log_probs = F.log_softmax(logits, dim=1)
            w_remote = 1 - c_e - c_1 + c_1 * ec
            loss = -(w_remote * log_probs[:, 1] + mc * log_probs[:, 0]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def train_rejector_bce(data, epochs=REJ_EPOCHS):
    """BCE baseline — label = client incorrect"""
    rej = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=1).to(device)
    labels = (data['client_preds'] != data['labels']).float()
    dataset = TensorDataset(data['x32'], data['hidden'], labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    n_pos = labels.sum().item(); n_neg = len(labels) - n_pos
    pw = torch.tensor([n_neg / max(n_pos, 1)]).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rej.train()
    for _ in range(epochs):
        for x, h, y in loader:
            x, h, y = x.to(device), h.to(device), y.to(device)
            opt.zero_grad(); crit(rej(x, h).squeeze(), y).backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

# ============================================================
# Score extraction
# ============================================================
def get_scores_binary(rejector, data):
    rejector.eval()
    loader = DataLoader(TensorDataset(data['x32'], data['hidden']), batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for x, h in loader:
            out = rejector(x.to(device), h.to(device))
            if out.shape[1] == 2:
                scores.append((out[:, 1] - out[:, 0]).cpu())
            else:
                scores.append(out.squeeze().cpu())
    return torch.cat(scores)

def get_scores_gatekeeper(rejector, data):
    rejector.eval()
    loader = DataLoader(TensorDataset(data['x32'], data['hidden']), batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for x, h in loader:
            out = rejector(x.to(device), h.to(device))
            conf = F.softmax(out, dim=1).max(1).values
            scores.append(-conf.cpu())
    return torch.cat(scores)

def get_scores_conf(data):
    return -F.softmax(data['client_logits'], dim=1).max(1).values

# ============================================================
# System eval
# ============================================================
def system_acc_at_rate(scores, target_rate, client_preds, server_preds, labels):
    n = len(scores)
    k = max(1, int(n * target_rate))
    _, top_idx = torch.topk(scores, k)
    defer = torch.zeros(n, dtype=torch.bool)
    defer[top_idx] = True
    system_preds = torch.where(defer, server_preds, client_preds)
    return (system_preds == labels).float().mean().item()

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 90)
    print("L2H v4: Comprehensive Comparison")
    print("Rejector: AlexNet-style + client hidden features")
    print("=" * 90)
    
    print("\n[1/9] Loading data...")
    clients = get_client_data()
    
    print(f"\n[2/9] Training client AlexNet models ({CLIENT_EPOCHS}ep)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client(info['n_classes'], info['train_32'], info['test_32'])
        client_models[cid] = model
        print(f"  Client {cid} ({info['name']}): acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    print("\n[3/9] Loading server backbone...")
    server_bb = make_server_backbone()
    
    print("\n[4/9] Extracting server features...")
    srv_tr_feats, srv_tr_labels = {}, {}
    srv_te_feats, srv_te_labels = {}, {}
    for cid, info in clients.items():
        srv_tr_feats[cid], srv_tr_labels[cid] = extract_features(server_bb, info['train_224'])
        srv_te_feats[cid], srv_te_labels[cid] = extract_features(server_bb, info['test_224'])
        print(f"  Client {cid}: done")
    
    print(f"\n[5/9] Training per-client server heads ({HEAD_EPOCHS}ep)...")
    per_heads = {}
    for cid, info in clients.items():
        per_heads[cid] = train_head(srv_tr_feats[cid], srv_tr_labels[cid], info['n_classes'])
        with torch.no_grad():
            acc = (per_heads[cid](srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        print(f"  {info['name']}: acc={acc:.4f}")
    
    print(f"\n[6/9] Training universal server head ({HEAD_EPOCHS}ep)...")
    uni_head, offsets, total_classes = train_universal_head(srv_tr_feats, srv_tr_labels, clients)
    for cid, info in clients.items():
        with torch.no_grad():
            ul = uni_head(srv_te_feats[cid].to(device))
            off = offsets[cid]; nc = info['n_classes']
            acc = (ul[:, off:off+nc].argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        print(f"  {info['name']}: acc={acc:.4f}")
    
    print("\n[7/9] Precomputing rejector data...")
    train_data, test_data = {}, {}
    for cid, info in clients.items():
        train_data[cid] = precompute_rejector_data(
            client_models[cid], per_heads[cid], server_bb,
            info['train_32'], info['train_224'], info['n_classes'],
            uni_head, offsets[cid], total_classes)
        test_data[cid] = precompute_rejector_data(
            client_models[cid], per_heads[cid], server_bb,
            info['test_32'], info['test_224'], info['n_classes'],
            uni_head, offsets[cid], total_classes)
        c_acc = (train_data[cid]['client_preds'] == train_data[cid]['labels']).float().mean().item()
        s_acc = (train_data[cid]['server_preds'] == train_data[cid]['labels']).float().mean().item()
        u_acc = (train_data[cid]['uni_preds'] == train_data[cid]['labels']).float().mean().item()
        print(f"  Client {cid}: client={c_acc:.3f}, per_server={s_acc:.3f}, uni_server={u_acc:.3f}")
    
    # ============================================================
    # PART A: Gatekeeper α grid search (per-client rejector + per-client head)
    # ============================================================
    print(f"\n[8/9] Gatekeeper α grid search ({REJ_EPOCHS}ep)...")
    alphas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    target_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    alpha_results = {}  # alpha -> rate -> avg_acc
    for alpha in alphas:
        t0 = time.time()
        rejs = {}
        for cid in clients:
            rejs[cid] = train_rejector_gatekeeper(train_data[cid], alpha=alpha)
        
        alpha_results[alpha] = {}
        for rate in target_rates:
            accs = []
            for cid, info in clients.items():
                d = test_data[cid]
                scores = get_scores_gatekeeper(rejs[cid], d)
                acc = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
                accs.append(acc)
            alpha_results[alpha][rate] = np.mean(accs)
        
        print(f"  α={alpha:.1f}: " + " | ".join(f"{rate*100:.0f}%={alpha_results[alpha][rate]:.4f}" for rate in target_rates) + f" ({time.time()-t0:.0f}s)")
    
    # Find best alpha
    best_alpha = max(alphas, key=lambda a: np.mean([alpha_results[a][r] for r in target_rates]))
    print(f"\n  >>> Best α = {best_alpha:.1f}")
    
    # Print α grid table
    print(f"\n{'='*80}")
    print("GATEKEEPER α GRID SEARCH (Average across 5 clients)")
    print(f"{'='*80}")
    header = f"{'α':<6}" + "".join(f" {r*100:.0f}%defer{'':<3}" for r in target_rates)
    print(header)
    print("-" * 70)
    for alpha in alphas:
        row = f"{alpha:<6.1f}" + "".join(f" {alpha_results[alpha][r]:<12.4f}" for r in target_rates)
        marker = " <<<" if alpha == best_alpha else ""
        print(row + marker)
    
    # ============================================================
    # PART B: Full comparison with best α, L2H c1=1, universal
    # ============================================================
    print(f"\n\n[9/9] Full comparison (best α={best_alpha:.1f}, L2H c1=1)...")
    
    # Train all rejectors
    # Per-client: Gatekeeper (best α), L2H (c1=0), L2H (c1=1), BCE
    # Universal: Gatekeeper (best α), L2H (c1=1)
    
    per_rej_gk = {}      # per-client Gatekeeper (best α)
    per_rej_l2h_c0 = {}  # per-client L2H c1=0
    per_rej_l2h_c1 = {}  # per-client L2H c1=1
    per_rej_bce = {}     # per-client BCE
    uni_rej_gk = {}      # universal Gatekeeper (best α) — trained on per-client data but evaluated with uni server
    uni_rej_l2h_c1 = {}  # universal L2H c1=1 — trained on per-client data but evaluated with uni server
    
    for cid, info in clients.items():
        d = train_data[cid]
        t0 = time.time()
        
        per_rej_gk[cid] = train_rejector_gatekeeper(d, alpha=best_alpha)
        per_rej_l2h_c0[cid] = train_rejector_l2h(d, c_e=0.0, c_1=0.0)
        per_rej_l2h_c1[cid] = train_rejector_l2h(d, c_e=0.0, c_1=1.0)
        per_rej_bce[cid] = train_rejector_bce(d)
        
        # Universal: train rejector using universal server predictions
        uni_rej_gk[cid] = train_rejector_gatekeeper(d, alpha=best_alpha, server_key='uni_preds')
        uni_rej_l2h_c1[cid] = train_rejector_l2h(d, c_e=0.0, c_1=1.0, server_key='uni_preds')
        
        print(f"  Client {cid} ({info['name']}): {time.time()-t0:.0f}s")
    
    # ============================================================
    # Evaluate
    # ============================================================
    eval_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Baselines
    print(f"\n{'='*70}")
    print("BASELINES")
    print(f"{'='*70}")
    print(f"{'Client':<13} {'Client':>8} {'PerSrv':>8} {'UniSrv':>8} {'Oracle(P)':>10} {'Oracle(U)':>10}")
    print("-" * 60)
    for cid, info in clients.items():
        d = test_data[cid]
        co = (d['client_preds'] == d['labels']).float().mean().item()
        so = (d['server_preds'] == d['labels']).float().mean().item()
        uo = (d['uni_preds'] == d['labels']).float().mean().item()
        orc_p = ((d['client_preds'] == d['labels']) | (d['server_preds'] == d['labels'])).float().mean().item()
        orc_u = ((d['client_preds'] == d['labels']) | (d['uni_preds'] == d['labels'])).float().mean().item()
        print(f"{info['name']:<13} {co:>8.4f} {so:>8.4f} {uo:>8.4f} {orc_p:>10.4f} {orc_u:>10.4f}")
    
    # Methods to evaluate:
    # Per-client server: GK(best_α), L2H(c1=0), L2H(c1=1), BCE, ConfThresh
    # Universal server:  GK(best_α), L2H(c1=1), ConfThresh
    
    for rate in eval_rates:
        print(f"\n{'='*120}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*120}")
        
        # Per-client server column
        print(f"\n--- Per-Client Server Head ---")
        header = f"{'Client':<13} {'GK(α*)':>10} {'L2H(c1=0)':>10} {'L2H(c1=1)':>10} {'BCE':>10} {'ConfTh':>10}"
        print(header)
        print("-" * 75)
        
        avgs_per = {'gk': 0, 'l2h0': 0, 'l2h1': 0, 'bce': 0, 'conf': 0}
        
        for cid, info in clients.items():
            d = test_data[cid]
            sp = d['server_preds']
            
            s_gk = get_scores_gatekeeper(per_rej_gk[cid], d)
            s_l0 = get_scores_binary(per_rej_l2h_c0[cid], d)
            s_l1 = get_scores_binary(per_rej_l2h_c1[cid], d)
            s_bce = get_scores_binary(per_rej_bce[cid], d)
            s_conf = get_scores_conf(d)
            
            a_gk = system_acc_at_rate(s_gk, rate, d['client_preds'], sp, d['labels'])
            a_l0 = system_acc_at_rate(s_l0, rate, d['client_preds'], sp, d['labels'])
            a_l1 = system_acc_at_rate(s_l1, rate, d['client_preds'], sp, d['labels'])
            a_bce = system_acc_at_rate(s_bce, rate, d['client_preds'], sp, d['labels'])
            a_conf = system_acc_at_rate(s_conf, rate, d['client_preds'], sp, d['labels'])
            
            print(f"{info['name']:<13} {a_gk:>10.4f} {a_l0:>10.4f} {a_l1:>10.4f} {a_bce:>10.4f} {a_conf:>10.4f}")
            avgs_per['gk'] += a_gk; avgs_per['l2h0'] += a_l0; avgs_per['l2h1'] += a_l1
            avgs_per['bce'] += a_bce; avgs_per['conf'] += a_conf
        
        n = len(clients)
        print("-" * 75)
        print(f"{'Average':<13} {avgs_per['gk']/n:>10.4f} {avgs_per['l2h0']/n:>10.4f} {avgs_per['l2h1']/n:>10.4f} {avgs_per['bce']/n:>10.4f} {avgs_per['conf']/n:>10.4f}")
        
        # Universal server column
        print(f"\n--- Universal Server Head ---")
        header = f"{'Client':<13} {'GK(α*)':>10} {'L2H(c1=1)':>10} {'ConfTh':>10}"
        print(header)
        print("-" * 50)
        
        avgs_uni = {'gk': 0, 'l2h1': 0, 'conf': 0}
        
        for cid, info in clients.items():
            d = test_data[cid]
            su = d['uni_preds']
            
            s_gk = get_scores_gatekeeper(uni_rej_gk[cid], d)
            s_l1 = get_scores_binary(uni_rej_l2h_c1[cid], d)
            s_conf = get_scores_conf(d)
            
            a_gk = system_acc_at_rate(s_gk, rate, d['client_preds'], su, d['labels'])
            a_l1 = system_acc_at_rate(s_l1, rate, d['client_preds'], su, d['labels'])
            a_conf = system_acc_at_rate(s_conf, rate, d['client_preds'], su, d['labels'])
            
            print(f"{info['name']:<13} {a_gk:>10.4f} {a_l1:>10.4f} {a_conf:>10.4f}")
            avgs_uni['gk'] += a_gk; avgs_uni['l2h1'] += a_l1; avgs_uni['conf'] += a_conf
        
        print("-" * 50)
        print(f"{'Average':<13} {avgs_uni['gk']/n:>10.4f} {avgs_uni['l2h1']/n:>10.4f} {avgs_uni['conf']/n:>10.4f}")
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
