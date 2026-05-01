"""
Two-Stage L2H v3: Loss Function Comparison
===========================================
commit: TBD

Fixed setup:
- Client: AlexNet (trained 50ep, frozen)
- Server: ResNet-50 pretrained (frozen) + linear head (500ep)
- Rejector: LeNet backbone on raw x + client hidden features (AlexNet fc1, 256-dim)

Loss functions compared:
1. L2H Multi-class (our Ch3 paper) — stage-switching surrogate L2
2. Gatekeeper — α·CE(correct) + (1-α)·KL(incorrect||U)
3. Mozannar & Sontag cost-sensitive (L2D→L2H adaptation)
4. OvA (L2D→L2H adaptation)
5. BCE baseline (client incorrect label)

All evaluated at fixed deferral rates.
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
    # Eval
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

# ============================================================
# Rejector: LeNet on raw x + client hidden features
# ============================================================
class RejectorLeNet(nn.Module):
    """LeNet processes raw 32x32 image, then concatenates with client hidden features."""
    def __init__(self, hidden_dim=256, out_dim=1):
        super().__init__()
        self.lenet = nn.Sequential(
            nn.Conv2d(3, 6, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
        )
        # LeNet features (400) + client hidden (256) = 656
        self.head = nn.Sequential(
            nn.Linear(400 + hidden_dim, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(),
            nn.Linear(84, out_dim),
        )
    
    def forward(self, x, hidden):
        lf = self.lenet(x)
        combined = torch.cat([lf, hidden], dim=1)
        return self.head(combined)

# ============================================================
# Precompute all data needed for rejector training
# ============================================================
def precompute_rejector_data(client_model, server_head, server_backbone, 
                             dataset_32, dataset_224, n_classes):
    """Precompute everything needed for rejector training."""
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
    with torch.no_grad():
        for (x224, _) in loader_224:
            sf = server_backbone(x224.to(device))
            sl = server_head(sf)
            all_server_logits.append(sl.cpu())
            all_server_preds.append(sl.argmax(1).cpu())
    
    return {
        'x32': torch.cat(all_x32),
        'hidden': torch.cat(all_hidden),
        'labels': torch.cat(all_labels),
        'client_logits': torch.cat(all_client_logits),
        'client_preds': torch.cat(all_client_preds),
        'server_logits': torch.cat(all_server_logits),
        'server_preds': torch.cat(all_server_preds),
        'n_classes': n_classes,
    }

# ============================================================
# Loss Functions
# ============================================================

def train_rejector_bce(data, epochs=REJ_EPOCHS):
    """Loss 0: BCE baseline — label = client incorrect"""
    rej = RejectorLeNet(CLIENT_FEAT_DIM, out_dim=1).to(device)
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

def train_rejector_l2h(data, epochs=REJ_EPOCHS, c_e=0.0, c_1=0.0):
    """Loss 1: L2H Multi-class surrogate L2 from our Ch3 paper.
    L2 = -(1-c_e-c_1+c_1*𝟙[e(x)=y]) * log σ(r₂) - 𝟙[m(x)=y] * log σ(r₁)
    Here r outputs 2 logits: r₁ (local/client) and r₂ (remote/server).
    """
    rej = RejectorLeNet(CLIENT_FEAT_DIM, out_dim=2).to(device)
    
    # Precompute indicators
    server_correct = (data['server_preds'] == data['labels']).float()
    client_correct = (data['client_preds'] == data['labels']).float()
    
    dataset = TensorDataset(data['x32'], data['hidden'], client_correct, server_correct)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    rej.train()
    for _ in range(epochs):
        for x, h, mc, ec in loader:
            x, h, mc, ec = x.to(device), h.to(device), mc.to(device), ec.to(device)
            logits = rej(x, h)  # (B, 2): [r1(local), r2(remote)]
            log_probs = F.log_softmax(logits, dim=1)  # log σ
            # L2H surrogate: weight for remote = (1-c_e-c_1+c_1*server_correct)
            w_remote = 1 - c_e - c_1 + c_1 * ec
            loss = -(w_remote * log_probs[:, 1] + mc * log_probs[:, 0]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def train_rejector_gatekeeper(data, epochs=REJ_EPOCHS, alpha=0.3):
    """Loss 2: Gatekeeper — finetune rejector to output K classes.
    Correct samples: CE loss. Incorrect samples: KL to uniform.
    Since we can't finetune client, we train a separate classifier head on rejector features.
    """
    n_classes = data['n_classes']
    
    # Rejector outputs K classes (not binary)
    rej = RejectorLeNet(CLIENT_FEAT_DIM, out_dim=n_classes).to(device)
    
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
            probs = F.softmax(logits, dim=1)
            
            # CE for correct samples
            ce = F.cross_entropy(logits, y, reduction='none')
            l_corr = (corr * ce).sum() / max(corr.sum(), 1)
            
            # KL to uniform for incorrect samples
            log_probs = F.log_softmax(logits, dim=1)
            kl = F.kl_div(log_probs, uniform.expand_as(log_probs), reduction='none').sum(1)
            l_incorr = (incorr * kl).sum() / max(incorr.sum(), 1)
            
            loss = alpha * l_corr + (1 - alpha) * l_incorr
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def train_rejector_mozannar(data, epochs=REJ_EPOCHS, alpha_cost=1.0):
    """Loss 3: Mozannar & Sontag cost-sensitive (L2D→L2H adaptation).
    Original L2D: fix expert, jointly train classifier+rejector.
    L2H adaptation: fix client, rejector decides defer to server.
    
    Output: K+1 classes (K for classification + 1 for defer).
    Loss: -m * log p(defer) - m2 * log p(y)
    where m = cost when NOT deferring = alpha * 𝟙[server≠y] + 𝟙[server=y]  (L2H: server is the expert)
          m2 = cost when deferring to server, but we use client prediction
    
    L2H adaptation: 
    - "expert" in L2D → "server" in L2H (the one we defer TO)
    - "classifier" in L2D → "client" in L2H (the local model)
    - defer class = send to server
    - m (defer cost weight) = α*𝟙[server≠y] + 𝟙[server=y]  → higher weight when server correct
    - m2 (local cost weight) = α*𝟙[client≠y] + 𝟙[client=y]
    """
    n_classes = data['n_classes']
    rej = RejectorLeNet(CLIENT_FEAT_DIM, out_dim=n_classes + 1).to(device)
    
    server_correct = (data['server_preds'] == data['labels']).float()
    client_correct = (data['client_preds'] == data['labels']).float()
    
    # m: weight for defer class (higher when server is correct → encourage deferral)
    m = alpha_cost * (1 - server_correct) + server_correct
    # m2: weight for keeping local (higher when client is correct)
    m2 = alpha_cost * (1 - client_correct) + client_correct
    
    dataset = TensorDataset(data['x32'], data['hidden'], data['labels'], m, m2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    rej.train()
    for _ in range(epochs):
        for x, h, y, m_batch, m2_batch in loader:
            x, h, y = x.to(device), h.to(device), y.to(device)
            m_batch, m2_batch = m_batch.to(device), m2_batch.to(device)
            
            logits = rej(x, h)  # (B, K+1)
            log_probs = F.log_softmax(logits, dim=1)
            
            defer_idx = n_classes  # last class = defer
            loss = -(m_batch * log_probs[:, defer_idx] + m2_batch * log_probs[range(len(y)), y]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

def train_rejector_ova(data, epochs=REJ_EPOCHS, alpha_cost=1.0):
    """Loss 4: OvA L2D→L2H adaptation.
    One-vs-All consistent surrogate with logistic loss.
    
    Output: K+1 logits (K classes + 1 defer).
    OvA loss per sample:
      l1 = LogLoss(f_y, +1)                    # correct class
      l2 = Σ_{j≠y} LogLoss(f_j, -1)            # other classes
      l3 = LogLoss(f_{K+1}, -1)                # defer logit, negative
      l4 = LogLoss(f_{K+1}, +1)                # defer logit, positive
      L = m2 * (l1 + l2) + l3 + m * (l4 - l3)
    
    L2H adaptation: m = server correctness weight, m2 = client correctness weight
    """
    n_classes = data['n_classes']
    rej = RejectorLeNet(CLIENT_FEAT_DIM, out_dim=n_classes + 1).to(device)
    
    server_correct = (data['server_preds'] == data['labels']).float()
    client_correct = (data['client_preds'] == data['labels']).float()
    m = alpha_cost * (1 - server_correct) + server_correct
    m2 = alpha_cost * (1 - client_correct) + client_correct
    
    dataset = TensorDataset(data['x32'], data['hidden'], data['labels'], m, m2)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    opt = optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    
    def logistic_loss(z, y_sign):
        """log2(1 + exp(-y*z)), clamped for stability"""
        return torch.log2(1 + torch.exp(torch.clamp(-y_sign * z, max=50)))
    
    rej.train()
    for _ in range(epochs):
        for x, h, y, m_batch, m2_batch in loader:
            x, h, y = x.to(device), h.to(device), y.to(device)
            m_batch, m2_batch = m_batch.to(device), m2_batch.to(device)
            
            logits = rej(x, h)  # (B, K+1)
            B = logits.shape[0]
            
            # l1: logistic loss for correct class (positive)
            l1 = logistic_loss(logits[range(B), y], 1)
            
            # l2: sum of logistic loss for wrong classes (negative)
            mask = torch.ones_like(logits[:, :n_classes])
            mask[range(B), y] = 0
            l2 = (logistic_loss(logits[:, :n_classes], -1) * mask).sum(1)
            
            # l3, l4: defer logit
            l3 = logistic_loss(logits[:, n_classes], -1)
            l4 = logistic_loss(logits[:, n_classes], 1)
            
            loss = (m2_batch * (l1 + l2) + l3 + m_batch * (l4 - l3)).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rej.eval()
    return rej

# ============================================================
# Get deferral scores from different rejector types
# ============================================================
def get_scores_binary(rejector, data):
    """For BCE and L2H rejectors with 1 or 2 outputs."""
    rejector.eval()
    loader = DataLoader(TensorDataset(data['x32'], data['hidden']), batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for x, h in loader:
            out = rejector(x.to(device), h.to(device))
            if out.shape[1] == 2:
                # L2H: score = log p(remote) - log p(local) = r2 - r1
                scores.append((out[:, 1] - out[:, 0]).cpu())
            else:
                scores.append(out.squeeze().cpu())
    return torch.cat(scores)

def get_scores_gatekeeper(rejector, data):
    """Gatekeeper: defer when confidence is LOW → score = negative max softmax."""
    rejector.eval()
    loader = DataLoader(TensorDataset(data['x32'], data['hidden']), batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for x, h in loader:
            out = rejector(x.to(device), h.to(device))
            conf = F.softmax(out, dim=1).max(1).values
            scores.append(-conf.cpu())  # lower confidence → higher defer score
    return torch.cat(scores)

def get_scores_kplus1(rejector, data, n_classes):
    """For Mozannar and OvA: defer class is index n_classes.
    Score = logit of defer class."""
    rejector.eval()
    loader = DataLoader(TensorDataset(data['x32'], data['hidden']), batch_size=BATCH_SIZE)
    scores = []
    with torch.no_grad():
        for x, h in loader:
            out = rejector(x.to(device), h.to(device))
            scores.append(out[:, n_classes].cpu())  # defer logit
    return torch.cat(scores)

def get_scores_conf(data):
    """ConfThresh baseline: negative client confidence."""
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
    print("=" * 80)
    print("L2H v3: Loss Function Comparison")
    print("Rejector: LeNet + client hidden features")
    print("=" * 80)
    
    print("\n[1/8] Loading data...")
    clients = get_client_data()
    
    print(f"\n[2/8] Training client AlexNet models ({CLIENT_EPOCHS}ep)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client(info['n_classes'], info['train_32'], info['test_32'])
        client_models[cid] = model
        print(f"  Client {cid} ({info['name']}): acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    print("\n[3/8] Loading server backbone...")
    server_bb = make_server_backbone()
    
    print("\n[4/8] Extracting server features...")
    srv_tr_feats, srv_tr_labels = {}, {}
    srv_te_feats, srv_te_labels = {}, {}
    for cid, info in clients.items():
        srv_tr_feats[cid], srv_tr_labels[cid] = extract_features(server_bb, info['train_224'])
        srv_te_feats[cid], srv_te_labels[cid] = extract_features(server_bb, info['test_224'])
        print(f"  Client {cid}: done")
    
    print(f"\n[5/8] Training per-client server heads ({HEAD_EPOCHS}ep)...")
    per_heads = {}
    for cid, info in clients.items():
        per_heads[cid] = train_head(srv_tr_feats[cid], srv_tr_labels[cid], info['n_classes'])
        with torch.no_grad():
            acc = (per_heads[cid](srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        print(f"  {info['name']}: acc={acc:.4f}")
    
    print("\n[6/8] Precomputing rejector data...")
    train_data, test_data = {}, {}
    for cid, info in clients.items():
        train_data[cid] = precompute_rejector_data(
            client_models[cid], per_heads[cid], server_bb,
            info['train_32'], info['train_224'], info['n_classes'])
        test_data[cid] = precompute_rejector_data(
            client_models[cid], per_heads[cid], server_bb,
            info['test_32'], info['test_224'], info['n_classes'])
        c_acc = (train_data[cid]['client_preds'] == train_data[cid]['labels']).float().mean().item()
        s_acc = (train_data[cid]['server_preds'] == train_data[cid]['labels']).float().mean().item()
        print(f"  Client {cid}: client_train_acc={c_acc:.3f}, server_train_acc={s_acc:.3f}")
    
    print(f"\n[7/8] Training rejectors with 5 loss functions ({REJ_EPOCHS}ep)...")
    
    loss_names = ['BCE', 'L2H-Multi', 'Gatekeeper', 'Mozannar', 'OvA']
    rejectors = {name: {} for name in loss_names}
    
    for cid, info in clients.items():
        d = train_data[cid]
        print(f"\n  Client {cid} ({info['name']}):")
        
        t0 = time.time()
        rejectors['BCE'][cid] = train_rejector_bce(d)
        print(f"    BCE: {time.time()-t0:.0f}s")
        
        t0 = time.time()
        rejectors['L2H-Multi'][cid] = train_rejector_l2h(d)
        print(f"    L2H-Multi: {time.time()-t0:.0f}s")
        
        t0 = time.time()
        rejectors['Gatekeeper'][cid] = train_rejector_gatekeeper(d)
        print(f"    Gatekeeper: {time.time()-t0:.0f}s")
        
        t0 = time.time()
        rejectors['Mozannar'][cid] = train_rejector_mozannar(d)
        print(f"    Mozannar: {time.time()-t0:.0f}s")
        
        t0 = time.time()
        rejectors['OvA'][cid] = train_rejector_ova(d)
        print(f"    OvA: {time.time()-t0:.0f}s")
    
    # ============================================================
    # Evaluate
    # ============================================================
    print(f"\n[8/8] Evaluating at fixed deferral rates...")
    
    target_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Baselines
    print(f"\n{'='*60}")
    print("BASELINES")
    print(f"{'='*60}")
    print(f"{'Client':<13} {'Client':>8} {'Server':>8} {'Oracle':>8}")
    print("-" * 40)
    for cid, info in clients.items():
        d = test_data[cid]
        co = (d['client_preds'] == d['labels']).float().mean().item()
        so = (d['server_preds'] == d['labels']).float().mean().item()
        orc = ((d['client_preds'] == d['labels']) | (d['server_preds'] == d['labels'])).float().mean().item()
        print(f"{info['name']:<13} {co:>8.4f} {so:>8.4f} {orc:>8.4f}")
    
    # Per-rate comparison
    for rate in target_rates:
        print(f"\n{'='*80}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*80}")
        header = f"{'Client':<13}"
        for name in loss_names:
            header += f" {name:>10}"
        header += f" {'ConfThresh':>10}"
        print(header)
        print("-" * 80)
        
        avgs = {name: 0 for name in loss_names + ['ConfThresh']}
        
        for cid, info in clients.items():
            d = test_data[cid]
            nc = info['n_classes']
            results = {}
            
            # BCE
            scores = get_scores_binary(rejectors['BCE'][cid], d)
            results['BCE'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            # L2H-Multi
            scores = get_scores_binary(rejectors['L2H-Multi'][cid], d)
            results['L2H-Multi'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            # Gatekeeper
            scores = get_scores_gatekeeper(rejectors['Gatekeeper'][cid], d)
            results['Gatekeeper'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            # Mozannar
            scores = get_scores_kplus1(rejectors['Mozannar'][cid], d, nc)
            results['Mozannar'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            # OvA
            scores = get_scores_kplus1(rejectors['OvA'][cid], d, nc)
            results['OvA'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            # ConfThresh
            scores = get_scores_conf(d)
            results['ConfThresh'] = system_acc_at_rate(scores, rate, d['client_preds'], d['server_preds'], d['labels'])
            
            row = f"{info['name']:<13}"
            for name in loss_names:
                row += f" {results[name]:>10.4f}"
            row += f" {results['ConfThresh']:>10.4f}"
            print(row)
            
            for name in results:
                avgs[name] += results[name]
        
        n = len(clients)
        row = f"{'Average':<13}"
        for name in loss_names:
            row += f" {avgs[name]/n:>10.4f}"
        row += f" {avgs['ConfThresh']/n:>10.4f}"
        print("-" * 80)
        print(row)
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
