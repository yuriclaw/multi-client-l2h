"""
Two-Stage L2H v6: ViT Server Backbone
=======================================
Same as v5 but server backbone changed from ResNet-50 to ViT-B/16.
ViT-B/16 outputs 768-dim features (vs ResNet-50's 2048-dim).
Server head: Linear(768 → n_classes).
All other settings identical to v5.
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
import copy

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

N_TRAIN = 8000
N_TEST = 2000
DATA_DIR = '/tmp/data'
CLIENT_EPOCHS = 50
HEAD_EPOCHS = 500
STAGE2_EPOCHS = 100
BATCH_SIZE = 128

INET_MEAN = (0.485, 0.456, 0.406)
INET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# ============================================================
# Data (same as v4)
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
# Client: AlexNet
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
        self.fc2 = nn.Linear(256, n_classes)
    
    def forward(self, x):
        f = self.features(x)
        h = self.fc1(f)
        return self.fc2(h)
    
    def get_hidden(self, x):
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
# Server
# ============================================================
def make_server_backbone():
    bb = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0).to(device)
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
    opt = optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)  # same LR as ResNet version
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
    offsets = {}
    total_classes = 0
    for cid in sorted(clients.keys()):
        offsets[cid] = total_classes
        total_classes += clients[cid]['n_classes']
    
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
# Rejector: AlexNet-style (for L2H method)
# ============================================================
class RejectorAlexNet(nn.Module):
    def __init__(self, hidden_dim=256, out_dim=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 96, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(96, 192, 3, padding=1), nn.ReLU(),
            nn.Conv2d(192, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(128 + hidden_dim, 256), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(256, out_dim),
        )
    
    def forward(self, x, hidden):
        img_feat = self.features(x)
        return self.head(torch.cat([img_feat, hidden], dim=1))

# ============================================================
# METHOD 1: L2H Multi-class (separate rejector, jointly train with server head)
# ============================================================
def train_l2h_joint(client_model, server_head_init, server_bb,
                    train_32, train_224, n_classes, c_1=1.0):
    """L2H surrogate L2 with JOINT training of rejector + server head.
    Returns (rejector, updated_server_head).
    """
    rejector = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=2).to(device)
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    # Precompute server features (backbone frozen)
    srv_feats, labels = extract_features(server_bb, train_224)
    
    # Precompute client hidden + preds
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_hidden, all_labels, all_client_correct = [], [], [], []
    with torch.no_grad():
        for x, y in loader_32:
            x_d = x.to(device)
            cp = client_model(x_d).argmax(1).cpu()
            h = client_model.get_hidden(x_d).cpu()
            all_x32.append(x); all_hidden.append(h); all_labels.append(y)
            all_client_correct.append((cp == y).float())
    
    all_x32 = torch.cat(all_x32)
    all_hidden = torch.cat(all_hidden)
    all_labels = torch.cat(all_labels)
    all_client_correct = torch.cat(all_client_correct)
    
    dataset = TensorDataset(all_x32, all_hidden, all_client_correct, srv_feats, all_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Jointly optimize rejector + server head
    opt = optim.Adam(list(rejector.parameters()) + list(server_head.parameters()), 
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    rejector.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, hidden, mc, sf, y in loader:
            x32, hidden, mc = x32.to(device), hidden.to(device), mc.to(device)
            sf, y = sf.to(device), y.to(device)
            
            # Server head prediction (live, not precomputed)
            server_logits = server_head(sf)
            server_correct = (server_logits.argmax(1) == y).float()
            
            # Rejector logits
            rej_logits = rejector(x32, hidden)
            log_probs = F.log_softmax(rej_logits, dim=1)
            
            # L2H surrogate L2: c_e=0
            w_remote = 1 - c_1 + c_1 * server_correct
            rej_loss = -(w_remote * log_probs[:, 1] + mc * log_probs[:, 0]).mean()
            
            # Server head CE loss (helps server adapt to deferred samples)
            server_ce = F.cross_entropy(server_logits, y)
            
            loss = rej_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    
    rejector.eval(); server_head.eval()
    return rejector, server_head

# ============================================================
# METHOD 2: Gatekeeper (finetune client fc2, jointly train server head)
# ============================================================
def train_gatekeeper_joint(client_model, server_head_init, server_bb,
                           train_32, train_224, n_classes, alpha=0.8):
    """Finetune client's fc2 with Gatekeeper loss + jointly update server head.
    Returns (finetuned_client, updated_server_head).
    """
    # Deep copy client, unfreeze only fc2
    client_ft = copy.deepcopy(client_model).to(device)
    for p in client_ft.parameters(): p.requires_grad = False
    for p in client_ft.fc2.parameters(): p.requires_grad = True
    
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    # Precompute server features
    srv_feats, labels = extract_features(server_bb, train_224)
    
    # Build dataset with raw images for client forward pass
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_labels = [], []
    for x, y in loader_32:
        all_x32.append(x); all_labels.append(y)
    all_x32 = torch.cat(all_x32)
    all_labels = torch.cat(all_labels)
    
    dataset = TensorDataset(all_x32, srv_feats, all_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    uniform = torch.ones(n_classes).to(device) / n_classes
    
    opt = optim.Adam(list(client_ft.fc2.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    client_ft.train()  # Need train mode for dropout, but only fc2 has grad
    for _ in range(STAGE2_EPOCHS):
        for x32, sf, y in loader:
            x32, sf, y = x32.to(device), sf.to(device), y.to(device)
            
            # Client forward (only fc2 gets gradient)
            client_logits = client_ft(x32)
            client_pred = client_logits.argmax(1).detach()
            
            corr = (client_pred == y).float()
            incorr = 1 - corr
            
            # Gatekeeper loss on client
            ce = F.cross_entropy(client_logits, y, reduction='none')
            l_corr = (corr * ce).sum() / max(corr.sum(), 1)
            
            log_probs = F.log_softmax(client_logits, dim=1)
            kl = F.kl_div(log_probs, uniform.expand_as(log_probs), reduction='none').sum(1)
            l_incorr = (incorr * kl).sum() / max(incorr.sum(), 1)
            
            gk_loss = alpha * l_corr + (1 - alpha) * l_incorr
            
            # Server head CE loss
            server_logits = server_head(sf)
            server_ce = F.cross_entropy(server_logits, y)
            
            loss = gk_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    
    client_ft.eval(); server_head.eval()
    return client_ft, server_head

# ============================================================
# METHOD 3: Mozannar cost-sensitive (extend client to K+1, jointly train server)
# ============================================================
def train_mozannar_joint(client_model, server_head_init, server_bb,
                         train_32, train_224, n_classes, alpha_cost=1.0):
    """Extend client fc2 to K+1, jointly train with server head.
    Returns (extended_client, updated_server_head).
    """
    # Deep copy client, replace fc2 with K+1 output
    client_ext = copy.deepcopy(client_model).to(device)
    for p in client_ext.parameters(): p.requires_grad = False
    
    # New fc2: K+1 outputs. Initialize first K weights from original
    old_fc2 = client_ext.fc2
    new_fc2 = nn.Linear(256, n_classes + 1).to(device)
    with torch.no_grad():
        new_fc2.weight[:n_classes] = old_fc2.weight
        new_fc2.bias[:n_classes] = old_fc2.bias
        nn.init.zeros_(new_fc2.weight[n_classes])
        nn.init.zeros_(new_fc2.bias[n_classes])
    client_ext.fc2 = new_fc2
    # Only new fc2 is trainable
    
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    srv_feats, labels = extract_features(server_bb, train_224)
    
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_labels = [], []
    for x, y in loader_32:
        all_x32.append(x); all_labels.append(y)
    all_x32 = torch.cat(all_x32); all_labels = torch.cat(all_labels)
    
    dataset = TensorDataset(all_x32, srv_feats, all_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    opt = optim.Adam(list(new_fc2.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    client_ext.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, sf, y in loader:
            x32, sf, y = x32.to(device), sf.to(device), y.to(device)
            
            outputs = client_ext(x32)  # (B, K+1)
            
            # Server prediction for cost weighting
            server_logits = server_head(sf)
            server_pred = server_logits.argmax(1).detach()
            server_correct = (server_pred == y).float()
            
            # Mozannar cost-sensitive: m=weight for defer, m2=weight for classify
            # In L2H: "expert" = server (we defer TO server)
            # m = 1 when server correct, alpha when server wrong
            m = alpha_cost * (1 - server_correct) + server_correct
            # m2 = weight for correct classification (always active)
            # Using alpha when NOT the correct class situation
            m2 = torch.ones_like(m)  # simplified: always weight 1 for classification
            
            # Softmax loss: -m * log p(defer) - m2 * log p(y)
            log_probs = F.log_softmax(outputs, dim=1)
            loss_defer = -m * log_probs[:, n_classes]
            loss_class = -m2 * log_probs[range(len(y)), y]
            mozannar_loss = (loss_defer + loss_class).mean()
            
            # Server CE
            server_ce = F.cross_entropy(server_logits, y)
            
            loss = mozannar_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    
    client_ext.eval(); server_head.eval()
    return client_ext, server_head

# ============================================================
# METHOD 4: OvA (extend client to K+1, logistic loss, jointly train server)
# ============================================================
def train_ova_joint(client_model, server_head_init, server_bb,
                    train_32, train_224, n_classes, alpha_cost=1.0):
    """OvA L2D→L2H: extend client to K+1, logistic loss, joint server training."""
    client_ext = copy.deepcopy(client_model).to(device)
    for p in client_ext.parameters(): p.requires_grad = False
    
    old_fc2 = client_ext.fc2
    new_fc2 = nn.Linear(256, n_classes + 1).to(device)
    with torch.no_grad():
        new_fc2.weight[:n_classes] = old_fc2.weight
        new_fc2.bias[:n_classes] = old_fc2.bias
        nn.init.zeros_(new_fc2.weight[n_classes])
        nn.init.zeros_(new_fc2.bias[n_classes])
    client_ext.fc2 = new_fc2
    
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    srv_feats, labels = extract_features(server_bb, train_224)
    
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_labels = [], []
    for x, y in loader_32:
        all_x32.append(x); all_labels.append(y)
    all_x32 = torch.cat(all_x32); all_labels = torch.cat(all_labels)
    
    dataset = TensorDataset(all_x32, srv_feats, all_labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    def logistic_loss(z, y_sign):
        return torch.log2(1 + torch.exp(torch.clamp(-y_sign * z, max=50)))
    
    opt = optim.Adam(list(new_fc2.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    client_ext.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, sf, y in loader:
            x32, sf, y = x32.to(device), sf.to(device), y.to(device)
            B = x32.shape[0]
            
            logits = client_ext(x32)  # (B, K+1)
            
            server_logits = server_head(sf)
            server_pred = server_logits.argmax(1).detach()
            server_correct = (server_pred == y).float()
            
            m = alpha_cost * (1 - server_correct) + server_correct
            m2 = torch.ones_like(m)
            
            # OvA loss
            l1 = logistic_loss(logits[range(B), y], 1)
            mask = torch.ones_like(logits[:, :n_classes])
            mask[range(B), y] = 0
            l2 = (logistic_loss(logits[:, :n_classes], -1) * mask).sum(1)
            l3 = logistic_loss(logits[:, n_classes], -1)
            l4 = logistic_loss(logits[:, n_classes], 1)
            
            ova_loss = (m2 * (l1 + l2) + l3 + m * (l4 - l3)).mean()
            
            server_ce = F.cross_entropy(server_logits, y)
            
            loss = ova_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    
    client_ext.eval(); server_head.eval()
    return client_ext, server_head

# ============================================================
# Score & Eval
# ============================================================
def get_scores_l2h(rejector, client_model, dataset_32):
    """L2H: score = r2 - r1 from separate rejector."""
    rejector.eval(); client_model.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x_d = x.to(device)
            h = client_model.get_hidden(x_d)
            out = rejector(x_d, h)
            scores.append((out[:, 1] - out[:, 0]).cpu())
    return torch.cat(scores)

def get_scores_gatekeeper(client_ft, dataset_32):
    """Gatekeeper: score = negative max softmax (low confidence → defer)."""
    client_ft.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            out = client_ft(x.to(device))
            conf = F.softmax(out, dim=1).max(1).values
            scores.append(-conf.cpu())
    return torch.cat(scores)

def get_scores_kplus1(client_ext, dataset_32, n_classes):
    """Mozannar/OvA: score = defer logit (class K+1)."""
    client_ext.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            out = client_ext(x.to(device))
            scores.append(out[:, n_classes].cpu())
    return torch.cat(scores)

def get_scores_conf(client_model, dataset_32):
    """ConfThresh baseline: negative original client confidence."""
    client_model.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            out = client_model(x.to(device))
            conf = F.softmax(out, dim=1).max(1).values
            scores.append(-conf.cpu())
    return torch.cat(scores)

def get_client_preds(client_model, dataset_32):
    client_model.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(client_model(x.to(device)).argmax(1).cpu())
            labels.append(y)
    return torch.cat(preds), torch.cat(labels)

def get_kplus1_client_preds(client_ext, dataset_32, n_classes):
    """For K+1 models, client prediction is argmax over first K classes."""
    client_ext.eval()
    loader = DataLoader(dataset_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            out = client_ext(x.to(device))
            preds.append(out[:, :n_classes].argmax(1).cpu())
            labels.append(y)
    return torch.cat(preds), torch.cat(labels)

def get_server_preds(server_head, server_bb, dataset_224):
    server_head.eval()
    loader = DataLoader(dataset_224, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            sf = server_bb(x.to(device))
            preds.append(server_head(sf).argmax(1).cpu())
    return torch.cat(preds)

def system_acc_at_rate(scores, target_rate, client_preds, server_preds, labels):
    n = len(scores)
    k = max(1, int(n * target_rate))
    _, top_idx = torch.topk(scores, k)
    defer = torch.zeros(n, dtype=torch.bool)
    defer[top_idx] = True
    system_preds = torch.where(defer, server_preds, client_preds)
    return (system_preds == labels).float().mean().item()

def random_defer_acc(target_rate, client_preds, server_preds, labels, n_trials=50):
    """Random defer baseline: randomly select target_rate fraction to defer, average over n_trials."""
    n = len(labels)
    k = max(1, int(n * target_rate))
    accs = []
    for t in range(n_trials):
        idx = torch.randperm(n)[:k]
        defer = torch.zeros(n, dtype=torch.bool)
        defer[idx] = True
        system_preds = torch.where(defer, server_preds, client_preds)
        accs.append((system_preds == labels).float().mean().item())
    return np.mean(accs)

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 90)
    print("L2H v6: ViT-B/16 Server Backbone + Joint Training")
    print("=" * 90)
    
    print("\n[1/7] Loading data...")
    clients = get_client_data()
    
    print(f"\n[2/7] Training client AlexNet models ({CLIENT_EPOCHS}ep)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client(info['n_classes'], info['train_32'], info['test_32'])
        client_models[cid] = model
        print(f"  Client {cid} ({info['name']}): acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    print("\n[3/7] Loading server backbone...")
    server_bb = make_server_backbone()
    
    print("\n[4/7] Extracting features & training Stage 1 server heads...")
    srv_tr_feats, srv_tr_labels = {}, {}
    srv_te_feats, srv_te_labels = {}, {}
    per_heads_s1 = {}  # Stage 1 heads (before joint training)
    for cid, info in clients.items():
        srv_tr_feats[cid], srv_tr_labels[cid] = extract_features(server_bb, info['train_224'])
        srv_te_feats[cid], srv_te_labels[cid] = extract_features(server_bb, info['test_224'])
        per_heads_s1[cid] = train_head(srv_tr_feats[cid], srv_tr_labels[cid], info['n_classes'])
        with torch.no_grad():
            acc = (per_heads_s1[cid](srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        print(f"  {info['name']}: head_acc={acc:.4f}")
    
    print(f"\n[4b/7] Training universal server head ({HEAD_EPOCHS}ep)...")
    uni_head, offsets, total_classes = train_universal_head(srv_tr_feats, srv_tr_labels, clients)
    for cid, info in clients.items():
        with torch.no_grad():
            ul = uni_head(srv_te_feats[cid].to(device))
            off = offsets[cid]; nc = info['n_classes']
            acc = (ul[:, off:off+nc].argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        print(f"  {info['name']}: uni_acc={acc:.4f}")
    
    # ============================================================
    # Stage 2: Joint training per method
    # ============================================================
    print(f"\n[5/7] Stage 2: Joint training ({STAGE2_EPOCHS}ep)...")
    
    # Results storage: method -> cid -> (scores_fn, client_preds, server_preds, labels)
    methods = {}
    
    for cid, info in clients.items():
        nc = info['n_classes']
        t0 = time.time()
        print(f"\n  Client {cid} ({info['name']}):")
        
        # L2H (c1=1)
        t1 = time.time()
        rej_l2h, head_l2h = train_l2h_joint(
            client_models[cid], per_heads_s1[cid], server_bb,
            info['train_32'], info['train_224'], nc, c_1=1.0)
        methods.setdefault('L2H(c1=1)', {})[cid] = {
            'rejector': rej_l2h, 'server_head': head_l2h, 'type': 'l2h'}
        print(f"    L2H(c1=1): {time.time()-t1:.0f}s")
        
        # Gatekeeper (α=0.8)
        t1 = time.time()
        client_gk, head_gk = train_gatekeeper_joint(
            client_models[cid], per_heads_s1[cid], server_bb,
            info['train_32'], info['train_224'], nc, alpha=0.8)
        methods.setdefault('GK(α=0.8)', {})[cid] = {
            'client_ft': client_gk, 'server_head': head_gk, 'type': 'gk'}
        print(f"    GK(α=0.8): {time.time()-t1:.0f}s")
        
        # Mozannar
        t1 = time.time()
        client_moz, head_moz = train_mozannar_joint(
            client_models[cid], per_heads_s1[cid], server_bb,
            info['train_32'], info['train_224'], nc)
        methods.setdefault('Mozannar', {})[cid] = {
            'client_ext': client_moz, 'server_head': head_moz, 'type': 'kplus1'}
        print(f"    Mozannar: {time.time()-t1:.0f}s")
        
        # OvA
        t1 = time.time()
        client_ova, head_ova = train_ova_joint(
            client_models[cid], per_heads_s1[cid], server_bb,
            info['train_32'], info['train_224'], nc)
        methods.setdefault('OvA', {})[cid] = {
            'client_ext': client_ova, 'server_head': head_ova, 'type': 'kplus1'}
        print(f"    OvA: {time.time()-t1:.0f}s")
        
        print(f"    Total: {time.time()-t0:.0f}s")
    
    # ============================================================
    # Evaluate
    # ============================================================
    print(f"\n[6/7] Evaluating...")
    
    eval_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    method_names = ['L2H(c1=1)', 'GK(α=0.8)', 'Mozannar', 'OvA', 'ConfTh', 'ConfTh-U', 'Random']
    
    # Baselines
    print(f"\n{'='*70}")
    print("BASELINES (Stage 1 server heads)")
    print(f"{'='*70}")
    print(f"{'Client':<13} {'Client':>8} {'Server':>8} {'Oracle':>8}")
    print("-" * 40)
    for cid, info in clients.items():
        cp, labels = get_client_preds(client_models[cid], info['test_32'])
        sp = get_server_preds(per_heads_s1[cid], server_bb, info['test_224'])
        co = (cp == labels).float().mean().item()
        so = (sp == labels).float().mean().item()
        orc = ((cp == labels) | (sp == labels)).float().mean().item()
        print(f"{info['name']:<13} {co:>8.4f} {so:>8.4f} {orc:>8.4f}")
    
    # Server head accuracy after joint training
    print(f"\n{'='*70}")
    print("SERVER HEAD ACCURACY (after joint Stage 2 training)")
    print(f"{'='*70}")
    header = f"{'Client':<13} {'Stage1':>8}"
    trained_methods = ['L2H(c1=1)', 'GK(α=0.8)', 'Mozannar', 'OvA']
    for mn in trained_methods:
        header += f" {mn:>10}"
    print(header)
    print("-" * 70)
    for cid, info in clients.items():
        s1_acc = (per_heads_s1[cid](srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
        row = f"{info['name']:<13} {s1_acc:>8.4f}"
        for mn in trained_methods:
            m = methods[mn][cid]
            sh = m['server_head']
            with torch.no_grad():
                acc = (sh(srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
            row += f" {acc:>10.4f}"
        print(row)
    
    # Per-rate comparison
    for rate in eval_rates:
        print(f"\n{'='*90}")
        print(f"DEFERRAL RATE = {rate*100:.0f}%")
        print(f"{'='*90}")
        header = f"{'Client':<13}"
        for mn in method_names:
            header += f" {mn:>10}"
        print(header)
        print("-" * 80)
        
        avgs = {mn: 0 for mn in method_names}
        
        for cid, info in clients.items():
            nc = info['n_classes']
            results = {}
            
            # L2H
            m = methods['L2H(c1=1)'][cid]
            scores = get_scores_l2h(m['rejector'], client_models[cid], info['test_32'])
            cp, labels = get_client_preds(client_models[cid], info['test_32'])
            sp = get_server_preds(m['server_head'], server_bb, info['test_224'])
            results['L2H(c1=1)'] = system_acc_at_rate(scores, rate, cp, sp, labels)
            
            # Gatekeeper
            m = methods['GK(α=0.8)'][cid]
            scores = get_scores_gatekeeper(m['client_ft'], info['test_32'])
            # GK client preds come from finetuned client
            cp_gk, labels = get_client_preds(m['client_ft'], info['test_32'])
            sp_gk = get_server_preds(m['server_head'], server_bb, info['test_224'])
            results['GK(α=0.8)'] = system_acc_at_rate(scores, rate, cp_gk, sp_gk, labels)
            
            # Mozannar
            m = methods['Mozannar'][cid]
            scores = get_scores_kplus1(m['client_ext'], info['test_32'], nc)
            cp_moz, labels = get_kplus1_client_preds(m['client_ext'], info['test_32'], nc)
            sp_moz = get_server_preds(m['server_head'], server_bb, info['test_224'])
            results['Mozannar'] = system_acc_at_rate(scores, rate, cp_moz, sp_moz, labels)
            
            # OvA
            m = methods['OvA'][cid]
            scores = get_scores_kplus1(m['client_ext'], info['test_32'], nc)
            cp_ova, labels = get_kplus1_client_preds(m['client_ext'], info['test_32'], nc)
            sp_ova = get_server_preds(m['server_head'], server_bb, info['test_224'])
            results['OvA'] = system_acc_at_rate(scores, rate, cp_ova, sp_ova, labels)
            
            # ConfThresh (original client, Stage 1 server head)
            scores = get_scores_conf(client_models[cid], info['test_32'])
            cp_orig, labels = get_client_preds(client_models[cid], info['test_32'])
            sp_s1 = get_server_preds(per_heads_s1[cid], server_bb, info['test_224'])
            results['ConfTh'] = system_acc_at_rate(scores, rate, cp_orig, sp_s1, labels)
            
            # ConfThresh with Universal server head
            off = offsets[cid]; nc = info['n_classes']
            with torch.no_grad():
                ul = uni_head(srv_te_feats[cid].to(device))
                sp_uni = ul[:, off:off+nc].argmax(1).cpu()
            results['ConfTh-U'] = system_acc_at_rate(scores, rate, cp_orig, sp_uni, labels)
            
            # Random defer (average over 50 trials)
            results['Random'] = random_defer_acc(rate, cp_orig, sp_s1, labels, n_trials=50)
            
            row = f"{info['name']:<13}"
            for mn in method_names:
                row += f" {results[mn]:>10.4f}"
            print(row)
            
            for mn in results:
                avgs[mn] += results[mn]
        
        n = len(clients)
        print("-" * 80)
        row = f"{'Average':<13}"
        for mn in method_names:
            row += f" {avgs[mn]/n:>10.4f}"
        print(row)
    
    print(f"\n[7/7] Done! Total time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
