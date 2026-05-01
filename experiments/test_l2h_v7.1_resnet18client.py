"""
Two-Stage L2H v7.1: ResNet-18 Client + 3 Server Backbones
==========================================================
Server backbones: ResNet-50, DINOv2-B/14, CLIP-ViT-B/16
Datasets: CIFAR-100, Flowers102, OxfordIIITPet, Food101, DTD
Methods: L2H(c1=1), Gatekeeper(α=0.8), ConfThresh, Random
Client: ResNet-18 (CIFAR-style, 32×32), Rejector: small CNN + client hidden
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

N_TRAIN = 5000
N_TEST = 1000
DATA_DIR = '/tmp/data'
CLIENT_EPOCHS = 100
HEAD_EPOCHS = 200
STAGE2_EPOCHS = 100
BATCH_SIZE = 128

INET_MEAN = (0.485, 0.456, 0.406)
INET_STD = (0.229, 0.224, 0.225)

# ============================================================
# Data
# ============================================================
def get_transform_32():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(INET_MEAN, INET_STD),
    ])

def get_transform_224():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(INET_MEAN, INET_STD),
    ])

CLIENT_CONFIGS = {
    0: {'name': 'CIFAR-100', 'n_classes': 100},
    1: {'name': 'Flowers102', 'n_classes': 102},
    2: {'name': 'OxfordPet', 'n_classes': 37},
    3: {'name': 'Food101', 'n_classes': 101},
    4: {'name': 'DTD', 'n_classes': 47},
}

def _ensure_3ch(img):
    """Convert grayscale/RGBA to RGB."""
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

class EnsureRGB:
    def __call__(self, img):
        return _ensure_3ch(img)

def get_transform_32_rgb():
    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(INET_MEAN, INET_STD),
    ])

def get_transform_224_rgb():
    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(INET_MEAN, INET_STD),
    ])

def load_dataset_pair(name, tf32, tf224):
    """Load train and test datasets at both resolutions."""
    if name == 'CIFAR-100':
        tr32 = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tf32)
        te32 = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf32)
        tr224 = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tf224)
        te224 = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf224)
    elif name == 'Flowers102':
        tr32 = torchvision.datasets.Flowers102(DATA_DIR, split='train', download=True, transform=tf32)
        te32 = torchvision.datasets.Flowers102(DATA_DIR, split='test', download=True, transform=tf32)
        tr224 = torchvision.datasets.Flowers102(DATA_DIR, split='train', download=True, transform=tf224)
        te224 = torchvision.datasets.Flowers102(DATA_DIR, split='test', download=True, transform=tf224)
    elif name == 'OxfordPet':
        tr32 = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='trainval', download=True, transform=tf32)
        te32 = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='test', download=True, transform=tf32)
        tr224 = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='trainval', download=True, transform=tf224)
        te224 = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='test', download=True, transform=tf224)
    elif name == 'Food101':
        tr32 = torchvision.datasets.Food101(DATA_DIR, split='train', download=True, transform=tf32)
        te32 = torchvision.datasets.Food101(DATA_DIR, split='test', download=True, transform=tf32)
        tr224 = torchvision.datasets.Food101(DATA_DIR, split='train', download=True, transform=tf224)
        te224 = torchvision.datasets.Food101(DATA_DIR, split='test', download=True, transform=tf224)
    elif name == 'DTD':
        tr32 = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf32)
        te32 = torchvision.datasets.DTD(DATA_DIR, split='test', download=True, transform=tf32)
        tr224 = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf224)
        te224 = torchvision.datasets.DTD(DATA_DIR, split='test', download=True, transform=tf224)
    return tr32, te32, tr224, te224

def get_client_data():
    tf32 = get_transform_32_rgb()
    tf224 = get_transform_224_rgb()
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tr32, te32, tr224, te224 = load_dataset_pair(cfg['name'], tf32, tf224)
        # Subsample
        tr_n = min(N_TRAIN, len(tr32))
        te_n = min(N_TEST, len(te32))
        tr_idx = random.sample(range(len(tr32)), tr_n)
        te_idx = random.sample(range(len(te32)), te_n)
        clients[cid] = {
            'name': cfg['name'], 'n_classes': cfg['n_classes'],
            'train_32': Subset(tr32, tr_idx), 'test_32': Subset(te32, te_idx),
            'train_224': Subset(tr224, tr_idx), 'test_224': Subset(te224, te_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {tr_n} train, {te_n} test, {cfg['n_classes']} classes")
    return clients

# ============================================================
# Client ResNet-18 (CIFAR-style: conv1=3x3, no maxpool)
# ============================================================
CLIENT_FEAT_DIM = 512

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ClientResNet18(nn.Module):
    """CIFAR-style ResNet-18: conv1=3x3/s1, no maxpool, for 32x32 input."""
    def __init__(self, n_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
    
    def _make_layer(self, in_ch, out_ch, n_blocks, stride):
        layers = [BasicBlock(in_ch, out_ch, stride)]
        for _ in range(1, n_blocks):
            layers.append(BasicBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)
    
    def _features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)  # (B, 512)
    
    def forward(self, x):
        return self.fc(self._features(x))
    
    def get_hidden(self, x):
        return self._features(x)

def train_client(n_classes, train_ds, test_ds):
    model = ClientResNet18(n_classes).to(device)
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
# Server Backbones
# ============================================================
BACKBONES = {
    'ResNet-50': 'resnet50',
    'DINOv2-B': 'vit_base_patch14_dinov2.lvd142m',
    'CLIP-ViT': 'vit_base_patch16_clip_224.openai',
}

def make_backbone(name):
    model_name = BACKBONES[name]
    kwargs = {'pretrained': True, 'num_classes': 0}
    if 'dinov2' in model_name:
        kwargs['img_size'] = 224  # Override default 518
    bb = timm.create_model(model_name, **kwargs).to(device)
    bb.eval()
    for p in bb.parameters(): p.requires_grad = False
    # Get feature dim
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        feat_dim = bb(dummy).shape[1]
    print(f"  {name}: feat_dim={feat_dim}")
    return bb, feat_dim

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
# Rejector (L2H method)
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
        return self.head(torch.cat([self.features(x), hidden], dim=1))

# ============================================================
# L2H Joint Training
# ============================================================
def train_l2h_joint(client_model, server_head_init, server_bb,
                    train_32, train_224, n_classes, c_1=1.0):
    rejector = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=2).to(device)
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    srv_feats, _ = extract_features(server_bb, train_224)
    
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_hidden, all_labels, all_cc = [], [], [], []
    with torch.no_grad():
        for x, y in loader_32:
            x_d = x.to(device)
            all_x32.append(x); all_hidden.append(client_model.get_hidden(x_d).cpu())
            all_labels.append(y); all_cc.append((client_model(x_d).argmax(1).cpu() == y).float())
    
    dataset = TensorDataset(torch.cat(all_x32), torch.cat(all_hidden), torch.cat(all_cc),
                           srv_feats, torch.cat(all_labels))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    opt = optim.Adam(list(rejector.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    rejector.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, hidden, mc, sf, y in loader:
            x32, hidden, mc = x32.to(device), hidden.to(device), mc.to(device)
            sf, y = sf.to(device), y.to(device)
            sl = server_head(sf)
            ec = (sl.argmax(1) == y).float()
            rej_logits = rejector(x32, hidden)
            log_probs = F.log_softmax(rej_logits, dim=1)
            w_remote = 1 - c_1 + c_1 * ec
            rej_loss = -(w_remote * log_probs[:, 1] + mc * log_probs[:, 0]).mean()
            server_ce = F.cross_entropy(sl, y)
            loss = rej_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    rejector.eval(); server_head.eval()
    return rejector, server_head

# ============================================================
# Gatekeeper Joint Training
# ============================================================
def train_gatekeeper_joint(client_model, server_head_init, server_bb,
                           train_32, train_224, n_classes, alpha=0.8):
    client_ft = copy.deepcopy(client_model).to(device)
    for p in client_ft.parameters(): p.requires_grad = False
    for p in client_ft.fc.parameters(): p.requires_grad = True
    
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    srv_feats, _ = extract_features(server_bb, train_224)
    
    loader_32 = DataLoader(train_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32, all_labels = [], []
    for x, y in loader_32:
        all_x32.append(x); all_labels.append(y)
    
    dataset = TensorDataset(torch.cat(all_x32), srv_feats, torch.cat(all_labels))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    uniform = torch.ones(n_classes).to(device) / n_classes
    
    opt = optim.Adam(list(client_ft.fc.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    client_ft.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, sf, y in loader:
            x32, sf, y = x32.to(device), sf.to(device), y.to(device)
            cl = client_ft(x32)
            cp = cl.argmax(1).detach()
            corr = (cp == y).float(); incorr = 1 - corr
            ce = F.cross_entropy(cl, y, reduction='none')
            l_corr = (corr * ce).sum() / max(corr.sum(), 1)
            lp = F.log_softmax(cl, dim=1)
            kl = F.kl_div(lp, uniform.expand_as(lp), reduction='none').sum(1)
            l_incorr = (incorr * kl).sum() / max(incorr.sum(), 1)
            gk_loss = alpha * l_corr + (1 - alpha) * l_incorr
            sl = server_head(sf)
            server_ce = F.cross_entropy(sl, y)
            loss = gk_loss + server_ce
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    client_ft.eval(); server_head.eval()
    return client_ft, server_head

# ============================================================
# Scoring
# ============================================================
def get_scores_l2h(rejector, client_model, ds):
    rejector.eval(); client_model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            x_d = x.to(device)
            out = rejector(x_d, client_model.get_hidden(x_d))
            scores.append((out[:, 1] - out[:, 0]).cpu())
    return torch.cat(scores)

def get_scores_gk(client_ft, ds):
    client_ft.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            scores.append(-F.softmax(client_ft(x.to(device)), dim=1).max(1).values.cpu())
    return torch.cat(scores)

def get_scores_conf(client_model, ds):
    return get_scores_gk(client_model, ds)

def get_preds(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(preds), torch.cat(labels)

def get_server_preds(head, bb, ds):
    head.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds = []
    with torch.no_grad():
        for x, _ in loader:
            preds.append(head(bb(x.to(device))).argmax(1).cpu())
    return torch.cat(preds)

def system_acc(scores, rate, cp, sp, labels):
    n = len(scores); k = max(1, int(n * rate))
    _, idx = torch.topk(scores, k)
    defer = torch.zeros(n, dtype=torch.bool); defer[idx] = True
    return (torch.where(defer, sp, cp) == labels).float().mean().item()

def random_acc(rate, cp, sp, labels, trials=30):
    n = len(labels); k = max(1, int(n * rate))
    return np.mean([
        (torch.where(torch.zeros(n, dtype=torch.bool).scatter_(0, torch.randperm(n)[:k], True), sp, cp) == labels).float().mean().item()
        for _ in range(trials)])

# ============================================================
# Main
# ============================================================
def main():
    t_start = time.time()
    print("=" * 100)
    print("L2H v7: Backbone Comparison + New Datasets")
    print("=" * 100)
    
    print("\n[1] Loading data...")
    clients = get_client_data()
    
    print(f"\n[2] Training client ResNet-18 ({CLIENT_EPOCHS}ep)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client(info['n_classes'], info['train_32'], info['test_32'])
        client_models[cid] = model
        print(f"  {info['name']}: acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    eval_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    # For each backbone
    for bb_name in BACKBONES:
        print(f"\n{'='*100}")
        print(f"SERVER BACKBONE: {bb_name}")
        print(f"{'='*100}")
        
        print(f"\n[3] Loading {bb_name}...")
        bb, feat_dim = make_backbone(bb_name)
        
        print(f"\n[4] Extracting features & training heads ({HEAD_EPOCHS}ep)...")
        heads = {}
        srv_te_feats, srv_te_labels = {}, {}
        for cid, info in clients.items():
            t0 = time.time()
            tr_f, tr_l = extract_features(bb, info['train_224'])
            srv_te_feats[cid], srv_te_labels[cid] = extract_features(bb, info['test_224'])
            heads[cid] = train_head(tr_f, tr_l, info['n_classes'])
            with torch.no_grad():
                acc = (heads[cid](srv_te_feats[cid].to(device)).argmax(1).cpu() == srv_te_labels[cid]).float().mean().item()
            print(f"  {info['name']}: server_acc={acc:.4f} ({time.time()-t0:.0f}s)")
        
        print(f"\n[5] Stage 2: Joint training ({STAGE2_EPOCHS}ep)...")
        l2h_rej, l2h_head = {}, {}
        gk_client, gk_head = {}, {}
        for cid, info in clients.items():
            t0 = time.time()
            l2h_rej[cid], l2h_head[cid] = train_l2h_joint(
                client_models[cid], heads[cid], bb,
                info['train_32'], info['train_224'], info['n_classes'])
            gk_client[cid], gk_head[cid] = train_gatekeeper_joint(
                client_models[cid], heads[cid], bb,
                info['train_32'], info['train_224'], info['n_classes'])
            print(f"  {info['name']}: {time.time()-t0:.0f}s")
        
        print(f"\n[6] Results for {bb_name}:")
        
        # Baselines
        print(f"\n  {'Client':<13} {'Client':>8} {'Server':>8} {'Oracle':>8}")
        print(f"  {'-'*40}")
        for cid, info in clients.items():
            cp, labels = get_preds(client_models[cid], info['test_32'])
            sp = get_server_preds(heads[cid], bb, info['test_224'])
            co = (cp == labels).float().mean().item()
            so = (sp == labels).float().mean().item()
            orc = ((cp == labels) | (sp == labels)).float().mean().item()
            print(f"  {info['name']:<13} {co:>8.4f} {so:>8.4f} {orc:>8.4f}")
        
        # Per-rate results
        methods = ['L2H(c1=1)', 'GK(0.8)', 'ConfTh', 'Random']
        for rate in eval_rates:
            print(f"\n  --- Deferral {rate*100:.0f}% ---")
            header = f"  {'Client':<13}" + "".join(f" {m:>10}" for m in methods)
            print(header)
            print(f"  {'-'*60}")
            avgs = {m: 0 for m in methods}
            for cid, info in clients.items():
                nc = info['n_classes']
                cp, labels = get_preds(client_models[cid], info['test_32'])
                
                # L2H
                s = get_scores_l2h(l2h_rej[cid], client_models[cid], info['test_32'])
                sp_l2h = get_server_preds(l2h_head[cid], bb, info['test_224'])
                r_l2h = system_acc(s, rate, cp, sp_l2h, labels)
                
                # GK
                s = get_scores_gk(gk_client[cid], info['test_32'])
                cp_gk, _ = get_preds(gk_client[cid], info['test_32'])
                sp_gk = get_server_preds(gk_head[cid], bb, info['test_224'])
                r_gk = system_acc(s, rate, cp_gk, sp_gk, labels)
                
                # ConfTh
                s = get_scores_conf(client_models[cid], info['test_32'])
                sp_s1 = get_server_preds(heads[cid], bb, info['test_224'])
                r_conf = system_acc(s, rate, cp, sp_s1, labels)
                
                # Random
                r_rand = random_acc(rate, cp, sp_s1, labels)
                
                results = {'L2H(c1=1)': r_l2h, 'GK(0.8)': r_gk, 'ConfTh': r_conf, 'Random': r_rand}
                row = f"  {info['name']:<13}" + "".join(f" {results[m]:>10.4f}" for m in methods)
                print(row)
                for m in methods: avgs[m] += results[m]
            
            n = len(clients)
            print(f"  {'-'*60}")
            row = f"  {'Average':<13}" + "".join(f" {avgs[m]/n:>10.4f}" for m in methods)
            print(row)
        
        # Free backbone memory
        del bb
        torch.cuda.empty_cache()
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
