"""
Two-Stage L2H v7.3: MobileNetV2 FULL FINETUNE Client + 3 Server Backbones
==========================================================================
Server backbones: ResNet-50, DINOv2-B/14, CLIP-ViT-B/16
Datasets: CIFAR-100, Flowers102, OxfordIIITPet, Food101, DTD
Methods: L2H(c1=1), Gatekeeper(α=0.8), ConfThresh, Random
Client: MobileNetV2 (ImageNet pretrained, ALL layers finetuned)
Rejector: MLP on client hidden features only
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
CLIENT_EPOCHS = 30  # Full finetune — fewer epochs needed with pretrained weights
HEAD_EPOCHS = 200
STAGE2_EPOCHS = 100
BATCH_SIZE = 128

INET_MEAN = (0.485, 0.456, 0.406)
INET_STD = (0.229, 0.224, 0.225)

# ============================================================
# Data — only 224x224 needed now (both client and server use 224)
# ============================================================
class EnsureRGB:
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

def get_transform_224():
    return transforms.Compose([
        EnsureRGB(),
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

def load_dataset(name, tf):
    if name == 'CIFAR-100':
        tr = torchvision.datasets.CIFAR100(DATA_DIR, train=True, download=True, transform=tf)
        te = torchvision.datasets.CIFAR100(DATA_DIR, train=False, download=True, transform=tf)
    elif name == 'Flowers102':
        tr = torchvision.datasets.Flowers102(DATA_DIR, split='train', download=True, transform=tf)
        te = torchvision.datasets.Flowers102(DATA_DIR, split='test', download=True, transform=tf)
    elif name == 'OxfordPet':
        tr = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='trainval', download=True, transform=tf)
        te = torchvision.datasets.OxfordIIITPet(DATA_DIR, split='test', download=True, transform=tf)
    elif name == 'Food101':
        tr = torchvision.datasets.Food101(DATA_DIR, split='train', download=True, transform=tf)
        te = torchvision.datasets.Food101(DATA_DIR, split='test', download=True, transform=tf)
    elif name == 'DTD':
        tr = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf)
        te = torchvision.datasets.DTD(DATA_DIR, split='test', download=True, transform=tf)
    return tr, te

def get_client_data():
    tf = get_transform_224()
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tr, te = load_dataset(cfg['name'], tf)
        tr_n = min(N_TRAIN, len(tr))
        te_n = min(N_TEST, len(te))
        tr_idx = random.sample(range(len(tr)), tr_n)
        te_idx = random.sample(range(len(te)), te_n)
        clients[cid] = {
            'name': cfg['name'], 'n_classes': cfg['n_classes'],
            'train': Subset(tr, tr_idx), 'test': Subset(te, te_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {tr_n} train, {te_n} test, {cfg['n_classes']} classes")
    return clients

# ============================================================
# Client: MobileNetV2 (ImageNet pretrained, frozen backbone + linear head)
# ============================================================
CLIENT_FEAT_DIM = 1280  # MobileNetV2 feature dim

class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features  # ALL layers trainable
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(CLIENT_FEAT_DIM, n_classes)
        # All layers trainable (full finetune)
    
    def get_hidden(self, x):
        f = self.features(x)
        return self.pool(f).flatten(1)  # (B, 1280)
    
    def forward(self, x):
        h = self.get_hidden(x)
        return self.classifier(h)

def train_client(n_classes, train_ds, test_ds):
    model = ClientMobileNetV2(n_classes).to(device)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    # Full finetune: lower LR for backbone, higher for classifier
    opt = optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3},
    ], weight_decay=1e-4)
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
# Pre-extract client features (avoid repeated forward passes)
# ============================================================
def extract_client_features(client_model, dataset):
    """Extract frozen MobileNetV2 backbone features + labels."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            feats.append(client_model.get_hidden(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)

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
        kwargs['img_size'] = 224
    bb = timm.create_model(model_name, **kwargs).to(device)
    bb.eval()
    for p in bb.parameters(): p.requires_grad = False
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
# Rejector (L2H) — MLP on client hidden features only
# ============================================================
class RejectorMLP(nn.Module):
    """MLP rejector using only client hidden features (no raw image)."""
    def __init__(self, hidden_dim=1280, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, out_dim),
        )
    
    def forward(self, hidden):
        return self.net(hidden)

# ============================================================
# L2H Joint Training
# ============================================================
def train_l2h_joint(client_model, server_head_init, server_bb,
                    train_ds, n_classes, c_1=1.0):
    rejector = RejectorMLP(CLIENT_FEAT_DIM, out_dim=2).to(device)
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    # Pre-extract all features
    srv_feats, _ = extract_features(server_bb, train_ds)
    cli_feats, labels = extract_client_features(client_model, train_ds)
    
    # Client predictions
    with torch.no_grad():
        cli_preds = nn.Linear(CLIENT_FEAT_DIM, n_classes).to('cpu')  # dummy
    # Actually get client predictions from features + classifier
    cli_logits = client_model.classifier(cli_feats.to(device)).cpu()
    cli_correct = (cli_logits.argmax(1) == labels).float()
    
    dataset = TensorDataset(cli_feats, cli_correct, srv_feats, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    opt = optim.Adam(list(rejector.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    rejector.train()
    for _ in range(STAGE2_EPOCHS):
        for cf, mc, sf, y in loader:
            cf, mc = cf.to(device), mc.to(device)
            sf, y = sf.to(device), y.to(device)
            sl = server_head(sf)
            ec = (sl.argmax(1) == y).float()
            rej_logits = rejector(cf)
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
                           train_ds, n_classes, alpha=0.8):
    # Deep copy client — finetune classifier only
    client_ft = copy.deepcopy(client_model).to(device)
    for p in client_ft.parameters(): p.requires_grad = False
    for p in client_ft.classifier.parameters(): p.requires_grad = True
    
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    # Pre-extract features (backbone frozen, so features are the same)
    cli_feats, labels = extract_client_features(client_model, train_ds)
    srv_feats, _ = extract_features(server_bb, train_ds)
    
    dataset = TensorDataset(cli_feats, srv_feats, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    uniform = torch.ones(n_classes).to(device) / n_classes
    
    opt = optim.Adam(list(client_ft.classifier.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    client_ft.classifier.train()
    for _ in range(STAGE2_EPOCHS):
        for cf, sf, y in loader:
            cf, sf, y = cf.to(device), sf.to(device), y.to(device)
            # Client logits from features (backbone frozen, just pass through classifier)
            cl = client_ft.classifier(cf)
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
    """L2H: score = rejector output (defer - keep)."""
    rejector.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            h = client_model.get_hidden(x.to(device))
            out = rejector(h)
            scores.append((out[:, 1] - out[:, 0]).cpu())
    return torch.cat(scores)

def get_scores_gk(client_ft, ds):
    """GK: score = -max softmax prob (lower confidence → more likely to defer)."""
    client_ft.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for x, _ in loader:
            h = client_ft.get_hidden(x.to(device))
            logits = client_ft.classifier(h)
            scores.append(-F.softmax(logits, dim=1).max(1).values.cpu())
    return torch.cat(scores)

def get_scores_conf(client_model, ds):
    """ConfThresh: same as GK but on original (unfinetuned) client."""
    return get_scores_gk(client_model, ds)

def get_preds(model, ds):
    model.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            preds.append(model(x.to(device)).argmax(1).cpu()); labels.append(y)
    return torch.cat(preds), torch.cat(labels)

def get_preds_from_feats(classifier, client_model, ds):
    """Get predictions using finetuned classifier on frozen features."""
    classifier.eval()
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    preds, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            h = client_model.get_hidden(x.to(device))
            # Use the finetuned client's classifier
            preds.append(classifier(h).argmax(1).cpu()); labels.append(y)
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
    print("L2H v7.3: MobileNetV2 Full Finetune Client + 3 Server Backbones")
    print("=" * 100)
    
    print("\n[1] Loading data...")
    clients = get_client_data()
    
    print(f"\n[2] Training client MobileNetV2 full finetune ({CLIENT_EPOCHS}ep)...")
    client_models = {}
    for cid, info in clients.items():
        t0 = time.time()
        model, acc = train_client(info['n_classes'], info['train'], info['test'])
        client_models[cid] = model
        print(f"  {info['name']}: acc={acc:.4f} ({time.time()-t0:.0f}s)")
    
    eval_rates = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for bb_name in BACKBONES:
        print(f"\n{'='*100}")
        print(f"SERVER BACKBONE: {bb_name}")
        print(f"{'='*100}")
        
        print(f"\n[3] Loading {bb_name}...")
        bb, feat_dim = make_backbone(bb_name)
        
        print(f"\n[4] Extracting features & training heads ({HEAD_EPOCHS}ep)...")
        heads = {}
        for cid, info in clients.items():
            t0 = time.time()
            tr_f, tr_l = extract_features(bb, info['train'])
            te_f, te_l = extract_features(bb, info['test'])
            heads[cid] = train_head(tr_f, tr_l, info['n_classes'])
            with torch.no_grad():
                acc = (heads[cid](te_f.to(device)).argmax(1).cpu() == te_l).float().mean().item()
            print(f"  {info['name']}: server_acc={acc:.4f} ({time.time()-t0:.0f}s)")
        
        print(f"\n[5] Stage 2: Joint training ({STAGE2_EPOCHS}ep)...")
        l2h_rej, l2h_head = {}, {}
        gk_client, gk_head = {}, {}
        for cid, info in clients.items():
            t0 = time.time()
            l2h_rej[cid], l2h_head[cid] = train_l2h_joint(
                client_models[cid], heads[cid], bb,
                info['train'], info['n_classes'])
            gk_client[cid], gk_head[cid] = train_gatekeeper_joint(
                client_models[cid], heads[cid], bb,
                info['train'], info['n_classes'])
            print(f"  {info['name']}: {time.time()-t0:.0f}s")
        
        print(f"\n[6] Results for {bb_name}:")
        
        # Baselines
        print(f"\n  {'Client':<13} {'Client':>8} {'Server':>8} {'Oracle':>8}")
        print(f"  {'-'*40}")
        for cid, info in clients.items():
            cp, labels = get_preds(client_models[cid], info['test'])
            sp = get_server_preds(heads[cid], bb, info['test'])
            co = (cp == labels).float().mean().item()
            so = (sp == labels).float().mean().item()
            orc = ((cp == labels) | (sp == labels)).float().mean().item()
            print(f"  {info['name']:<13} {co:>8.4f} {so:>8.4f} {orc:>8.4f}")
        
        methods = ['L2H(c1=1)', 'GK(0.8)', 'ConfTh', 'Random']
        for rate in eval_rates:
            print(f"\n  --- Deferral {rate*100:.0f}% ---")
            header = f"  {'Client':<13}" + "".join(f" {m:>10}" for m in methods)
            print(header)
            print(f"  {'-'*60}")
            avgs = {m: 0 for m in methods}
            for cid, info in clients.items():
                cp, labels = get_preds(client_models[cid], info['test'])
                
                # L2H
                s = get_scores_l2h(l2h_rej[cid], client_models[cid], info['test'])
                sp_l2h = get_server_preds(l2h_head[cid], bb, info['test'])
                r_l2h = system_acc(s, rate, cp, sp_l2h, labels)
                
                # GK — use finetuned client predictions
                s = get_scores_gk(gk_client[cid], info['test'])
                cp_gk, _ = get_preds(gk_client[cid], info['test'])
                sp_gk = get_server_preds(gk_head[cid], bb, info['test'])
                r_gk = system_acc(s, rate, cp_gk, sp_gk, labels)
                
                # ConfTh
                s = get_scores_conf(client_models[cid], info['test'])
                sp_s1 = get_server_preds(heads[cid], bb, info['test'])
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
        
        del bb
        torch.cuda.empty_cache()
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
