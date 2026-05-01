"""
Two-Stage L2H v7.6: AlexNet Rejector + Threshold-based Deferral
================================================================
Key change: Rejector back to AlexNet CNN (raw 32×32 image + client hidden)
Server: DINOv2-B/14 (frozen) + 2-layer MLP head
Client: MobileNetV2 (ImageNet pretrained, full finetune)
Methods: L2H(c1=1), Gatekeeper(α=0.8), ConfThresh, Random
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
N_VAL = 500
N_TEST = 500
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

def get_transform_32():
    return transforms.Compose([
        EnsureRGB(),
        transforms.Resize((32, 32)),
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
    tf224 = get_transform_224()
    tf32 = get_transform_32()
    clients = {}
    for cid, cfg in CLIENT_CONFIGS.items():
        tr224, te224 = load_dataset(cfg['name'], tf224)
        tr32, te32 = load_dataset(cfg['name'], tf32)
        # Use all available train data for training
        tr_n = min(N_TRAIN, len(tr224))
        tr_idx = random.sample(range(len(tr224)), tr_n)
        # Split test set into val + test
        te_total = min(N_VAL + N_TEST, len(te224))
        te_all_idx = random.sample(range(len(te224)), te_total)
        val_n = min(N_VAL, te_total // 2)
        te_n = te_total - val_n
        val_idx = te_all_idx[:val_n]
        te_idx = te_all_idx[val_n:]
        clients[cid] = {
            'name': cfg['name'], 'n_classes': cfg['n_classes'],
            'train': Subset(tr224, tr_idx), 'val': Subset(te224, val_idx), 'test': Subset(te224, te_idx),
            'train_32': Subset(tr32, tr_idx), 'val_32': Subset(te32, val_idx), 'test_32': Subset(te32, te_idx),
        }
        print(f"  Client {cid} ({cfg['name']}): {tr_n} train, {val_n} val, {te_n} test, {cfg['n_classes']} classes")
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
    'DINOv2-B': 'vit_base_patch14_dinov2.lvd142m',
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

def make_mlp_head(feat_dim, n_classes):
    """2-layer MLP head with ReLU and dropout."""
    return nn.Sequential(
        nn.Linear(feat_dim, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, n_classes),
    )

def train_head(feats, labels, n_classes, epochs=HEAD_EPOCHS):
    head = make_mlp_head(feats.shape[1], n_classes).to(device)
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
# Rejector (L2H) — AlexNet CNN on raw 32×32 image + client hidden
# ============================================================
class RejectorAlexNet(nn.Module):
    """AlexNet-style CNN rejector: raw 32×32 image + client hidden features."""
    def __init__(self, hidden_dim=1280, out_dim=2):
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
    
    def forward(self, x32, hidden):
        img_feat = self.features(x32)
        return self.head(torch.cat([img_feat, hidden], dim=1))

# ============================================================
# L2H Joint Training
# ============================================================
def train_l2h_joint(client_model, server_head_init, server_bb,
                    train_ds_224, train_ds_32, n_classes, c_1=1.0):
    rejector = RejectorAlexNet(CLIENT_FEAT_DIM, out_dim=2).to(device)
    server_head = copy.deepcopy(server_head_init).to(device)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    
    # Pre-extract server features (224)
    srv_feats, _ = extract_features(server_bb, train_ds_224)
    # Pre-extract client hidden features (224) and predictions
    cli_feats, labels = extract_client_features(client_model, train_ds_224)
    cli_logits = client_model.classifier(cli_feats.to(device)).cpu()
    cli_correct = (cli_logits.argmax(1) == labels).float()
    # Pre-extract 32×32 images
    loader_32 = DataLoader(train_ds_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    all_x32 = []
    for x, _ in loader_32:
        all_x32.append(x)
    all_x32 = torch.cat(all_x32)
    
    dataset = TensorDataset(all_x32, cli_feats, cli_correct, srv_feats, labels)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    opt = optim.Adam(list(rejector.parameters()) + list(server_head.parameters()),
                     lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STAGE2_EPOCHS)
    
    rejector.train()
    for _ in range(STAGE2_EPOCHS):
        for x32, cf, mc, sf, y in loader:
            x32, cf, mc = x32.to(device), cf.to(device), mc.to(device)
            sf, y = sf.to(device), y.to(device)
            sl = server_head(sf)
            ec = (sl.argmax(1) == y).float()
            rej_logits = rejector(x32, cf)
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
def get_scores_l2h(rejector, client_model, ds_32, ds_224):
    """L2H: score = rejector output (defer - keep). Uses 32×32 image + client hidden."""
    rejector.eval()
    loader_32 = DataLoader(ds_32, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    loader_224 = DataLoader(ds_224, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    scores = []
    with torch.no_grad():
        for (x32, _), (x224, _) in zip(loader_32, loader_224):
            h = client_model.get_hidden(x224.to(device))
            out = rejector(x32.to(device), h)
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

def find_threshold(val_scores, target_rate):
    """Find score threshold on validation set that achieves target deferral rate."""
    n = len(val_scores)
    k = max(1, int(n * target_rate))
    sorted_scores, _ = torch.sort(val_scores, descending=True)
    # threshold = score of the k-th highest sample
    threshold = sorted_scores[min(k - 1, n - 1)].item()
    return threshold

def system_acc_threshold(scores, threshold, cp, sp, labels):
    """Apply threshold: defer if score >= threshold."""
    defer = scores >= threshold
    actual_rate = defer.float().mean().item()
    acc = (torch.where(defer, sp, cp) == labels).float().mean().item()
    return acc, actual_rate

def system_acc_ranking(scores, rate, cp, sp, labels):
    """Original ranking-based (for comparison)."""
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
    print("L2H v7.6: AlexNet Rejector + Threshold-based Deferral")
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
        
        print(f"\n[4] Extracting features...")
        tr_feats, tr_labels, te_feats, te_labels = {}, {}, {}, {}
        for cid, info in clients.items():
            tr_feats[cid], tr_labels[cid] = extract_features(bb, info['train'])
            te_feats[cid], te_labels[cid] = extract_features(bb, info['test'])
        
        # --- Per-client heads ---
        print(f"\n[4a] Training PER-CLIENT heads ({HEAD_EPOCHS}ep)...")
        heads = {}
        for cid, info in clients.items():
            t0 = time.time()
            heads[cid] = train_head(tr_feats[cid], tr_labels[cid], info['n_classes'])
            with torch.no_grad():
                acc = (heads[cid](te_feats[cid].to(device)).argmax(1).cpu() == te_labels[cid]).float().mean().item()
            print(f"  {info['name']}: server_acc={acc:.4f} ({time.time()-t0:.0f}s)")
        
        # --- Universal head ---
        # Each client has different label space → remap to unified labels
        # Client 0: classes 0..99, Client 1: 100..201, Client 2: 202..238, etc.
        print(f"\n[4b] Training UNIVERSAL head ({HEAD_EPOCHS}ep)...")
        label_offsets = {}
        offset = 0
        for cid in sorted(clients.keys()):
            label_offsets[cid] = offset
            offset += clients[cid]['n_classes']
        total_classes = offset
        print(f"  Total unified classes: {total_classes}")
        
        all_tr_feats, all_tr_labels = [], []
        for cid in sorted(clients.keys()):
            all_tr_feats.append(tr_feats[cid])
            all_tr_labels.append(tr_labels[cid] + label_offsets[cid])
        
        t0 = time.time()
        uni_head = train_head(torch.cat(all_tr_feats), torch.cat(all_tr_labels), 
                              total_classes, epochs=HEAD_EPOCHS)
        print(f"  Universal head trained ({time.time()-t0:.0f}s)")
        
        # Evaluate universal head per-client
        for cid, info in clients.items():
            with torch.no_grad():
                logits = uni_head(te_feats[cid].to(device)).cpu()
                # Only look at this client's class range
                lo = label_offsets[cid]
                hi = lo + info['n_classes']
                client_logits = logits[:, lo:hi]
                acc = (client_logits.argmax(1) == te_labels[cid]).float().mean().item()
            print(f"  {info['name']}: universal_server_acc={acc:.4f}")
        
        print(f"\n[5] Stage 2: Joint training ({STAGE2_EPOCHS}ep)...")
        l2h_rej, l2h_head = {}, {}
        gk_client, gk_head = {}, {}
        for cid, info in clients.items():
            t0 = time.time()
            l2h_rej[cid], l2h_head[cid] = train_l2h_joint(
                client_models[cid], heads[cid], bb,
                info['train'], info['train_32'], info['n_classes'])
            gk_client[cid], gk_head[cid] = train_gatekeeper_joint(
                client_models[cid], heads[cid], bb,
                info['train'], info['n_classes'])
            print(f"  {info['name']}: {time.time()-t0:.0f}s")
        
        # Pre-compute universal server predictions per client
        def get_uni_server_preds(cid):
            with torch.no_grad():
                logits = uni_head(te_feats[cid].to(device)).cpu()
                lo = label_offsets[cid]
                hi = lo + clients[cid]['n_classes']
                return logits[:, lo:hi].argmax(1)
        
        print(f"\n[6] Results for {bb_name}:")
        
        # Baselines — show per-client AND universal server
        print(f"\n  {'Client':<13} {'Client':>8} {'Srv(PC)':>8} {'Srv(Uni)':>9} {'Orc(PC)':>8} {'Orc(Uni)':>9}")
        print(f"  {'-'*58}")
        for cid, info in clients.items():
            cp, labels = get_preds(client_models[cid], info['test'])
            sp_pc = get_server_preds(heads[cid], bb, info['test'])
            sp_uni = get_uni_server_preds(cid)
            co = (cp == labels).float().mean().item()
            so_pc = (sp_pc == labels).float().mean().item()
            so_uni = (sp_uni == labels).float().mean().item()
            orc_pc = ((cp == labels) | (sp_pc == labels)).float().mean().item()
            orc_uni = ((cp == labels) | (sp_uni == labels)).float().mean().item()
            print(f"  {info['name']:<13} {co:>8.4f} {so_pc:>8.4f} {so_uni:>9.4f} {orc_pc:>8.4f} {orc_uni:>9.4f}")
        
        # ---- Threshold-based evaluation ----
        # For each method: compute scores on val set → find threshold → apply on test set
        methods = ['L2H(c1=1)', 'GK(0.8)', 'ConfTh', 'Random']
        
        print(f"\n  === THRESHOLD-BASED DEFERRAL (threshold from val, applied on test) ===")
        for rate in eval_rates:
            print(f"\n  --- Target Deferral {rate*100:.0f}% ---")
            header = f"  {'Client':<13}" + "".join(f" {m:>12}" for m in methods) + f" {'Srv(PC)':>9} {'Srv(Uni)':>9}"
            print(header)
            print(f"  {'-'*88}")
            avgs = {m: 0 for m in methods}
            avgs['Srv(PC)'] = 0; avgs['Srv(Uni)'] = 0
            rate_avgs = {m: 0 for m in methods}
            
            for cid, info in clients.items():
                cp_te, labels_te = get_preds(client_models[cid], info['test'])
                sp_pc_te = get_server_preds(heads[cid], bb, info['test'])
                sp_uni_te = get_uni_server_preds(cid)
                
                # Server-only accuracies
                srv_pc = (sp_pc_te == labels_te).float().mean().item()
                srv_uni = (sp_uni_te == labels_te).float().mean().item()
                
                # --- L2H: threshold from val ---
                val_scores_l2h = get_scores_l2h(l2h_rej[cid], client_models[cid], info['val_32'], info['val'])
                thresh_l2h = find_threshold(val_scores_l2h, rate)
                test_scores_l2h = get_scores_l2h(l2h_rej[cid], client_models[cid], info['test_32'], info['test'])
                sp_l2h_te = get_server_preds(l2h_head[cid], bb, info['test'])
                r_l2h, dr_l2h = system_acc_threshold(test_scores_l2h, thresh_l2h, cp_te, sp_l2h_te, labels_te)
                
                # --- GK: threshold from val ---
                val_scores_gk = get_scores_gk(gk_client[cid], info['val'])
                thresh_gk = find_threshold(val_scores_gk, rate)
                test_scores_gk = get_scores_gk(gk_client[cid], info['test'])
                cp_gk_te, _ = get_preds(gk_client[cid], info['test'])
                sp_gk_te = get_server_preds(gk_head[cid], bb, info['test'])
                r_gk, dr_gk = system_acc_threshold(test_scores_gk, thresh_gk, cp_gk_te, sp_gk_te, labels_te)
                
                # --- ConfTh: threshold from val ---
                val_scores_conf = get_scores_conf(client_models[cid], info['val'])
                thresh_conf = find_threshold(val_scores_conf, rate)
                test_scores_conf = get_scores_conf(client_models[cid], info['test'])
                r_conf, dr_conf = system_acc_threshold(test_scores_conf, thresh_conf, cp_te, sp_pc_te, labels_te)
                
                # --- Random ---
                r_rand = random_acc(rate, cp_te, sp_pc_te, labels_te)
                
                results = {'L2H(c1=1)': r_l2h, 'GK(0.8)': r_gk, 'ConfTh': r_conf, 'Random': r_rand}
                row = f"  {info['name']:<13}" + "".join(f" {results[m]:>12.4f}" for m in methods) + f" {srv_pc:>9.4f} {srv_uni:>9.4f}"
                print(row)
                for m in methods: avgs[m] += results[m]
                avgs['Srv(PC)'] += srv_pc; avgs['Srv(Uni)'] += srv_uni
            
            n = len(clients)
            print(f"  {'-'*88}")
            row = f"  {'Average':<13}" + "".join(f" {avgs[m]/n:>12.4f}" for m in methods) + f" {avgs['Srv(PC)']/n:>9.4f} {avgs['Srv(Uni)']/n:>9.4f}"
            print(row)
        
        del bb
        torch.cuda.empty_cache()
    
    print(f"\nTotal time: {time.time()-t_start:.0f}s")

if __name__ == '__main__':
    main()
