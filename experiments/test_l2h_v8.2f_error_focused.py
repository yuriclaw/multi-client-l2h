#!/usr/bin/env python3
"""
v8.2f: Error-Focused Server Training + Sequential Rejector
============================================================
Step 1: Train server head ONLY on samples where client predicts wrong
Step 2: Freeze server head, then train rejector on all samples

Comparison:
- Per(All): server trained on all 5K (baseline, same as v8.2c)
- Per(ErrOnly): server trained only on client-error samples
- ConfTh: softmax confidence baseline
"""
import random, copy, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T, timm
from torch.utils.data import DataLoader, Subset
from collections import defaultdict
import time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 100
DEFERRAL_RATES = [10, 30, 50, 70, 90]

# ─── Transforms ───
tf_32_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                          T.RandAugment(num_ops=2, magnitude=9),  # NEW
                          T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
tf_32_test = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
tf_224_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.RandAugment(num_ops=2, magnitude=9),  # NEW
                           T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
tf_224_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                          T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
tf_dino = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# ─── Client Models ───
class ClientAlexNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU())
        self.fc2 = nn.Linear(256, n_classes)
        self.hidden_dim = 256
    def get_hidden(self, x): return self.fc1(self.features(x))
    def forward(self, x): return self.fc2(self.get_hidden(x))

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.shortcut(x))

class ClientResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)  # NEW
        self.fc = nn.Linear(512, n_classes)
        self.hidden_dim = 512
    def _make(self, inc, outc, n, s):
        layers = [BasicBlock(inc, outc, s)]
        for _ in range(1, n): layers.append(BasicBlock(outc, outc))
        return nn.Sequential(*layers)
    def get_hidden(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.pool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1)
    def forward(self, x): return self.fc(self.drop(self.get_hidden(x)))

class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)  # NEW
        self.classifier = nn.Linear(1280, n_classes)
        self.hidden_dim = 1280
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

class ClientShuffleNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)  # NEW
        self.classifier = nn.Linear(1024, n_classes)
        self.hidden_dim = 1024
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

class ClientSqueezeNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)  # NEW
        self.hidden_dim = 512
        self.classifier = nn.Linear(512, n_classes)
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

# ─── Mixup ───
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ─── Dataset wrapper ───
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

# ─── Multi-transform dataset ───
class MultiTransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, tf32, tf224, tfdino):
        self.subset = subset; self.tf32 = tf32; self.tf224 = tf224; self.tfdino = tfdino
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.tf32(img), self.tf224(img), self.tfdino(img), label

# ─── Warmup + Cosine LR ───
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            alpha = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            return [base_lr * 0.5 * (1 + np.cos(np.pi * progress)) for base_lr in self.base_lrs]

# ─── Training with improvements ───
def train_client_improved(model, train_sub, tf_train, epochs, lr, backbone_lr=None, mixup_alpha=0.2, warmup=5):
    model.to(DEVICE)
    ds = TransformSubset(train_sub, tf_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)  # LABEL SMOOTHING

    if backbone_lr and hasattr(model, 'features'):
        opt = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': backbone_lr},
            {'params': (p for n, p in model.named_parameters() if 'features' not in n), 'lr': lr}
        ], weight_decay=5e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    sched = WarmupCosineLR(opt, warmup_epochs=warmup, total_epochs=epochs)

    model.train()
    for ep in range(epochs):
        total_loss = 0; total = 0; correct = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # Mixup
            mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=mixup_alpha)
            out = model(mixed_x)
            loss = mixup_criterion(crit, out, y_a, y_b, lam)
            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item() * x.size(0); total += x.size(0)
            # Track unmixed accuracy for logging
            with torch.no_grad():
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"    Epoch {ep+1}/{epochs}  Loss={total_loss/total:.4f}  TrainAcc={100*correct/total:.1f}%")
    model.eval()
    return model

# ─── Server ───
class DINOv2Server:
    def __init__(self):
        self.backbone = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
        for p in self.backbone.parameters(): p.requires_grad = False
        self.feat_dim = 768
    @torch.no_grad()
    def extract(self, images_224):
        return self.backbone(images_224)

class ServerHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, x): return self.net(x)

def train_server_head(server, head, subset, epochs=30):
    head.to(DEVICE).train()
    ds = TransformSubset(subset, tf_dino)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    for ep in range(epochs):
        tl = 0; tc = 0; tt = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): f = server.extract(x)
            out = head(f); loss = crit(out, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x.size(0); tc += (out.argmax(1)==y).sum().item(); tt += x.size(0)
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  SrvHead Ep {ep+1}/{epochs}  Acc={100*tc/tt:.1f}%")
    head.eval(); return head

def get_error_indices(client_model, subset, client_tf):
    """Return indices where client predicts incorrectly."""
    ds = TransformSubset(subset, client_tf)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    error_idx = []
    idx = 0
    with torch.no_grad():
        client_model.eval()
        for x, y in loader:
            pred = client_model(x.to(DEVICE)).argmax(1).cpu()
            for j in range(len(y)):
                if pred[j] != y[j]:
                    error_idx.append(idx)
                idx += 1
    return error_idx

def train_rejector_frozen_server(rejector, client_model, client_tf, server, server_head, subset,
                                  n_classes=100, epochs=30, lr=1e-3, c_1=1.0):
    """Train rejector with FROZEN server head (no joint training). L2H cost-sensitive loss."""
    rejector.to(DEVICE).train()
    # Pre-extract
    ds = MultiTransformDataset(subset, tf_32_train, client_tf, tf_dino)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    all_x32, all_hidden, all_mc, all_sf, all_labels = [], [], [], [], []
    with torch.no_grad():
        client_model.eval(); server_head.eval()
        for x32, xcl, xd, y in loader:
            xcl, xd = xcl.to(DEVICE), xd.to(DEVICE)
            h = client_model.get_hidden(xcl); cp = client_model(xcl).argmax(1)
            mc = (cp == y.to(DEVICE)).float().cpu()
            sf = server.extract(xd)
            # Server predictions are fixed
            sp = server_head(sf).argmax(1)
            ec = (sp == y.to(DEVICE)).float().cpu()
            all_x32.append(x32); all_hidden.append(h.cpu()); all_mc.append(mc)
            all_sf.append(ec); all_labels.append(y)
    tds = torch.utils.data.TensorDataset(torch.cat(all_x32), torch.cat(all_hidden),
                                          torch.cat(all_mc), torch.cat(all_sf), torch.cat(all_labels))
    loader = DataLoader(tds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(rejector.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        tl = 0; tt = 0
        for x32, hidden, mc, ec, y in loader:
            x32, hidden, mc, ec = x32.to(DEVICE), hidden.to(DEVICE), mc.to(DEVICE), ec.to(DEVICE)
            rej_logits = rejector(x32, hidden)
            lp = F.log_softmax(rej_logits, dim=1)
            w_remote = 1 - c_1 + c_1 * ec
            rej_loss = -(w_remote * lp[:,1] + mc * lp[:,0]).mean()
            opt.zero_grad(); rej_loss.backward(); opt.step()
            tl += rej_loss.item()*x32.size(0); tt += x32.size(0)
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Rej(frozen) Ep {ep+1}/{epochs}  Loss={tl/tt:.4f}")
    rejector.eval()
    return rejector

# ─── Rejector (AlexNet CNN) ───
class Rejector(nn.Module):
    def __init__(self, hidden_dim, out_dim=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128+hidden_dim, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x32, hidden):
        return self.head(torch.cat([self.features(x32), hidden], dim=1))

# ─── Universal Rejector ───
class UniversalRejector(nn.Module):
    def __init__(self, client_hidden_dims, out_dim=2):
        super().__init__()
        self.projections = nn.ModuleDict({
            str(cid): nn.Linear(dim, 256) for cid, dim in client_hidden_dims.items()
        })
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128+256, 128), nn.ReLU(), nn.Linear(128, out_dim))
    def forward(self, x32, hidden, client_id):
        proj = self.projections[str(client_id)](hidden)
        return self.head(torch.cat([self.features(x32), proj], dim=1))

# ─── L2H Joint Training (v7.6 style) ───
def train_rejector_joint(rejector, client_model, client_tf, server, server_head_init, subset,
                         n_classes=100, epochs=30, lr=1e-3, c_1=1.0):
    rejector.to(DEVICE).train()
    server_head = copy.deepcopy(server_head_init).to(DEVICE)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    ds = MultiTransformDataset(subset, tf_32_train, client_tf, tf_dino)
    loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
    all_x32, all_hidden, all_mc, all_sf, all_labels = [], [], [], [], []
    with torch.no_grad():
        client_model.eval()
        for x32, xcl, xd, y in loader:
            xcl, xd = xcl.to(DEVICE), xd.to(DEVICE)
            h = client_model.get_hidden(xcl); cp = client_model(xcl).argmax(1)
            mc = (cp == y.to(DEVICE)).float().cpu()
            sf = server.extract(xd).cpu()
            all_x32.append(x32); all_hidden.append(h.cpu()); all_mc.append(mc); all_sf.append(sf); all_labels.append(y)
    tds = torch.utils.data.TensorDataset(torch.cat(all_x32), torch.cat(all_hidden), torch.cat(all_mc), torch.cat(all_sf), torch.cat(all_labels))
    loader = DataLoader(tds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(list(rejector.parameters())+list(server_head.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        tl = 0; tt = 0
        for x32, hidden, mc, sf, y in loader:
            x32, hidden, mc, sf, y = x32.to(DEVICE), hidden.to(DEVICE), mc.to(DEVICE), sf.to(DEVICE), y.to(DEVICE)
            sl = server_head(sf); ec = (sl.argmax(1)==y).float()
            rej_logits = rejector(x32, hidden)
            lp = F.log_softmax(rej_logits, dim=1)
            w_remote = 1 - c_1 + c_1 * ec
            rej_loss = -(w_remote * lp[:,1] + mc * lp[:,0]).mean()
            loss = rej_loss + F.cross_entropy(sl, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x32.size(0); tt += x32.size(0)
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  L2H Ep {ep+1}/{epochs}  Loss={tl/tt:.4f}")
    rejector.eval(); server_head.eval()
    return rejector, server_head

# ─── Universal L2H Joint Training ───
def train_universal_rejector_joint(uni_rej, client_models, client_tfs, server, server_head_init,
                                    client_subsets, client_hidden_dims, n_classes=100,
                                    epochs=30, lr=1e-3, c_1=1.0):
    uni_rej.to(DEVICE).train()
    server_head = copy.deepcopy(server_head_init).to(DEVICE)
    server_head.train()
    for p in server_head.parameters(): p.requires_grad = True
    # Pre-extract per client
    all_x32, all_hidden, all_mc, all_sf, all_labels, all_cids = [], [], [], [], [], []
    for cid in range(len(client_models)):
        ds = MultiTransformDataset(client_subsets[cid], tf_32_train, client_tfs[cid], tf_dino)
        loader = DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)
        with torch.no_grad():
            client_models[cid].eval()
            for x32, xcl, xd, y in loader:
                xcl, xd = xcl.to(DEVICE), xd.to(DEVICE)
                h = client_models[cid].get_hidden(xcl)
                cp = client_models[cid](xcl).argmax(1)
                mc = (cp == y.to(DEVICE)).float().cpu()
                sf = server.extract(xd).cpu()
                all_x32.append(x32); all_hidden.append(h.cpu()); all_mc.append(mc)
                all_sf.append(sf); all_labels.append(y)
                all_cids.append(torch.full((x32.size(0),), cid, dtype=torch.long))
    # Pad hidden to max dim
    all_h_cat = [torch.cat([all_hidden[j] for j in range(len(all_hidden)) if all_cids[j][0]==cid]) for cid in range(len(client_models))]
    max_dim = max(h.shape[1] for h in all_h_cat)
    padded = []
    for h in all_hidden:
        if h.shape[1] < max_dim:
            h = torch.cat([h, torch.zeros(h.shape[0], max_dim - h.shape[1])], dim=1)
        padded.append(h)
    tds = torch.utils.data.TensorDataset(torch.cat(all_x32), torch.cat(padded), torch.cat(all_mc),
                                          torch.cat(all_sf), torch.cat(all_labels), torch.cat(all_cids))
    loader = DataLoader(tds, batch_size=128, shuffle=True)
    opt = torch.optim.Adam(list(uni_rej.parameters())+list(server_head.parameters()), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        tl = 0; tt = 0
        for x32, hidden, mc, sf, y, cids in loader:
            x32, mc, sf, y = x32.to(DEVICE), mc.to(DEVICE), sf.to(DEVICE), y.to(DEVICE)
            hidden = hidden.to(DEVICE)
            sl = server_head(sf); ec = (sl.argmax(1)==y).float()
            # Project hidden per client
            hp = torch.zeros(x32.size(0), 256, device=DEVICE)
            for c in cids.unique().tolist():
                mask = cids == c
                orig_dim = client_hidden_dims[c]
                hp[mask] = uni_rej.projections[str(c)](hidden[mask][:, :orig_dim])
            img_feat = uni_rej.features(x32)
            rl = uni_rej.head(torch.cat([img_feat, hp], dim=1))
            lp = F.log_softmax(rl, dim=1)
            w_remote = 1 - c_1 + c_1 * ec
            rej_loss = -(w_remote * lp[:,1] + mc * lp[:,0]).mean()
            loss = rej_loss + F.cross_entropy(sl, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item()*x32.size(0); tt += x32.size(0)
        sched.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  UniRej Joint Ep {ep+1}/{epochs}  Loss={tl/tt:.4f}")
    uni_rej.eval(); server_head.eval()
    return uni_rej, server_head

# ─── Scoring & Evaluation ───
def get_scores_and_preds(client_model, client_tf, server, server_head, rejector, subset,
                          is_universal=False, client_id=None):
    ds = MultiTransformDataset(subset, tf_32_test, client_tf, tf_dino)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    all_cp, all_sp, all_y, all_sc, all_conf = [], [], [], [], []
    with torch.no_grad():
        for x32, xcl, xd, y in loader:
            x32, xcl, xd = x32.to(DEVICE), xcl.to(DEVICE), xd.to(DEVICE)
            client_model.eval()
            cl = client_model(xcl); cp = cl.argmax(1)
            conf = F.softmax(cl, dim=1).max(1).values
            sf = server.extract(xd); sp = server_head(sf).argmax(1)
            h = client_model.get_hidden(xcl)
            if is_universal:
                rl = rejector(x32, h, client_id)
            else:
                rl = rejector(x32, h)
            sc = F.softmax(rl, dim=1)[:,1]
            all_cp.append(cp.cpu()); all_sp.append(sp.cpu()); all_y.append(y)
            all_sc.append(sc.cpu()); all_conf.append(conf.cpu())
    return (torch.cat(all_cp).numpy(), torch.cat(all_sp).numpy(), torch.cat(all_y).numpy(),
            torch.cat(all_sc).numpy(), torch.cat(all_conf).numpy())

def find_threshold(scores, rate):
    n = len(scores); k = int(n * rate / 100)
    s = np.sort(scores)[::-1]
    if k == 0: return float('inf')
    if k >= n: return -float('inf')
    return s[k-1]

def apply_threshold(cp, sp, y, scores, thresh):
    final = np.where(scores >= thresh, sp, cp)
    return np.mean(final == y)

def find_conf_threshold(conf, rate):
    n = len(conf); k = int(n * rate / 100)
    s = np.sort(conf)
    if k == 0: return -float('inf')
    if k >= n: return float('inf')
    return s[k-1]

def apply_conf_threshold(cp, sp, y, conf, thresh):
    final = np.where(conf <= thresh, sp, cp)
    return np.mean(final == y)

def random_defer(cp, sp, y, rate, trials=50):
    n = len(y); k = int(n * rate / 100)
    accs = []
    for _ in range(trials):
        idx = set(np.random.choice(n, k, replace=False))
        c = sum((sp[i]==y[i]) if i in idx else (cp[i]==y[i]) for i in range(n))
        accs.append(c/n)
    return np.mean(accs)

# ─── Calibration Analysis ───
def calibration_analysis(model, test_sub, tf_test, name):
    ds = TransformSubset(test_sub, tf_test)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
    confs, corrects = [], []
    with torch.no_grad():
        model.eval()
        for x, y in loader:
            probs = F.softmax(model(x.to(DEVICE)), dim=1)
            mc, pred = probs.max(1)
            confs.append(mc.cpu().numpy()); corrects.append((pred.cpu()==y).numpy())
    confs = np.concatenate(confs); corrects = np.concatenate(corrects)
    bins = np.linspace(0, 1, 11)
    print(f"\n  {name}: Overall={100*corrects.mean():.1f}%  AvgConf={confs.mean():.3f}")
    print(f"  {'Bin':>12} {'Count':>6} {'%Data':>6} {'Acc':>7} {'AvgConf':>8}")
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        mask = (confs > lo) & (confs <= hi) if i > 0 else (confs >= lo) & (confs <= hi)
        cnt = mask.sum()
        acc = corrects[mask].mean() if cnt > 0 else 0
        ac = confs[mask].mean() if cnt > 0 else 0
        print(f"  ({lo:.1f},{hi:.1f}] {cnt:>6} {100*cnt/len(confs):>5.1f}% {100*acc:>6.1f}% {ac:>8.4f}")
    return confs, corrects

# ═══════════════════════════════════════════
def main():
    t0 = time.time()
    print("=" * 80)
    print("v8.2c: Improved Training (LabelSmooth + Mixup + Warmup)")
    print("=" * 80)

    # 1) Load data
    print("\n[1] Loading CIFAR-100...")
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)
    idx = list(range(len(train_full))); random.shuffle(idx)
    client_train_idx = [idx[i*5000:(i+1)*5000] for i in range(5)]
    te_idx = list(range(len(test_full))); random.shuffle(te_idx)
    val_idx = te_idx[:500]; tst_idx = te_idx[500:]
    client_train_subs = [Subset(train_full, ci) for ci in client_train_idx]
    val_sub = Subset(test_full, val_idx); tst_sub = Subset(test_full, tst_idx)

    NAMES = ["AlexNet", "ResNet-18", "MobileNetV2", "ShuffleNetV2", "SqueezeNet"]
    CLASSES = [ClientAlexNet, ClientResNet18, ClientMobileNetV2, ClientShuffleNetV2, ClientSqueezeNet]
    TRAIN_TFS = [tf_32_train, tf_32_train, tf_224_train, tf_224_train, tf_224_train]
    TEST_TFS = [tf_32_test, tf_32_test, tf_224_test, tf_224_test, tf_224_test]
    EPOCHS = [150, 150, 60, 60, 60]  # LONGER
    LR = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    BB_LR = [None, None, 1e-4, 1e-4, 1e-4]
    WARMUP = [5, 5, 5, 5, 5]

    # 2) Train clients with improvements
    print("\n[2] Training client models (LabelSmooth=0.1, Mixup=0.2, Warmup=5ep)...")
    models = []
    for cid in range(5):
        print(f"\n--- Client {cid}: {NAMES[cid]} ({EPOCHS[cid]}ep) ---")
        m = CLASSES[cid](N_CLASSES)
        m = train_client_improved(m, client_train_subs[cid], TRAIN_TFS[cid],
                                   EPOCHS[cid], LR[cid], BB_LR[cid], mixup_alpha=0.2, warmup=WARMUP[cid])
        models.append(m)

    # 3) Test accuracy
    print("\n[3] Client accuracies:")
    for cid in range(5):
        ds = TransformSubset(tst_sub, TEST_TFS[cid])
        loader = DataLoader(ds, batch_size=256, num_workers=2)
        c = t = 0
        with torch.no_grad():
            for x, y in loader:
                c += (models[cid](x.to(DEVICE)).argmax(1)==y.to(DEVICE)).sum().item(); t += len(y)
        print(f"  {NAMES[cid]}: {100*c/t:.2f}%")

    # 4) Calibration analysis
    print("\n[4] Calibration Analysis:")
    for cid in range(5):
        calibration_analysis(models[cid], tst_sub, TEST_TFS[cid], NAMES[cid])

    # 5) DINOv2 server
    print("\n[5] Loading DINOv2-B/14 server...")
    server = DINOv2Server()

    # 6a) Per-client server heads (ALL data, baseline)
    print("\n[6a] Training per-client server heads (ALL 5K)...")
    srv_heads_all = []
    for cid in range(5):
        print(f"  Client {cid} ({NAMES[cid]}):")
        h = ServerHead(server.feat_dim, N_CLASSES)
        h = train_server_head(server, h, client_train_subs[cid], epochs=30)
        srv_heads_all.append(h)

    # 6b) Per-client server heads (ERROR-ONLY data)
    print("\n[6b] Training per-client server heads (ERROR-ONLY)...")
    srv_heads_err = []
    err_subsets = []
    for cid in range(5):
        err_idx = get_error_indices(models[cid], client_train_subs[cid], TEST_TFS[cid])
        n_err = len(err_idx)
        n_total = len(client_train_subs[cid])
        print(f"  Client {cid} ({NAMES[cid]}): {n_err}/{n_total} errors ({100*n_err/n_total:.1f}%)")
        # Map error indices back to original subset indices
        orig_indices = [client_train_subs[cid].indices[i] for i in err_idx]
        err_sub = Subset(train_full, orig_indices)
        err_subsets.append(err_sub)
        h = ServerHead(server.feat_dim, N_CLASSES)
        # More epochs for fewer samples to ensure convergence
        err_epochs = max(30, int(30 * n_total / max(n_err, 1)))
        err_epochs = min(err_epochs, 200)  # cap at 200
        print(f"    Training {err_epochs} epochs (scaled for {n_err} samples)...")
        h = train_server_head(server, h, err_sub, epochs=err_epochs)
        srv_heads_err.append(h)

    # 7a) L2H rejectors with ALL-data server head (joint, baseline)
    print("\n[7a] Training L2H rejectors (joint, ALL-data server head)...")
    rej_all = []; l2h_heads_all = []
    for cid in range(5):
        print(f"\n  Client {cid} ({NAMES[cid]}):")
        rej = Rejector(models[cid].hidden_dim)
        rej, lh = train_rejector_joint(rej, models[cid], TEST_TFS[cid], server, srv_heads_all[cid],
                                        client_train_subs[cid], epochs=30, c_1=1.0)
        rej_all.append(rej); l2h_heads_all.append(lh)

    # 7b) L2H rejectors with ERROR-ONLY server head (frozen server, train rejector on ALL data)
    print("\n[7b] Training L2H rejectors (frozen ERROR server head, rejector on ALL data)...")
    rej_err = []
    for cid in range(5):
        print(f"\n  Client {cid} ({NAMES[cid]}):")
        rej = Rejector(models[cid].hidden_dim)
        rej = train_rejector_frozen_server(rej, models[cid], TEST_TFS[cid], server, srv_heads_err[cid],
                                            client_train_subs[cid], epochs=30, c_1=1.0)
        rej_err.append(rej)

    # 8) Evaluate
    print("\n[8] Computing scores...")
    val_all = {}; test_all = {}  # L2H with all-data server
    val_err = {}; test_err = {}  # L2H with error-only server
    val_conf = {}; test_conf = {}  # ConfTh (all-data server)
    for cid in range(5):
        # L2H (all-data joint)
        val_all[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, l2h_heads_all[cid], rej_all[cid], val_sub)
        test_all[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, l2h_heads_all[cid], rej_all[cid], tst_sub)
        # L2H (error-only server, frozen)
        val_err[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, srv_heads_err[cid], rej_err[cid], val_sub)
        test_err[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, srv_heads_err[cid], rej_err[cid], tst_sub)
        # ConfTh (all-data server)
        val_conf[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, srv_heads_all[cid], rej_all[cid], val_sub)
        test_conf[cid] = get_scores_and_preds(models[cid], TEST_TFS[cid], server, srv_heads_all[cid], rej_all[cid], tst_sub)

    # 9) Results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    print("\n--- Server Head Accuracy ---")
    for cid in range(5):
        _, sp_all, y, _, _ = test_conf[cid]
        _, sp_err, _, _, _ = test_err[cid]
        print(f"  {NAMES[cid]}: All(5K)={100*np.mean(sp_all==y):.1f}%  ErrOnly={100*np.mean(sp_err==y):.1f}%")

    for rate in DEFERRAL_RATES:
        print(f"\n{'='*80}")
        print(f"  Deferral Rate = {rate}%")
        print(f"{'='*80}")
        print(f"  {'Client':<15} {'L2H(All)':>9} {'L2H(Err)':>9} {'ConfTh':>8} {'Random':>8} {'Err-CT':>7}")
        print(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")
        a_all, a_err, a_ct, a_rnd = [], [], [], []
        for cid in range(5):
            # L2H all-data
            th = find_threshold(val_all[cid][3], rate)
            acc_all = apply_threshold(test_all[cid][0], test_all[cid][1], test_all[cid][2], test_all[cid][3], th)
            # L2H error-only
            th_e = find_threshold(val_err[cid][3], rate)
            acc_err = apply_threshold(test_err[cid][0], test_err[cid][1], test_err[cid][2], test_err[cid][3], th_e)
            # ConfTh
            cth = find_conf_threshold(val_conf[cid][4], rate)
            ct = apply_conf_threshold(test_conf[cid][0], test_conf[cid][1], test_conf[cid][2], test_conf[cid][4], cth)
            # Random
            rnd = random_defer(test_conf[cid][0], test_conf[cid][1], test_conf[cid][2], rate)
            a_all.append(acc_all); a_err.append(acc_err); a_ct.append(ct); a_rnd.append(rnd)
            diff = acc_err - ct
            print(f"  {NAMES[cid]:<15} {100*acc_all:>8.2f}% {100*acc_err:>8.2f}% {100*ct:>7.2f}% {100*rnd:>7.2f}% {100*diff:>+6.1f}%")
        print(f"  {'-'*15} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*7}")
        print(f"  {'AVERAGE':<15} {100*np.mean(a_all):>8.2f}% {100*np.mean(a_err):>8.2f}% {100*np.mean(a_ct):>7.2f}% {100*np.mean(a_rnd):>7.2f}% {100*(np.mean(a_err)-np.mean(a_ct)):>+6.1f}%")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done.")

if __name__ == "__main__":
    main()
