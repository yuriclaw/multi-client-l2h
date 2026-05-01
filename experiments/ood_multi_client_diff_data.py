#!/usr/bin/env python3
"""
OOD Multi-Client Experiment 1: Different Datasets per Client
==============================================================
5 clients, each trained on a different 10-class subset of CIFAR-100.
- Client 0: classes 0-9
- Client 1: classes 10-19
- Client 2: classes 20-29
- Client 3: classes 30-39
- Client 4: classes 40-49
Server: DINOv2 on all 100 classes.
OOD for each client: the other 90 classes.
Test: mixed known + OOD samples.
Compare: L2H(+OOD) vs ConfTh vs Random per client.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T, timm
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np, random, time, copy, sys

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_CLIENTS = 5
CLASSES_PER_CLIENT = 10
N_TOTAL = 100
DEFERRAL_RATES = [10, 30, 50, 70, 90]

tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                       T.RandAugment(num_ops=2, magnitude=9),
                       T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_32_train = T.Compose([T.Resize(36), T.RandomCrop(32), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_32_test = T.Compose([T.Resize(32), T.ToTensor(),
                         T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)  # line-buffered

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

class MultiTransformDataset(torch.utils.data.Dataset):
    def __init__(self, subset, tf32, tf224):
        self.subset = subset; self.tf32 = tf32; self.tf224 = tf224
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.tf32(img), self.tf224(img), label

class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3); self.classifier = nn.Linear(1280, n_classes)
        self.hidden_dim = 1280
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

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

class ServerHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, x): return self.net(x)

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def train_client(model, train_sub, epochs=60):
    model.to(DEVICE).train()
    ds = TransformSubset(train_sub, tf_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': (p for n,p in model.named_parameters() if 'features' not in n), 'lr': 1e-3}
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        tc = tt = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mx, ya, yb, lam = mixup_data(x, y)
            loss = lam * crit(model(mx), ya) + (1-lam) * crit(model(mx), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): tc += (model(x).argmax(1)==y).sum().item(); tt += len(y)
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"    Epoch {ep+1}/{epochs}  TrainAcc={100*tc/tt:.1f}%")
    model.eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def find_threshold(scores, rate):
    n = len(scores); k = int(n * rate / 100)
    s = np.sort(scores)[::-1]
    if k == 0: return float('inf')
    if k >= n: return -float('inf')
    return s[k-1]

def main():
    t0 = time.time()
    print("=" * 80)
    print("OOD Multi-Client: Different Datasets (5 clients × 10 classes each)")
    print("=" * 80)

    # 1) Data
    print("\n[1] Loading CIFAR-100...")
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    # Client class assignments
    client_classes = {}
    for cid in range(N_CLIENTS):
        client_classes[cid] = set(range(cid*CLASSES_PER_CLIENT, (cid+1)*CLASSES_PER_CLIENT))
        print(f"  Client {cid}: classes {cid*CLASSES_PER_CLIENT}-{(cid+1)*CLASSES_PER_CLIENT-1}")

    # Build per-client train sets (remap labels to 0-9)
    # We need a wrapper that remaps labels
    class RemapSubset(torch.utils.data.Dataset):
        def __init__(self, subset, class_set):
            self.subset = subset
            self.sorted_classes = sorted(class_set)
            self.remap = {c: i for i, c in enumerate(self.sorted_classes)}
        def __len__(self): return len(self.subset)
        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return img, self.remap[label]

    client_train_subs = []
    for cid in range(N_CLIENTS):
        cls = client_classes[cid]
        idx = [i for i in range(len(train_full)) if train_full.targets[i] in cls]
        random.shuffle(idx); idx = idx[:5000]
        sub = RemapSubset(Subset(train_full, idx), cls)
        client_train_subs.append(sub)
        print(f"  Client {cid} train: {len(idx)} samples")

    # Test sets: per client, 300 known + 300 OOD
    # Plus val: 200 known + 200 OOD
    client_val_subs = []; client_tst_subs = []; client_tst_is_known = []
    for cid in range(N_CLIENTS):
        cls = client_classes[cid]
        known_idx = [i for i in range(len(test_full)) if test_full.targets[i] in cls]
        ood_idx = [i for i in range(len(test_full)) if test_full.targets[i] not in cls]
        random.shuffle(known_idx); random.shuffle(ood_idx)
        val_idx = known_idx[:200] + ood_idx[:200]
        tst_idx = known_idx[200:500] + ood_idx[200:500]
        client_val_subs.append(Subset(test_full, val_idx))
        client_tst_subs.append(Subset(test_full, tst_idx))
        client_tst_is_known.append(np.array([True]*300 + [False]*300))

    # 2) Train clients (one at a time, clear cache between)
    print(f"\n[2] Training {N_CLIENTS} clients (MobileNetV2, 60ep each)...")
    clients = []
    for cid in range(N_CLIENTS):
        torch.cuda.empty_cache()
        print(f"  Client {cid} (classes {cid*10}-{(cid+1)*10-1}):")
        model = ClientMobileNetV2(CLASSES_PER_CLIENT)
        model = train_client(model, client_train_subs[cid], epochs=60)
        model.cpu()  # move to CPU to free GPU
        clients.append(model)
        torch.cuda.empty_cache()

    # 3) DINOv2 server
    print(f"\n[3] Loading DINOv2 server...")
    dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    for p in dino.parameters(): p.requires_grad = False
    feat_dim = 768

    # 4) Server head on all 100 classes (pre-extract features for speed)
    print(f"\n[4] Pre-extracting DINOv2 features for server training...")
    srv_train_idx = list(range(len(train_full)))
    random.shuffle(srv_train_idx); srv_train_idx = srv_train_idx[:25000]
    srv_sub = Subset(train_full, srv_train_idx)
    ds_srv = TransformSubset(srv_sub, tf_test)
    loader_srv = DataLoader(ds_srv, batch_size=64, shuffle=False, num_workers=2)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader_srv:
            all_feats.append(dino(x.to(DEVICE)).cpu())
            all_labels.append(y)
    srv_feats = torch.cat(all_feats); srv_labels = torch.cat(all_labels)
    torch.cuda.empty_cache()
    print(f"  Pre-extracted {len(srv_feats)} features")

    print(f"\n  Training server head (all 100 classes, 30ep)...")
    srv_head = ServerHead(feat_dim, N_TOTAL).to(DEVICE)
    feat_ds = TensorDataset(srv_feats, srv_labels)
    feat_loader = DataLoader(feat_ds, batch_size=256, shuffle=True)
    opt_s = torch.optim.Adam(srv_head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=30)
    crit_s = nn.CrossEntropyLoss()
    srv_head.train()
    for ep in range(30):
        tc = tt = 0
        for f, y in feat_loader:
            f, y = f.to(DEVICE), y.to(DEVICE)
            out = srv_head(f); loss = crit_s(out, y)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
            tc += (out.argmax(1)==y).sum().item(); tt += len(y)
        sched_s.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  SrvHead Ep {ep+1}/30  Acc={100*tc/tt:.1f}%")
    srv_head.eval()
    del srv_feats, srv_labels, feat_ds, feat_loader
    torch.cuda.empty_cache()

    # 5) Per-client rejector training
    print(f"\n[5] Training per-client L2H rejectors (+OOD data)...")
    rejectors = []
    for cid in range(N_CLIENTS):
        torch.cuda.empty_cache()
        clients[cid].to(DEVICE)  # move current client to GPU
        print(f"\n  Client {cid}:")
        cls = client_classes[cid]
        sorted_cls = sorted(cls)
        remap = {c: i for i, c in enumerate(sorted_cls)}

        # Rejector training data: 5K known + 5K OOD from train set
        known_idx = [i for i in range(len(train_full)) if train_full.targets[i] in cls]
        ood_idx = [i for i in range(len(train_full)) if train_full.targets[i] not in cls]
        random.shuffle(known_idx); random.shuffle(ood_idx)
        rej_idx = known_idx[:5000] + ood_idx[:5000]
        rej_sub = Subset(train_full, rej_idx)
        rej_is_known = [True]*5000 + [False]*5000

        # Pre-extract
        ds_rej = MultiTransformDataset(rej_sub, tf_32_train, tf_test)
        loader_rej = DataLoader(ds_rej, batch_size=64, shuffle=False, num_workers=2)
        all_x32, all_hidden, all_mc, all_ec = [], [], [], []
        with torch.no_grad():
            batch_i = 0
            for x32, x224, y in loader_rej:
                x32, x224, y = x32.to(DEVICE), x224.to(DEVICE), y.to(DEVICE)
                h = clients[cid].get_hidden(x224)
                # Client prediction (remapped labels for known, always wrong for OOD)
                cp = clients[cid](x224).argmax(1)
                # For known samples, remap y and check; for OOD, mc=0
                bs = x32.size(0)
                start = batch_i * 64  # approximate
                mc_list = []
                for j in range(bs):
                    global_idx = batch_i * 64 + j
                    if global_idx < 5000:  # known
                        ry = remap[y[j].item()]
                        mc_list.append(float(cp[j].item() == ry))
                    else:  # OOD
                        mc_list.append(0.0)
                mc = torch.tensor(mc_list, device=DEVICE)
                # Server prediction (original labels)
                sf = dino(x224); sp = srv_head(sf).argmax(1)
                ec = (sp == y).float()
                all_x32.append(x32.cpu()); all_hidden.append(h.cpu())
                all_mc.append(mc.cpu()); all_ec.append(ec.cpu())
                batch_i += 1
        torch.cuda.empty_cache()

        tds = TensorDataset(torch.cat(all_x32), torch.cat(all_hidden),
                             torch.cat(all_mc), torch.cat(all_ec))
        loader = DataLoader(tds, batch_size=64, shuffle=True)

        rej = Rejector(clients[cid].hidden_dim).to(DEVICE)
        rej.train()
        opt_r = torch.optim.Adam(rej.parameters(), lr=1e-3, weight_decay=1e-4)
        sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=30)
        for ep in range(30):
            tl = tt = 0
            for x32, hidden, mc, ec in loader:
                x32, hidden, mc, ec = x32.to(DEVICE), hidden.to(DEVICE), mc.to(DEVICE), ec.to(DEVICE)
                rl = rej(x32, hidden); lp = F.log_softmax(rl, dim=1)
                rej_loss = -(ec * lp[:,1] + mc * lp[:,0]).mean()
                opt_r.zero_grad(); rej_loss.backward(); opt_r.step()
                tl += rej_loss.item()*x32.size(0); tt += x32.size(0)
            sched_r.step()
            if (ep+1) % 10 == 0 or ep == 0:
                print(f"    Rej Ep {ep+1}/30  Loss={tl/tt:.4f}")
        rej.eval(); rej.cpu()
        rejectors.append(rej)
        clients[cid].cpu()  # free GPU
        del tds, loader, all_x32, all_hidden, all_mc, all_ec
        torch.cuda.empty_cache()

    # 6) Evaluate
    print(f"\n[6] Evaluating...")
    from sklearn.metrics import roc_auc_score

    for cid in range(N_CLIENTS):
        torch.cuda.empty_cache()
        clients[cid].to(DEVICE); rejectors[cid].to(DEVICE)
        cls = client_classes[cid]
        sorted_cls = sorted(cls)
        remap = {c: i for i, c in enumerate(sorted_cls)}

        # Get scores on val and test
        for label, sub, is_known in [("val", client_val_subs[cid], None), ("test", client_tst_subs[cid], client_tst_is_known[cid])]:
            ds = MultiTransformDataset(sub, tf_32_test, tf_test)
            loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
            cp_list, sp_list, y_list, conf_list, rej_list = [], [], [], [], []
            with torch.no_grad():
                for x32, x224, y in loader:
                    x32, x224, y = x32.to(DEVICE), x224.to(DEVICE), y.to(DEVICE)
                    cl = clients[cid](x224); cp = cl.argmax(1)
                    conf = F.softmax(cl, dim=1).max(1).values
                    sf = dino(x224); sp = srv_head(sf).argmax(1)
                    h = clients[cid].get_hidden(x224)
                    rs = F.softmax(rejectors[cid](x32, h), dim=1)[:,1]
                    cp_list.append(cp.cpu()); sp_list.append(sp.cpu()); y_list.append(y.cpu())
                    conf_list.append(conf.cpu()); rej_list.append(rs.cpu())
            if label == "val":
                val_rej_scores = torch.cat(rej_list).numpy()
                val_conf = torch.cat(conf_list).numpy()
            else:
                tst_cp = torch.cat(cp_list).numpy(); tst_sp = torch.cat(sp_list).numpy()
                tst_y = torch.cat(y_list).numpy(); tst_conf = torch.cat(conf_list).numpy()
                tst_rej = torch.cat(rej_list).numpy(); tst_known = is_known

        # Print per-client results
        # Client acc on known
        known_mask = tst_known
        cli_known_correct = 0
        for i in range(len(tst_y)):
            if known_mask[i]:
                ry = remap.get(tst_y[i], -1)
                if ry >= 0 and tst_cp[i] == ry:
                    cli_known_correct += 1
        cli_acc = cli_known_correct / known_mask.sum()
        srv_acc = np.mean(tst_sp == tst_y)

        if cid == 0:
            print(f"\n{'='*80}")
            print(f"RESULTS")
            print(f"{'='*80}")
        print(f"\n--- Client {cid} (classes {cid*10}-{(cid+1)*10-1}) ---")
        print(f"  Client acc (known): {100*cli_acc:.1f}%  Server acc (all): {100*srv_acc:.1f}%")

        for rate in DEFERRAL_RATES:
            # ConfTh
            th_c = find_threshold(-val_conf, rate)
            defer_c = (-tst_conf) >= th_c
            # L2H
            th_r = find_threshold(val_rej_scores, rate)
            defer_r = tst_rej >= th_r
            # System acc
            def sys_acc(defer_mask):
                correct = 0
                for i in range(len(tst_y)):
                    if defer_mask[i]:
                        correct += (tst_sp[i] == tst_y[i])
                    elif tst_known[i]:
                        ry = remap.get(tst_y[i], -1)
                        correct += (ry >= 0 and tst_cp[i] == ry)
                return correct / len(tst_y)
            acc_c = sys_acc(defer_c); acc_r = sys_acc(defer_r)
            # Random
            n = len(tst_y); k = int(n * rate / 100)
            rnd_accs = []
            for _ in range(20):
                d = np.zeros(n, dtype=bool); d[np.random.choice(n, k, replace=False)] = True
                rnd_accs.append(sys_acc(d))
            acc_rnd = np.mean(rnd_accs)
            ood_defer_c = defer_c[~tst_known].mean(); ood_defer_r = defer_r[~tst_known].mean()
            print(f"  Rate={rate:>2}%: L2H={100*acc_r:.1f}% ConfTh={100*acc_c:.1f}% Rnd={100*acc_rnd:.1f}% | OOD→Defer: L2H={100*ood_defer_r:.0f}% CT={100*ood_defer_c:.0f}%")

        # AUROC
        ood_labels = np.concatenate([np.zeros(known_mask.sum()), np.ones((~known_mask).sum())])
        auroc_conf = roc_auc_score(ood_labels, np.concatenate([-tst_conf[tst_known], -tst_conf[~tst_known]]))
        auroc_rej = roc_auc_score(ood_labels, np.concatenate([tst_rej[tst_known], tst_rej[~tst_known]]))
        print(f"  AUROC: L2H={auroc_rej:.4f}  ConfTh={auroc_conf:.4f}")
        clients[cid].cpu(); rejectors[cid].cpu()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY (Average across {N_CLIENTS} clients)")
    print(f"{'='*80}")
    # Re-evaluate for summary
    all_acc_r = {r: [] for r in DEFERRAL_RATES}
    all_acc_c = {r: [] for r in DEFERRAL_RATES}
    all_auroc_r = []; all_auroc_c = []
    # (would need to re-run; just print note)
    print("  (See per-client results above)")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done.")

if __name__ == "__main__":
    main()
