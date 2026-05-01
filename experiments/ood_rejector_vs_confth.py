#!/usr/bin/env python3
"""
OOD Deferral: Rejector vs ConfThresh for Out-of-Distribution Detection
========================================================================
Client trained on 10 classes of CIFAR-100, server knows all 100 classes.
When OOD samples (90 unseen classes) arrive, compare:
- ConfThresh: defer based on client softmax confidence
- L2H Rejector: trained to detect "should defer" (client wrong OR OOD)
- Server handles all 100 classes via DINOv2

System accuracy = correct on known (client or server) + correct on OOD (server only)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T, timm
from torch.utils.data import DataLoader, Subset
import numpy as np, random, time, copy

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_KNOWN = 10
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

# ─── Client ───
class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3); self.classifier = nn.Linear(1280, n_classes)
        self.hidden_dim = 1280
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

# ─── Rejector ───
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

# ─── Server ───
class ServerHead(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, n_classes))
    def forward(self, x): return self.net(x)

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def main():
    t0 = time.time()
    print("=" * 80)
    print("OOD Deferral: L2H Rejector vs ConfThresh")
    print("=" * 80)

    # 1) Data
    print("\n[1] Loading CIFAR-100...")
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    # Train: known classes only (for client), all classes (for server)
    train_known_idx = [i for i in range(len(train_full)) if train_full.targets[i] < N_KNOWN]
    train_all_idx = list(range(len(train_full)))
    random.shuffle(train_known_idx); train_known_idx = train_known_idx[:5000]
    random.shuffle(train_all_idx); train_all_idx = train_all_idx[:25000]  # server gets more data

    # Test: mix of known + OOD
    test_known_idx = [i for i in range(len(test_full)) if test_full.targets[i] < N_KNOWN]
    test_ood_idx = [i for i in range(len(test_full)) if test_full.targets[i] >= N_KNOWN]
    random.shuffle(test_known_idx); random.shuffle(test_ood_idx)
    # Val: 500 known + 500 OOD; Test: 500 known + 500 OOD
    val_idx = test_known_idx[:500] + test_ood_idx[:500]
    tst_idx = test_known_idx[500:1000] + test_ood_idx[500:1000]
    # Also keep track of which are known/OOD in test
    tst_is_known = [True]*500 + [False]*500

    train_known_sub = Subset(train_full, train_known_idx)
    train_all_sub = Subset(train_full, train_all_idx)
    val_sub = Subset(test_full, val_idx)
    tst_sub = Subset(test_full, tst_idx)

    print(f"  Client train: {len(train_known_idx)} (classes 0-{N_KNOWN-1})")
    print(f"  Server train: {len(train_all_idx)} (all 100 classes)")
    print(f"  Val: {len(val_idx)} (500 known + 500 OOD)")
    print(f"  Test: {len(tst_idx)} (500 known + 500 OOD)")

    # 2) Train client on known classes
    print(f"\n[2] Training client MobileNetV2 on {N_KNOWN} classes (60ep)...")
    client = ClientMobileNetV2(N_KNOWN).to(DEVICE)
    ds = TransformSubset(train_known_sub, tf_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam([
        {'params': client.features.parameters(), 'lr': 1e-4},
        {'params': (p for n,p in client.named_parameters() if 'features' not in n), 'lr': 1e-3}
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    client.train()
    for ep in range(60):
        tc = tt = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mx, ya, yb, lam = mixup_data(x, y)
            loss = lam * crit(client(mx), ya) + (1-lam) * crit(client(mx), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad(): tc += (client(x).argmax(1)==y).sum().item(); tt += len(y)
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/60  TrainAcc={100*tc/tt:.1f}%")
    client.eval()
    for p in client.parameters(): p.requires_grad = False

    # 3) DINOv2 server (all 100 classes)
    print(f"\n[3] Loading DINOv2 server...")
    dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    for p in dino.parameters(): p.requires_grad = False
    feat_dim = 768

    print(f"\n[4] Training server head (all 100 classes, 30ep)...")
    srv_head = ServerHead(feat_dim, N_TOTAL).to(DEVICE)
    ds_srv = TransformSubset(train_all_sub, tf_test)
    loader_srv = DataLoader(ds_srv, batch_size=128, shuffle=True, num_workers=2)
    opt_s = torch.optim.Adam(srv_head.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt_s, T_max=30)
    crit_s = nn.CrossEntropyLoss()
    srv_head.train()
    for ep in range(30):
        tc = tt = 0
        for x, y in loader_srv:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.no_grad(): f = dino(x)
            out = srv_head(f); loss = crit_s(out, y)
            opt_s.zero_grad(); loss.backward(); opt_s.step()
            tc += (out.argmax(1)==y).sum().item(); tt += len(y)
        sched_s.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  SrvHead Ep {ep+1}/30  Acc={100*tc/tt:.1f}%")
    srv_head.eval()

    # 5) Train L2H rejector
    # Label: defer=1 if client wrong (includes ALL OOD since client can't classify them)
    # Use client training data (known) + some OOD data for rejector training
    # Rejector train set: known samples + OOD samples from train set
    print(f"\n[5] Training L2H rejector (cost-sensitive, frozen server, 30ep)...")
    train_ood_idx = [i for i in range(len(train_full)) if train_full.targets[i] >= N_KNOWN]
    random.shuffle(train_ood_idx); train_ood_idx = train_ood_idx[:5000]
    rej_train_idx = train_known_idx + train_ood_idx  # 5K known + 5K OOD
    rej_train_sub = Subset(train_full, rej_train_idx)
    print(f"  Rejector train: {len(rej_train_idx)} (5K known + 5K OOD)")

    # Pre-extract (smaller batch to avoid OOM)
    ds_rej = MultiTransformDataset(rej_train_sub, tf_32_train, tf_test)
    loader_rej = DataLoader(ds_rej, batch_size=64, shuffle=False, num_workers=2)
    all_x32, all_hidden, all_mc, all_ec = [], [], [], []
    with torch.no_grad():
        for x32, x224, y in loader_rej:
            x32, x224, y = x32.to(DEVICE), x224.to(DEVICE), y.to(DEVICE)
            h = client.get_hidden(x224)
            # Client prediction: for known classes, check if correct
            # For OOD classes (y >= N_KNOWN), client is always "wrong"
            cp = client(x224).argmax(1)
            # Map: if y < N_KNOWN, check cp==y; if y >= N_KNOWN, always wrong
            mc = torch.where(y < N_KNOWN, (cp == y).float(), torch.zeros_like(y, dtype=torch.float))
            # Server prediction
            sf = dino(x224)
            sp = srv_head(sf).argmax(1)
            ec = (sp == y).float()
            all_x32.append(x32.cpu()); all_hidden.append(h.cpu())
            all_mc.append(mc.cpu()); all_ec.append(ec.cpu())
    # Clear GPU cache after extraction
    torch.cuda.empty_cache()

    tds = torch.utils.data.TensorDataset(torch.cat(all_x32), torch.cat(all_hidden),
                                          torch.cat(all_mc), torch.cat(all_ec))
    loader = DataLoader(tds, batch_size=64, shuffle=True)

    rejector = Rejector(client.hidden_dim).to(DEVICE)
    rejector.train()
    opt_r = torch.optim.Adam(rejector.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_r = torch.optim.lr_scheduler.CosineAnnealingLR(opt_r, T_max=30)
    for ep in range(30):
        tl = tt = 0
        for x32, hidden, mc, ec in loader:
            x32, hidden, mc, ec = x32.to(DEVICE), hidden.to(DEVICE), mc.to(DEVICE), ec.to(DEVICE)
            rl = rejector(x32, hidden)
            lp = F.log_softmax(rl, dim=1)
            w_remote = ec  # c1=1: w_remote = ec
            rej_loss = -(w_remote * lp[:,1] + mc * lp[:,0]).mean()
            opt_r.zero_grad(); rej_loss.backward(); opt_r.step()
            tl += rej_loss.item()*x32.size(0); tt += x32.size(0)
        sched_r.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Rej Ep {ep+1}/30  Loss={tl/tt:.4f}")
    rejector.eval()

    # 6) Also train a rejector WITHOUT OOD data (only known classes)
    print(f"\n[6] Training L2H rejector (known-only data, 30ep)...")
    torch.cuda.empty_cache()  # Clear before second rejector
    ds_rej_known = MultiTransformDataset(train_known_sub, tf_32_train, tf_test)
    loader_rk = DataLoader(ds_rej_known, batch_size=64, shuffle=False, num_workers=2)
    all_x32k, all_hk, all_mck, all_eck = [], [], [], []
    with torch.no_grad():
        for x32, x224, y in loader_rk:
            x32, x224, y = x32.to(DEVICE), x224.to(DEVICE), y.to(DEVICE)
            h = client.get_hidden(x224); cp = client(x224).argmax(1)
            mc = (cp == y).float()
            sf = dino(x224); sp = srv_head(sf).argmax(1); ec = (sp == y).float()
            all_x32k.append(x32.cpu()); all_hk.append(h.cpu())
            all_mck.append(mc.cpu()); all_eck.append(ec.cpu())
    torch.cuda.empty_cache()
    tds_k = torch.utils.data.TensorDataset(torch.cat(all_x32k), torch.cat(all_hk),
                                             torch.cat(all_mck), torch.cat(all_eck))
    loader_k = DataLoader(tds_k, batch_size=64, shuffle=True)
    rej_known = Rejector(client.hidden_dim).to(DEVICE)
    rej_known.train()
    opt_rk = torch.optim.Adam(rej_known.parameters(), lr=1e-3, weight_decay=1e-4)
    sched_rk = torch.optim.lr_scheduler.CosineAnnealingLR(opt_rk, T_max=30)
    for ep in range(30):
        tl = tt = 0
        for x32, hidden, mc, ec in loader_k:
            x32, hidden, mc, ec = x32.to(DEVICE), hidden.to(DEVICE), mc.to(DEVICE), ec.to(DEVICE)
            rl = rej_known(x32, hidden)
            lp = F.log_softmax(rl, dim=1)
            rej_loss = -(ec * lp[:,1] + mc * lp[:,0]).mean()
            opt_rk.zero_grad(); rej_loss.backward(); opt_rk.step()
            tl += rej_loss.item()*x32.size(0); tt += x32.size(0)
        sched_rk.step()
        if (ep+1) % 10 == 0 or ep == 0:
            print(f"  Rej(known) Ep {ep+1}/30  Loss={tl/tt:.4f}")
    rej_known.eval()

    # 7) Evaluate on test set (500 known + 500 OOD)
    print(f"\n[7] Evaluating on test set...")
    ds_val = MultiTransformDataset(val_sub, tf_32_test, tf_test)
    ds_tst = MultiTransformDataset(tst_sub, tf_32_test, tf_test)

    def get_all_scores(dataset):
        loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)
        cp_list, sp_list, y_list, conf_list, rej_score_list, rej_known_score_list = [], [], [], [], [], []
        with torch.no_grad():
            for x32, x224, y in loader:
                x32, x224, y = x32.to(DEVICE), x224.to(DEVICE), y.to(DEVICE)
                # Client
                cl = client(x224); cp = cl.argmax(1)
                conf = F.softmax(cl, dim=1).max(1).values
                # Server
                sf = dino(x224); sp = srv_head(sf).argmax(1)
                # Rejector scores
                h = client.get_hidden(x224)
                rs = F.softmax(rejector(x32, h), dim=1)[:,1]
                rks = F.softmax(rej_known(x32, h), dim=1)[:,1]
                cp_list.append(cp.cpu()); sp_list.append(sp.cpu()); y_list.append(y.cpu())
                conf_list.append(conf.cpu()); rej_score_list.append(rs.cpu()); rej_known_score_list.append(rks.cpu())
        return (torch.cat(cp_list).numpy(), torch.cat(sp_list).numpy(), torch.cat(y_list).numpy(),
                torch.cat(conf_list).numpy(), torch.cat(rej_score_list).numpy(), torch.cat(rej_known_score_list).numpy())

    val_cp, val_sp, val_y, val_conf, val_rej, val_rejk = get_all_scores(ds_val)
    tst_cp, tst_sp, tst_y, tst_conf, tst_rej, tst_rejk = get_all_scores(ds_tst)
    tst_known = np.array(tst_is_known)

    # Client can only predict classes 0-9; for classes >=10, client is always wrong
    # System: if not deferred → client prediction (wrong for OOD); if deferred → server prediction
    def system_acc(cp, sp, y, defer_mask, is_known):
        """Client handles known classes (0-9), maps to those labels.
        For non-deferred OOD, client is wrong. For deferred, server handles."""
        correct = 0; n = len(y)
        for i in range(n):
            if defer_mask[i]:
                correct += (sp[i] == y[i])
            else:
                if is_known[i]:
                    correct += (cp[i] == y[i])
                # else: client wrong on OOD, correct += 0
        return correct / n

    def find_threshold(scores, rate, descending=True):
        n = len(scores); k = int(n * rate / 100)
        if descending:
            s = np.sort(scores)[::-1]
        else:
            s = np.sort(scores)
        if k == 0: return float('inf') if descending else -float('inf')
        if k >= n: return -float('inf') if descending else float('inf')
        return s[k-1]

    # 8) Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    # Server accuracy
    srv_acc_known = np.mean(tst_sp[tst_known] == tst_y[tst_known])
    srv_acc_ood = np.mean(tst_sp[~tst_known] == tst_y[~tst_known])
    srv_acc_all = np.mean(tst_sp == tst_y)
    cli_acc_known = np.mean(tst_cp[tst_known] == tst_y[tst_known])
    print(f"\n  Client acc (known only): {100*cli_acc_known:.1f}%")
    print(f"  Server acc (known): {100*srv_acc_known:.1f}%  (OOD): {100*srv_acc_ood:.1f}%  (all): {100*srv_acc_all:.1f}%")
    print(f"  Client-only system acc: {100*(cli_acc_known*0.5):.1f}% (OOD always wrong)")
    print(f"  Server-only system acc: {100*srv_acc_all:.1f}%")

    # Per deferral rate
    for rate in DEFERRAL_RATES:
        print(f"\n{'='*80}")
        print(f"  Deferral Rate = {rate}%")
        print(f"{'='*80}")

        # ConfTh: defer lowest confidence
        val_conf_neg = -val_conf  # lower conf → higher "defer score"
        tst_conf_neg = -tst_conf
        th_conf = find_threshold(val_conf_neg, rate)
        defer_conf = tst_conf_neg >= th_conf
        acc_conf = system_acc(tst_cp, tst_sp, tst_y, defer_conf, tst_known)
        known_defer_conf = defer_conf[tst_known].mean()
        ood_defer_conf = defer_conf[~tst_known].mean()

        # L2H Rejector (trained with OOD)
        th_rej = find_threshold(val_rej, rate)
        defer_rej = tst_rej >= th_rej
        acc_rej = system_acc(tst_cp, tst_sp, tst_y, defer_rej, tst_known)
        known_defer_rej = defer_rej[tst_known].mean()
        ood_defer_rej = defer_rej[~tst_known].mean()

        # L2H Rejector (trained known-only, no OOD exposure)
        th_rejk = find_threshold(val_rejk, rate)
        defer_rejk = tst_rejk >= th_rejk
        acc_rejk = system_acc(tst_cp, tst_sp, tst_y, defer_rejk, tst_known)
        known_defer_rejk = defer_rejk[tst_known].mean()
        ood_defer_rejk = defer_rejk[~tst_known].mean()

        # Random
        n = len(tst_y); k = int(n * rate / 100)
        rnd_accs = []
        for _ in range(50):
            d = np.zeros(n, dtype=bool)
            d[np.random.choice(n, k, replace=False)] = True
            rnd_accs.append(system_acc(tst_cp, tst_sp, tst_y, d, tst_known))
        acc_rnd = np.mean(rnd_accs)

        print(f"  {'Method':<20} {'SysAcc':>8} {'Known→Defer':>12} {'OOD→Defer':>10}")
        print(f"  {'-'*20} {'-'*8} {'-'*12} {'-'*10}")
        print(f"  {'ConfTh':<20} {100*acc_conf:>7.2f}% {100*known_defer_conf:>10.1f}% {100*ood_defer_conf:>9.1f}%")
        print(f"  {'L2H(+OOD train)':<20} {100*acc_rej:>7.2f}% {100*known_defer_rej:>10.1f}% {100*ood_defer_rej:>9.1f}%")
        print(f"  {'L2H(known only)':<20} {100*acc_rejk:>7.2f}% {100*known_defer_rejk:>10.1f}% {100*ood_defer_rejk:>9.1f}%")
        print(f"  {'Random':<20} {100*acc_rnd:>7.2f}%")

    # AUROC comparison
    from sklearn.metrics import roc_auc_score
    labels_ood = np.concatenate([np.zeros(500), np.ones(500)])  # 1=OOD should defer
    auroc_conf = roc_auc_score(labels_ood, -tst_conf)  # lower conf → more likely OOD
    auroc_rej = roc_auc_score(labels_ood, tst_rej)
    auroc_rejk = roc_auc_score(labels_ood, tst_rejk)
    print(f"\n--- AUROC for OOD Detection ---")
    print(f"  ConfTh (neg conf):     {auroc_conf:.4f}")
    print(f"  L2H (+OOD train):      {auroc_rej:.4f}")
    print(f"  L2H (known only):      {auroc_rejk:.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done.")

if __name__ == "__main__":
    main()
