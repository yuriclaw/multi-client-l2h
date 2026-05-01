#!/usr/bin/env python3
"""
OOD Multi-Client Experiment 2: Same Dataset, Different Architectures
=====================================================================
5 clients on same 10 classes (CIFAR-100 classes 0-9) but different architectures:
- Client 0: AlexNet
- Client 1: ResNet-18
- Client 2: MobileNetV2
- Client 3: ShuffleNetV2
- Client 4: SqueezeNet
Server: DINOv2 on all 100 classes.
OOD: 90 unseen classes.
Compare: L2H(+OOD) vs ConfTh per client architecture.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T, timm
from torch.utils.data import DataLoader, Subset, TensorDataset
import numpy as np, random, time, sys

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_KNOWN = 10
N_TOTAL = 100
DEFERRAL_RATES = [10, 30, 50, 70, 90]
NAMES = ["AlexNet", "ResNet-18", "MobileNetV2", "ShuffleNetV2", "SqueezeNet"]

tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_32_train = T.Compose([T.Resize(36), T.RandomCrop(32), T.RandomHorizontalFlip(),
                          T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_32_test = T.Compose([T.Resize(32), T.ToTensor(),
                         T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)

# Per-architecture train transforms (different input sizes)
def get_train_tf(name):
    if name == "AlexNet":
        return T.Compose([T.Resize(256), T.RandomCrop(227), T.RandomHorizontalFlip(),
                           T.RandAugment(num_ops=2, magnitude=9),
                           T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    else:
        return T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                           T.RandAugment(num_ops=2, magnitude=9),
                           T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

def get_test_tf(name):
    if name == "AlexNet":
        return T.Compose([T.Resize(256), T.CenterCrop(227), T.ToTensor(),
                           T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
    else:
        return tf_test

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

class TripleTransformDataset(torch.utils.data.Dataset):
    """Returns (x32, x_client, x_dino, label)"""
    def __init__(self, subset, tf32, tf_cli, tf_dino):
        self.subset = subset; self.tf32 = tf32; self.tf_cli = tf_cli; self.tf_dino = tf_dino
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.tf32(img), self.tf_cli(img), self.tf_dino(img), label

# ─── Client Architectures ───
class ClientAlexNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = base.features; self.avgpool = base.avgpool
        self.hidden_dim = 256
        self.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(256*6*6, 256), nn.ReLU(), nn.Linear(256, n_classes))
    def get_hidden(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        return self.classifier[0](f)  # through dropout
    def forward(self, x):
        h = self.avgpool(self.features(x)).flatten(1)
        return self.classifier(h)
    def get_hidden(self, x):
        h = self.avgpool(self.features(x)).flatten(1)
        # hidden = after first linear
        h = self.classifier[0](h)  # dropout
        h = self.classifier[1](h)  # linear 256*36 -> 256
        return h

class ClientResNet18(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.hidden_dim = 512; self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(512, n_classes)
    def get_hidden(self, x): return self.backbone(x).flatten(1)
    def forward(self, x): return self.fc(self.drop(self.get_hidden(x)))

class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3); self.classifier = nn.Linear(1280, n_classes)
        self.hidden_dim = 1280
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

class ClientShuffleNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5)
        self.pool = nn.AdaptiveAvgPool2d(1); self.hidden_dim = 1024
        self.drop = nn.Dropout(0.3); self.fc = nn.Linear(1024, n_classes)
    def get_hidden(self, x): return self.pool(self.backbone(x)).flatten(1)
    def forward(self, x): return self.fc(self.drop(self.get_hidden(x)))

class ClientSqueezeNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        self.features = base.features; self.pool = nn.AdaptiveAvgPool2d(1)
        self.hidden_dim = 512; self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(512, n_classes)
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.fc(self.drop(self.get_hidden(x)))

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

CLIENT_CLASSES = [ClientAlexNet, ClientResNet18, ClientMobileNetV2, ClientShuffleNetV2, ClientSqueezeNet]

def find_threshold(scores, rate):
    n = len(scores); k = int(n * rate / 100)
    s = np.sort(scores)[::-1]
    if k == 0: return float('inf')
    if k >= n: return -float('inf')
    return s[k-1]

def main():
    t0 = time.time()
    print("=" * 80)
    print("OOD Multi-Client: Same Dataset, Different Architectures")
    print("=" * 80)

    # 1) Data
    print("\n[1] Loading CIFAR-100...")
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    known_classes = set(range(N_KNOWN))
    train_known_idx = [i for i in range(len(train_full)) if train_full.targets[i] < N_KNOWN]
    random.shuffle(train_known_idx); train_known_idx = train_known_idx[:5000]
    train_known_sub = Subset(train_full, train_known_idx)

    # Test
    test_known_idx = [i for i in range(len(test_full)) if test_full.targets[i] < N_KNOWN]
    test_ood_idx = [i for i in range(len(test_full)) if test_full.targets[i] >= N_KNOWN]
    random.shuffle(test_known_idx); random.shuffle(test_ood_idx)
    val_idx = test_known_idx[:500] + test_ood_idx[:500]
    tst_idx = test_known_idx[500:1000] + test_ood_idx[500:1000]
    tst_is_known = np.array([True]*500 + [False]*500)
    val_sub = Subset(test_full, val_idx)
    tst_sub = Subset(test_full, tst_idx)

    print(f"  Train: {len(train_known_idx)} (classes 0-{N_KNOWN-1})")
    print(f"  Val/Test: 500 known + 500 OOD each")

    # 2) Train 5 clients with different architectures
    print(f"\n[2] Training 5 clients (different architectures)...")
    clients = []; client_test_tfs = []
    for cid in range(5):
        print(f"\n  Client {cid} ({NAMES[cid]}):")
        model = CLIENT_CLASSES[cid](N_KNOWN).to(DEVICE)
        train_tf = get_train_tf(NAMES[cid])
        test_tf_c = get_test_tf(NAMES[cid])
        client_test_tfs.append(test_tf_c)
        ds = TransformSubset(train_known_sub, train_tf)
        loader = DataLoader(ds, batch_size=64, shuffle=True, num_workers=2)
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        if NAMES[cid] == "AlexNet":
            opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        else:
            opt = torch.optim.Adam([
                {'params': [p for n,p in model.named_parameters() if 'fc' not in n and 'classifier' not in n], 'lr': 1e-4},
                {'params': [p for n,p in model.named_parameters() if 'fc' in n or 'classifier' in n], 'lr': 1e-3}
            ], weight_decay=5e-4)
        epochs = 60
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        model.train()
        for ep in range(epochs):
            tc = tt = 0
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                mx, ya, yb, lam = mixup_data(x, y)
                out_mx = model(mx)
                loss = lam * crit(out_mx, ya) + (1-lam) * crit(out_mx, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad(): tc += (model(x).argmax(1)==y).sum().item(); tt += len(y)
            sched.step()
            if (ep+1) % 20 == 0 or ep == 0:
                print(f"    Epoch {ep+1}/{epochs}  TrainAcc={100*tc/tt:.1f}%")
        model.eval()
        for p in model.parameters(): p.requires_grad = False
        model.cpu()  # free GPU
        clients.append(model)
        torch.cuda.empty_cache()

    # 3) DINOv2 server
    print(f"\n[3] Loading DINOv2 server...")
    dino = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0, img_size=224).to(DEVICE).eval()
    for p in dino.parameters(): p.requires_grad = False
    feat_dim = 768

    # 4) Server head (pre-extract features)
    print(f"\n[4] Pre-extracting DINOv2 features...")
    srv_train_idx = list(range(len(train_full)))
    random.shuffle(srv_train_idx); srv_train_idx = srv_train_idx[:25000]
    srv_sub = Subset(train_full, srv_train_idx)
    ds_srv = TransformSubset(srv_sub, tf_test)
    loader_srv = DataLoader(ds_srv, batch_size=64, shuffle=False, num_workers=2)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader_srv:
            all_feats.append(dino(x.to(DEVICE)).cpu()); all_labels.append(y)
    srv_feats = torch.cat(all_feats); srv_labels = torch.cat(all_labels)
    torch.cuda.empty_cache()

    print(f"  Training server head (all 100 classes, 30ep)...")
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
    print(f"\n[5] Training per-client rejectors (+OOD data)...")
    rejectors = []
    for cid in range(5):
        torch.cuda.empty_cache()
        clients[cid].to(DEVICE)
        print(f"\n  Client {cid} ({NAMES[cid]}):")
        # Rejector data: 5K known + 5K OOD
        known_idx = [i for i in range(len(train_full)) if train_full.targets[i] < N_KNOWN]
        ood_idx = [i for i in range(len(train_full)) if train_full.targets[i] >= N_KNOWN]
        random.shuffle(known_idx); random.shuffle(ood_idx)
        rej_idx = known_idx[:5000] + ood_idx[:5000]
        rej_sub = Subset(train_full, rej_idx)

        # Pre-extract using client's own test transform
        cli_tf = client_test_tfs[cid]
        ds_rej = TripleTransformDataset(rej_sub, tf_32_train, cli_tf, tf_test)
        loader_rej = DataLoader(ds_rej, batch_size=64, shuffle=False, num_workers=2)
        all_x32, all_hidden, all_mc, all_ec = [], [], [], []
        with torch.no_grad():
            batch_i = 0
            for x32, x_cli, x_dino, y in loader_rej:
                x32, x_cli, x_dino, y = x32.to(DEVICE), x_cli.to(DEVICE), x_dino.to(DEVICE), y.to(DEVICE)
                h = clients[cid].get_hidden(x_cli)
                cp = clients[cid](x_cli).argmax(1)
                # Known: y < N_KNOWN, check cp==y; OOD: mc=0
                mc = torch.where(y < N_KNOWN, (cp == y).float(), torch.zeros_like(y, dtype=torch.float))
                sf = dino(x_dino); sp = srv_head(sf).argmax(1); ec = (sp == y).float()
                all_x32.append(x32.cpu()); all_hidden.append(h.cpu())
                all_mc.append(mc.cpu()); all_ec.append(ec.cpu())
                batch_i += 1
        torch.cuda.empty_cache()

        tds = TensorDataset(torch.cat(all_x32), torch.cat(all_hidden), torch.cat(all_mc), torch.cat(all_ec))
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
        clients[cid].cpu()
        del tds, loader, all_x32, all_hidden, all_mc, all_ec
        torch.cuda.empty_cache()

    # 6) Evaluate
    print(f"\n[6] Evaluating...")
    from sklearn.metrics import roc_auc_score

    summary_l2h = {r: [] for r in DEFERRAL_RATES}
    summary_ct = {r: [] for r in DEFERRAL_RATES}
    summary_auroc_l2h = []; summary_auroc_ct = []

    for cid in range(5):
        torch.cuda.empty_cache()
        clients[cid].to(DEVICE); rejectors[cid].to(DEVICE)
        cli_tf = client_test_tfs[cid]
        # Val scores
        ds_v = TripleTransformDataset(val_sub, tf_32_test, cli_tf, tf_test)
        loader_v = DataLoader(ds_v, batch_size=64, shuffle=False, num_workers=2)
        val_conf_list, val_rej_list = [], []
        with torch.no_grad():
            for x32, x_cli, x_dino, y in loader_v:
                x32, x_cli = x32.to(DEVICE), x_cli.to(DEVICE)
                cl = clients[cid](x_cli); conf = F.softmax(cl, dim=1).max(1).values
                h = clients[cid].get_hidden(x_cli)
                rs = F.softmax(rejectors[cid](x32, h), dim=1)[:,1]
                val_conf_list.append(conf.cpu()); val_rej_list.append(rs.cpu())
        val_conf = torch.cat(val_conf_list).numpy(); val_rej = torch.cat(val_rej_list).numpy()

        # Test scores
        ds_t = TripleTransformDataset(tst_sub, tf_32_test, cli_tf, tf_test)
        loader_t = DataLoader(ds_t, batch_size=64, shuffle=False, num_workers=2)
        tst_cp_l, tst_sp_l, tst_y_l, tst_conf_l, tst_rej_l = [], [], [], [], []
        with torch.no_grad():
            for x32, x_cli, x_dino, y in loader_t:
                x32, x_cli, x_dino, y = x32.to(DEVICE), x_cli.to(DEVICE), x_dino.to(DEVICE), y.to(DEVICE)
                cl = clients[cid](x_cli); cp = cl.argmax(1)
                conf = F.softmax(cl, dim=1).max(1).values
                sf = dino(x_dino); sp = srv_head(sf).argmax(1)
                h = clients[cid].get_hidden(x_cli)
                rs = F.softmax(rejectors[cid](x32, h), dim=1)[:,1]
                tst_cp_l.append(cp.cpu()); tst_sp_l.append(sp.cpu()); tst_y_l.append(y.cpu())
                tst_conf_l.append(conf.cpu()); tst_rej_l.append(rs.cpu())
        tst_cp = torch.cat(tst_cp_l).numpy(); tst_sp = torch.cat(tst_sp_l).numpy()
        tst_y = torch.cat(tst_y_l).numpy(); tst_conf = torch.cat(tst_conf_l).numpy()
        tst_rej = torch.cat(tst_rej_l).numpy()

        def sys_acc(defer_mask):
            correct = 0
            for i in range(len(tst_y)):
                if defer_mask[i]:
                    correct += (tst_sp[i] == tst_y[i])
                elif tst_is_known[i]:
                    correct += (tst_cp[i] == tst_y[i])
            return correct / len(tst_y)

        cli_acc = np.mean(tst_cp[tst_is_known] == tst_y[tst_is_known])
        srv_acc = np.mean(tst_sp == tst_y)

        if cid == 0:
            print(f"\n{'='*80}")
            print("RESULTS")
            print(f"{'='*80}")
        print(f"\n--- {NAMES[cid]} (hidden_dim={clients[cid].hidden_dim}) ---")
        print(f"  Client acc (known): {100*cli_acc:.1f}%  Server acc (all): {100*srv_acc:.1f}%")

        for rate in DEFERRAL_RATES:
            th_c = find_threshold(-val_conf, rate)
            defer_c = (-tst_conf) >= th_c
            th_r = find_threshold(val_rej, rate)
            defer_r = tst_rej >= th_r
            acc_c = sys_acc(defer_c); acc_r = sys_acc(defer_r)
            n = len(tst_y); k = int(n * rate / 100)
            rnd_accs = []
            for _ in range(20):
                d = np.zeros(n, dtype=bool); d[np.random.choice(n, k, replace=False)] = True
                rnd_accs.append(sys_acc(d))
            acc_rnd = np.mean(rnd_accs)
            ood_defer_c = defer_c[~tst_is_known].mean()
            ood_defer_r = defer_r[~tst_is_known].mean()
            print(f"  Rate={rate:>2}%: L2H={100*acc_r:.1f}% ConfTh={100*acc_c:.1f}% Rnd={100*acc_rnd:.1f}% | OOD→Defer: L2H={100*ood_defer_r:.0f}% CT={100*ood_defer_c:.0f}%")
            summary_l2h[rate].append(acc_r); summary_ct[rate].append(acc_c)

        ood_labels = np.concatenate([np.zeros(500), np.ones(500)])
        auroc_c = roc_auc_score(ood_labels, -tst_conf)
        auroc_r = roc_auc_score(ood_labels, tst_rej)
        print(f"  AUROC: L2H={auroc_r:.4f}  ConfTh={auroc_c:.4f}")
        summary_auroc_l2h.append(auroc_r); summary_auroc_ct.append(auroc_c)
        clients[cid].cpu(); rejectors[cid].cpu()
        torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*80}")
    print(f"SUMMARY (Average across 5 architectures)")
    print(f"{'='*80}")
    print(f"  {'Rate':>5} {'L2H':>8} {'ConfTh':>8} {'Δ(L2H-CT)':>10}")
    for rate in DEFERRAL_RATES:
        avg_l = 100*np.mean(summary_l2h[rate])
        avg_c = 100*np.mean(summary_ct[rate])
        print(f"  {rate:>4}% {avg_l:>7.2f}% {avg_c:>7.2f}% {avg_l-avg_c:>+9.2f}%")
    print(f"\n  AUROC: L2H={np.mean(summary_auroc_l2h):.4f}  ConfTh={np.mean(summary_auroc_ct):.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done.")

if __name__ == "__main__":
    main()
