#!/usr/bin/env python3
"""
OOD Confidence Analysis: Client trained on CIFAR-10, tested on CIFAR-100.
=========================================================================
Question: When OOD samples (unseen classes) arrive, does client softmax
confidence distinguish them from in-distribution samples?

Setup:
- Client trained on CIFAR-10 (10 classes)
- Test on CIFAR-100 (100 classes): 10 overlap with CIFAR-10, 90 are new
- Analyze confidence distribution for: in-dist vs OOD samples
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision, torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np, random, time

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 classes mapped to CIFAR-100 superclass/fine labels
# CIFAR-100 has 100 fine classes. CIFAR-10 classes roughly correspond to:
# We'll use a simpler approach: train on first 10 classes of CIFAR-100,
# test on all 100 classes. This ensures exact label space overlap.

tf_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(),
                       T.RandAugment(num_ops=2, magnitude=9),
                       T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])
tf_test = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                      T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))])

# Client: MobileNetV2 pretrained
class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(1280, n_classes)
    def get_hidden(self, x): return self.pool(self.features(x)).flatten(1)
    def forward(self, x): return self.classifier(self.drop(self.get_hidden(x)))

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1-lam) * x[idx], y, y[idx], lam

def main():
    t0 = time.time()
    print("=" * 80)
    print("OOD Confidence Analysis")
    print("=" * 80)

    # Load CIFAR-100 (use as unified dataset)
    print("\n[1] Loading CIFAR-100...")
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    # Split: first 10 classes = "known", remaining 90 = "unknown/OOD"
    N_KNOWN = 10
    known_classes = list(range(N_KNOWN))  # classes 0-9
    ood_classes = list(range(N_KNOWN, 100))  # classes 10-99

    # Build train set: only known classes
    train_known_idx = [i for i in range(len(train_full)) if train_full.targets[i] < N_KNOWN]
    # Limit to 5000 samples
    random.shuffle(train_known_idx)
    train_known_idx = train_known_idx[:5000]
    train_sub = Subset(train_full, train_known_idx)
    print(f"  Train: {len(train_known_idx)} samples from {N_KNOWN} known classes")

    # Build test sets
    test_known_idx = [i for i in range(len(test_full)) if test_full.targets[i] < N_KNOWN]
    test_ood_idx = [i for i in range(len(test_full)) if test_full.targets[i] >= N_KNOWN]
    # Limit each to 1000
    random.shuffle(test_known_idx); test_known_idx = test_known_idx[:1000]
    random.shuffle(test_ood_idx); test_ood_idx = test_ood_idx[:1000]
    test_known_sub = Subset(test_full, test_known_idx)
    test_ood_sub = Subset(test_full, test_ood_idx)
    print(f"  Test known: {len(test_known_idx)} samples (classes 0-{N_KNOWN-1})")
    print(f"  Test OOD: {len(test_ood_idx)} samples (classes {N_KNOWN}-99)")

    # Train client on known classes only
    print(f"\n[2] Training MobileNetV2 on {N_KNOWN} known classes (60ep)...")
    model = ClientMobileNetV2(N_KNOWN).to(DEVICE)
    ds = TransformSubset(train_sub, tf_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.Adam([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': (p for n,p in model.named_parameters() if 'features' not in n), 'lr': 1e-3}
    ], weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    model.train()
    for ep in range(60):
        tc = tt = 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            mx, ya, yb, lam = mixup_data(x, y)
            out = model(mx)
            loss = lam * crit(out, ya) + (1-lam) * crit(out, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                tc += (model(x).argmax(1)==y).sum().item(); tt += len(y)
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            print(f"  Epoch {ep+1}/60  TrainAcc={100*tc/tt:.1f}%")
    model.eval()

    # Test on known classes
    ds_k = TransformSubset(test_known_sub, tf_test)
    loader_k = DataLoader(ds_k, batch_size=256, num_workers=2)
    k_confs, k_correct = [], []
    with torch.no_grad():
        for x, y in loader_k:
            probs = F.softmax(model(x.to(DEVICE)), dim=1)
            mc, pred = probs.max(1)
            k_confs.append(mc.cpu().numpy())
            k_correct.append((pred.cpu() == y).numpy())
    k_confs = np.concatenate(k_confs); k_correct = np.concatenate(k_correct)

    # Test on OOD classes
    ds_o = TransformSubset(test_ood_sub, tf_test)
    loader_o = DataLoader(ds_o, batch_size=256, num_workers=2)
    o_confs, o_preds = [], []
    with torch.no_grad():
        for x, y in loader_o:
            probs = F.softmax(model(x.to(DEVICE)), dim=1)
            mc, pred = probs.max(1)
            o_confs.append(mc.cpu().numpy())
            o_preds.append(pred.cpu().numpy())
    o_confs = np.concatenate(o_confs)

    # Results
    print(f"\n{'='*80}")
    print("RESULTS")
    print(f"{'='*80}")

    print(f"\n--- Known Classes (in-distribution) ---")
    print(f"  Accuracy: {100*k_correct.mean():.1f}%")
    print(f"  Confidence: mean={k_confs.mean():.4f}  median={np.median(k_confs):.4f}  min={k_confs.min():.4f}  max={k_confs.max():.4f}")

    print(f"\n--- OOD Classes (unseen) ---")
    print(f"  Confidence: mean={o_confs.mean():.4f}  median={np.median(o_confs):.4f}  min={o_confs.min():.4f}  max={o_confs.max():.4f}")

    # Bin analysis
    bins = np.linspace(0, 1, 11)
    print(f"\n--- Confidence Distribution ---")
    print(f"  {'Bin':>12} {'Known(%data)':>13} {'Known(acc)':>11} {'OOD(%data)':>11}")
    print(f"  {'-'*12} {'-'*13} {'-'*11} {'-'*11}")
    for i in range(10):
        lo, hi = bins[i], bins[i+1]
        mask_k = (k_confs > lo) & (k_confs <= hi) if i > 0 else (k_confs >= lo) & (k_confs <= hi)
        mask_o = (o_confs > lo) & (o_confs <= hi) if i > 0 else (o_confs >= lo) & (o_confs <= hi)
        pct_k = 100 * mask_k.sum() / len(k_confs)
        pct_o = 100 * mask_o.sum() / len(o_confs)
        acc_k = 100 * k_correct[mask_k].mean() if mask_k.sum() > 0 else 0
        print(f"  ({lo:.1f},{hi:.1f}] {pct_k:>8.1f}%     {acc_k:>6.1f}%     {pct_o:>8.1f}%")

    # Key metric: if we use confidence threshold, how well can we separate known vs OOD?
    print(f"\n--- Separation Analysis ---")
    for thresh in [0.3, 0.5, 0.7, 0.9, 0.95, 0.99]:
        known_above = (k_confs >= thresh).mean()
        ood_above = (o_confs >= thresh).mean()
        print(f"  Threshold {thresh:.2f}: Known above={100*known_above:.1f}%  OOD above={100*ood_above:.1f}%  "
              f"→ If defer below thresh: Known deferred={100*(1-known_above):.1f}%  OOD deferred={100*(1-ood_above):.1f}%")

    # AUROC for separating known vs OOD using confidence
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(k_confs)), np.zeros(len(o_confs))])
    scores = np.concatenate([k_confs, o_confs])
    auroc = roc_auc_score(labels, scores)
    print(f"\n  AUROC (confidence as OOD detector): {auroc:.4f}")

    # Also test with max logit (before softmax) — often better for OOD
    print(f"\n--- Max Logit Analysis (alternative to softmax conf) ---")
    k_logits, o_logits = [], []
    with torch.no_grad():
        for x, _ in DataLoader(ds_k, batch_size=256, num_workers=2):
            k_logits.append(model(x.to(DEVICE)).max(1).values.cpu().numpy())
        for x, _ in DataLoader(ds_o, batch_size=256, num_workers=2):
            o_logits.append(model(x.to(DEVICE)).max(1).values.cpu().numpy())
    k_logits = np.concatenate(k_logits); o_logits = np.concatenate(o_logits)
    print(f"  Known max logit: mean={k_logits.mean():.2f}  OOD max logit: mean={o_logits.mean():.2f}")
    logit_scores = np.concatenate([k_logits, o_logits])
    auroc_logit = roc_auc_score(labels, logit_scores)
    print(f"  AUROC (max logit as OOD detector): {auroc_logit:.4f}")

    # Energy score
    k_energy, o_energy = [], []
    with torch.no_grad():
        for x, _ in DataLoader(ds_k, batch_size=256, num_workers=2):
            logits = model(x.to(DEVICE))
            k_energy.append(torch.logsumexp(logits, dim=1).cpu().numpy())
        for x, _ in DataLoader(ds_o, batch_size=256, num_workers=2):
            logits = model(x.to(DEVICE))
            o_energy.append(torch.logsumexp(logits, dim=1).cpu().numpy())
    k_energy = np.concatenate(k_energy); o_energy = np.concatenate(o_energy)
    print(f"  Known energy: mean={k_energy.mean():.2f}  OOD energy: mean={o_energy.mean():.2f}")
    energy_scores = np.concatenate([k_energy, o_energy])
    auroc_energy = roc_auc_score(labels, energy_scores)
    print(f"  AUROC (energy as OOD detector): {auroc_energy:.4f}")

    print(f"\nTotal time: {(time.time()-t0)/60:.1f} min")
    print("Done.")

if __name__ == "__main__":
    main()
