"""
Confidence Calibration Analysis: Client model accuracy per confidence bin.
For each client in v8.2 setting (same CIFAR-100, different architectures).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
import numpy as np
import random

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 100

# ─── Transforms ───
tf_32 = T.Compose([T.ToTensor(), T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
tf_224 = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
tf_32_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor(),
                          T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
tf_224_train = T.Compose([T.Resize(256), T.RandomCrop(224), T.RandomHorizontalFlip(), T.ToTensor(),
                           T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# ─── Client Models (same as v8.2) ───
class ClientAlexNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc1 = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 256), nn.ReLU())
        self.fc2 = nn.Linear(256, n_classes)
    def forward(self, x): return self.fc2(self.fc1(self.features(x)))

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
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make(64, 64, 2, 1)
        self.layer2 = self._make(64, 128, 2, 2)
        self.layer3 = self._make(128, 256, 2, 2)
        self.layer4 = self._make(256, 512, 2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, n_classes)
    def _make(self, inc, outc, n, s):
        layers = [BasicBlock(inc, outc, s)]
        for _ in range(1, n): layers.append(BasicBlock(outc, outc))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        return self.fc(self.pool(self.layer4(self.layer3(self.layer2(self.layer1(x))))).flatten(1))

class ClientMobileNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1280, n_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(x)).flatten(1))

class ClientShuffleNetV2(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.shufflenet_v2_x1_0(weights=torchvision.models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(base.conv1, base.maxpool, base.stage2, base.stage3, base.stage4, base.conv5)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, n_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(x)).flatten(1))

class ClientSqueezeNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        base = torchvision.models.squeezenet1_1(weights=torchvision.models.SqueezeNet1_1_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(512, n_classes)
    def forward(self, x): return self.classifier(self.pool(self.features(x)).flatten(1))

# ─── Training ───
class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform):
        self.subset = subset; self.transform = transform
    def __len__(self): return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

def train_model(model, train_sub, tf_train, epochs, lr, backbone_lr=None):
    model.to(DEVICE)
    ds = TransformSubset(train_sub, tf_train)
    loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=2)
    if backbone_lr and hasattr(model, 'features'):
        opt = torch.optim.Adam([
            {'params': model.features.parameters(), 'lr': backbone_lr},
            {'params': (p for n, p in model.named_parameters() if 'features' not in n), 'lr': lr}
        ], weight_decay=1e-4)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    model.train()
    for ep in range(epochs):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(); crit(model(x), y).backward(); opt.step()
        sched.step()
        if (ep+1) % 20 == 0 or ep == 0:
            model.eval()
            correct = total = 0
            with torch.no_grad():
                for x, y in loader:
                    correct += (model(x.to(DEVICE)).argmax(1) == y.to(DEVICE)).sum().item()
                    total += len(y)
            print(f"    Epoch {ep+1}/{epochs}  TrainAcc={100*correct/total:.1f}%")
            model.train()
    model.eval()
    return model

def main():
    print("=" * 80)
    print("Confidence Calibration Analysis")
    print("=" * 80)

    # Load CIFAR-100
    train_full = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=None)
    test_full = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=None)

    # Same splits as v8.2
    idx = list(range(len(train_full)))
    random.shuffle(idx)
    client_train_idx = [idx[i*5000:(i+1)*5000] for i in range(5)]
    te_idx = list(range(len(test_full)))
    random.shuffle(te_idx)
    tst_idx = te_idx[500:]  # skip first 500 (val)

    CLIENT_NAMES = ["AlexNet", "ResNet-18", "MobileNetV2", "ShuffleNetV2", "SqueezeNet"]
    CLIENT_CLASSES = [ClientAlexNet, ClientResNet18, ClientMobileNetV2, ClientShuffleNetV2, ClientSqueezeNet]
    CLIENT_TRAIN_TFS = [tf_32_train, tf_32_train, tf_224_train, tf_224_train, tf_224_train]
    CLIENT_TEST_TFS = [tf_32, tf_32, tf_224, tf_224, tf_224]
    CLIENT_EPOCHS = [100, 100, 30, 30, 30]
    CLIENT_LR = [1e-3, 1e-3, 1e-3, 1e-3, 1e-3]
    CLIENT_BACKBONE_LR = [None, None, 1e-4, 1e-4, 1e-4]

    tst_subset = Subset(test_full, tst_idx)

    # Train and analyze each client
    BINS = 10
    bin_edges = np.linspace(0, 1, BINS + 1)  # [0, 0.1, 0.2, ..., 1.0]

    all_results = {}

    for cid in range(5):
        print(f"\n--- Client {cid}: {CLIENT_NAMES[cid]} ---")
        model = CLIENT_CLASSES[cid](N_CLASSES)
        model = train_model(model, Subset(train_full, client_train_idx[cid]),
                            CLIENT_TRAIN_TFS[cid], CLIENT_EPOCHS[cid],
                            CLIENT_LR[cid], CLIENT_BACKBONE_LR[cid])

        # Evaluate on test set
        ds = TransformSubset(tst_subset, CLIENT_TEST_TFS[cid])
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)
        all_conf, all_correct = [], []
        with torch.no_grad():
            for x, y in loader:
                logits = model(x.to(DEVICE))
                probs = F.softmax(logits, dim=1)
                max_conf, preds = probs.max(1)
                correct = (preds == y.to(DEVICE)).cpu().numpy()
                all_conf.append(max_conf.cpu().numpy())
                all_correct.append(correct)
        
        confs = np.concatenate(all_conf)
        corrects = np.concatenate(all_correct)
        
        overall_acc = corrects.mean()
        print(f"  Overall acc: {100*overall_acc:.2f}%")
        print(f"  Confidence stats: min={confs.min():.4f}, max={confs.max():.4f}, mean={confs.mean():.4f}, median={np.median(confs):.4f}")
        
        print(f"\n  {'Bin':>12} {'Count':>6} {'%Data':>6} {'Accuracy':>9} {'AvgConf':>8}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*9} {'-'*8}")
        
        bin_results = []
        for i in range(BINS):
            lo, hi = bin_edges[i], bin_edges[i+1]
            if i == 0:
                mask = (confs >= lo) & (confs <= hi)
            else:
                mask = (confs > lo) & (confs <= hi)
            count = mask.sum()
            if count > 0:
                acc = corrects[mask].mean()
                avg_conf = confs[mask].mean()
            else:
                acc = 0; avg_conf = 0
            pct = 100 * count / len(confs)
            bin_results.append((lo, hi, count, pct, acc, avg_conf))
            print(f"  ({lo:.1f}, {hi:.1f}] {count:>6} {pct:>5.1f}% {100*acc:>8.2f}% {avg_conf:>8.4f}")
        
        all_results[cid] = (confs, corrects, bin_results)

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Accuracy per confidence bin across all clients")
    print("=" * 80)
    
    header = f"  {'Bin':>12}"
    for cid in range(5):
        header += f" {CLIENT_NAMES[cid]:>12}"
    print(header)
    print("  " + "-" * (12 + 13 * 5))
    
    for i in range(BINS):
        lo, hi = bin_edges[i], bin_edges[i+1]
        row = f"  ({lo:.1f}, {hi:.1f}]"
        for cid in range(5):
            _, _, bins = all_results[cid]
            count = bins[i][2]
            acc = bins[i][4]
            if count > 0:
                row += f" {100*acc:>5.1f}%({count:>3})"
            else:
                row += f"     --(  0)"
        print(row)
    
    # Also print: what % of samples fall in each confidence range
    print(f"\n  {'Bin':>12}", end="")
    for cid in range(5):
        print(f" {CLIENT_NAMES[cid]:>12}", end="")
    print(" (% of data)")
    print("  " + "-" * (12 + 13 * 5))
    for i in range(BINS):
        lo, hi = bin_edges[i], bin_edges[i+1]
        row = f"  ({lo:.1f}, {hi:.1f}]"
        for cid in range(5):
            _, _, bins = all_results[cid]
            pct = bins[i][3]
            row += f" {pct:>11.1f}%"
        print(row)

    print("\nDone.")

if __name__ == "__main__":
    main()
