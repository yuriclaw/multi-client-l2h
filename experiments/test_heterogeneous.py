"""Heterogeneous clients: different datasets, frozen server backbone + per-client head.

Clients:
  0: CIFAR-10 (10 classes, natural objects)
  1: SVHN (10 classes, house numbers)
  2: FashionMNIST (10 classes, clothing)
  3: STL-10 (10 classes, similar to CIFAR but different distribution)
  4: GTSRB (43 classes, traffic signs)

Server: ResNet-50 pretrained on ImageNet
  - Backbone frozen
  - Per-client linear classification head trained on frozen representations
  - Then test: baseline vs universal LoRA vs per-client LoRA
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm
from peft import LoraConfig, get_peft_model
import torchvision
import torchvision.transforms as T

BATCH_SIZE = 128
LORA_RANK = 16
LORA_LR = 1e-4
LORA_EPOCHS = 20
HEAD_EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))

LORA_TARGETS = ["layer3.0.conv2", "layer3.1.conv2", "layer3.2.conv2",
                "layer3.3.conv2", "layer3.4.conv2", "layer3.5.conv2",
                "layer4.0.conv2", "layer4.1.conv2", "layer4.2.conv2"]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

# ── Dataset loaders ──

def get_transform(grayscale=False):
    """Standard transform: resize to 32x32, normalize."""
    transforms = []
    if grayscale:
        transforms.append(T.Grayscale(num_output_channels=3))
    transforms.extend([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    return T.Compose(transforms)

def load_client_data(client_id, data_dir, max_train=8000, max_test=2000):
    """Load train/test data for each client. Returns (train_ds, test_ds, num_classes)."""
    
    if client_id == 0:  # CIFAR-10
        tf = get_transform()
        train = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=tf)
        test = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=tf)
        nc = 10
        name = "CIFAR-10"
        
    elif client_id == 1:  # SVHN
        tf = get_transform()
        train = torchvision.datasets.SVHN(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.SVHN(data_dir, split='test', download=True, transform=tf)
        nc = 10
        name = "SVHN"
        
    elif client_id == 2:  # FashionMNIST
        tf = get_transform(grayscale=True)
        train = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
        test = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
        nc = 10
        name = "FashionMNIST"
        
    elif client_id == 3:  # STL-10
        tf = get_transform()
        train = torchvision.datasets.STL10(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.STL10(data_dir, split='test', download=True, transform=tf)
        nc = 10
        name = "STL-10"
        
    elif client_id == 4:  # GTSRB
        tf = get_transform()
        train = torchvision.datasets.GTSRB(data_dir, split='train', download=True, transform=tf)
        test = torchvision.datasets.GTSRB(data_dir, split='test', download=True, transform=tf)
        nc = 43
        name = "GTSRB"
    
    # Subsample
    if len(train) > max_train:
        indices = random.sample(range(len(train)), max_train)
        train = Subset(train, indices)
    if len(test) > max_test:
        indices = random.sample(range(len(test)), max_test)
        test = Subset(test, indices)
    
    print(f"  Client {client_id} ({name}): train={len(train)}, test={len(test)}, classes={nc}")
    return train, test, nc, name


class FeatureExtractor(nn.Module):
    """Extract intermediate features from ResNet-50 (before FC)."""
    def __init__(self, backbone, upsample):
        super().__init__()
        self.backbone = backbone
        self.upsample = upsample
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)  # [B, 2048]
        return x


class ClientHead(nn.Module):
    """Per-client classification head."""
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, features):
        return self.fc(features)


class ServerWithHead(nn.Module):
    """Server backbone + per-client head, for LoRA training."""
    def __init__(self, backbone, upsample, head):
        super().__init__()
        self.backbone = backbone
        self.upsample = upsample
        self.head = head
    
    def forward(self, x):
        x = self.upsample(x)
        x = self.backbone.forward_features(x)
        x = self.backbone.global_pool(x)
        return self.head(x)


def extract_features(extractor, loader, device):
    """Extract features from a dataset."""
    feats, labels = [], []
    extractor.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            f = extractor(xb).cpu()
            feats.append(f)
            labels.append(yb)
    return torch.cat(feats), torch.cat(labels)


def train_head(head, train_feats, train_labels, epochs, device):
    """Train a linear head on extracted features."""
    ds = TensorDataset(train_feats, train_labels)
    loader = DataLoader(ds, batch_size=256, shuffle=True)
    opt = Adam(head.parameters(), lr=1e-3)
    head.train()
    for ep in range(epochs):
        correct = total = 0
        for fb, yb in loader:
            fb, yb = fb.to(device), yb.to(device)
            logits = head(fb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if (ep+1) % 5 == 0:
            print(f"      Head epoch {ep+1}/{epochs}: acc={correct/total:.4f}")
    head.eval()
    return head


def eval_model(model, test_loader, device, bs_override=64):
    """Evaluate a full model (backbone+head) on test data."""
    correct = total = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
    return correct / total


def main():
    set_seed(42)
    print(f"Device: {DEVICE}")
    
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(DEVICE)
    
    # ── Load all client data ──
    print("\n=== Loading client data ===")
    client_data = {}
    for cid in range(5):
        train_ds, test_ds, nc, name = load_client_data(cid, DATA_DIR)
        client_data[cid] = {
            "train": train_ds, "test": test_ds, "nc": nc, "name": name,
            "train_loader": DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2),
            "test_loader": DataLoader(test_ds, batch_size=64, num_workers=2),
        }
    
    # ── Build server backbone ──
    print("\n=== Building server (ResNet-50, frozen) ===")
    backbone = timm.create_model("resnet50", pretrained=True, num_classes=0)  # no head
    for p in backbone.parameters():
        p.requires_grad = False
    backbone = backbone.to(DEVICE)
    backbone.eval()
    
    feat_extractor = FeatureExtractor(backbone, upsample)
    feat_dim = 2048  # ResNet-50 feature dim
    
    # ── Phase 1: Train per-client heads on frozen features ──
    print("\n=== Phase 1: Train per-client classification heads ===")
    heads = {}
    baselines = {}
    
    for cid in range(5):
        cd = client_data[cid]
        print(f"\n  Client {cid} ({cd['name']}):")
        
        # Extract features
        print(f"    Extracting features...")
        train_feats, train_labels = extract_features(feat_extractor, cd["train_loader"], DEVICE)
        test_feats, test_labels = extract_features(feat_extractor, cd["test_loader"], DEVICE)
        
        # Train head
        head = ClientHead(feat_dim, cd["nc"]).to(DEVICE)
        head = train_head(head, train_feats, train_labels, HEAD_EPOCHS, DEVICE)
        
        # Eval baseline
        head.eval()
        with torch.no_grad():
            logits = head(test_feats.to(DEVICE))
            acc = (logits.argmax(1) == test_labels.to(DEVICE)).float().mean().item()
        
        baselines[cid] = acc
        heads[cid] = head.cpu()
        print(f"    Baseline acc: {acc:.4f}")
        
        # Store features for later
        client_data[cid]["train_feats"] = train_feats
        client_data[cid]["train_labels"] = train_labels
        client_data[cid]["test_feats"] = test_feats
        client_data[cid]["test_labels"] = test_labels
    
    # Save backbone state
    backbone_state = {k: v.cpu().clone() for k, v in backbone.state_dict().items()}
    
    # ── Phase 2: Per-client LoRA ──
    print("\n=== Phase 2: Per-client LoRA (one adapter per client) ===")
    personal_acc = {}
    
    for cid in range(5):
        cd = client_data[cid]
        print(f"\n  Client {cid} ({cd['name']}):")
        
        # Fresh backbone
        bb = timm.create_model("resnet50", pretrained=True, num_classes=0)
        for p in bb.parameters():
            p.requires_grad = False
        bb.load_state_dict(backbone_state)
        
        # Build full model with head
        head = heads[cid].to(DEVICE)
        # We need a full model for LoRA: backbone that outputs logits
        full_model = timm.create_model("resnet50", pretrained=True, num_classes=cd["nc"])
        for p in full_model.parameters():
            p.requires_grad = False
        # Copy backbone weights
        bb_sd = backbone_state
        full_sd = full_model.state_dict()
        for k in bb_sd:
            if k in full_sd:
                full_sd[k] = bb_sd[k].clone()
        # Copy head weights
        head_sd = head.state_dict()
        full_sd["fc.weight"] = head_sd["fc.weight"].clone()
        full_sd["fc.bias"] = head_sd["fc.bias"].clone()
        full_model.load_state_dict(full_sd)
        
        # LoRA
        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_RANK * 2,
            target_modules=LORA_TARGETS, lora_dropout=0.05, bias="none",
        )
        lora_model = get_peft_model(full_model, lora_config).to(DEVICE)
        
        # Wrap with upsample
        class LoraWithUpsample(nn.Module):
            def __init__(self, lora_m, up):
                super().__init__()
                self.lora_m = lora_m
                self.up = up
            def forward(self, x):
                return self.lora_m(self.up(x))
        
        model = LoraWithUpsample(lora_model, upsample)
        
        # Train
        optimizer = Adam(lora_model.parameters(), lr=LORA_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LORA_EPOCHS)
        
        model.train()
        for ep in range(LORA_EPOCHS):
            correct = total = 0
            for xb, yb in cd["train_loader"]:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
            scheduler.step()
            if (ep+1) % 5 == 0:
                print(f"    Epoch {ep+1}/{LORA_EPOCHS}: train_acc={correct/total:.4f}")
        
        # Eval
        acc = eval_model(model, cd["test_loader"], DEVICE)
        personal_acc[cid] = acc
        print(f"    Per-client LoRA acc: {acc:.4f} (baseline: {baselines[cid]:.4f}, delta: {acc-baselines[cid]:+.4f})")
        
        del model, lora_model, full_model
        torch.cuda.empty_cache()
    
    # ── Phase 3: Universal LoRA (all client data pooled) ──
    print("\n=== Phase 3: Universal LoRA (all clients pooled) ===")
    
    # For universal, we need a common label space. Since datasets have different classes,
    # we create a unified label space: offset labels by cumulative class count
    # Client 0: CIFAR-10 (0-9), Client 1: SVHN (10-19), Client 2: FashionMNIST (20-29),
    # Client 3: STL-10 (30-39), Client 4: GTSRB (40-82)
    
    class OffsetDataset(torch.utils.data.Dataset):
        def __init__(self, ds, offset):
            self.ds = ds
            self.offset = offset
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            x, y = self.ds[idx]
            return x, y + self.offset
    
    total_classes = 0
    offset_datasets = []
    client_offsets = {}
    for cid in range(5):
        cd = client_data[cid]
        client_offsets[cid] = total_classes
        offset_datasets.append(OffsetDataset(cd["train"], total_classes))
        total_classes += cd["nc"]
    
    print(f"  Total unified classes: {total_classes}")
    universal_ds = ConcatDataset(offset_datasets)
    universal_loader = DataLoader(universal_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    # Build universal model
    full_model = timm.create_model("resnet50", pretrained=True, num_classes=total_classes)
    for p in full_model.parameters():
        p.requires_grad = False
    # Copy backbone
    full_sd = full_model.state_dict()
    for k in backbone_state:
        if k in full_sd:
            full_sd[k] = backbone_state[k].clone()
    full_model.load_state_dict(full_sd)
    
    # Unfreeze fc for universal head, train it first
    full_model.fc.weight.requires_grad = True
    full_model.fc.bias.requires_grad = True
    full_model = full_model.to(DEVICE)
    
    print("  Training universal head...")
    head_opt = Adam([full_model.fc.weight, full_model.fc.bias], lr=1e-3)
    full_model.train()
    for ep in range(HEAD_EPOCHS):
        correct = total = 0
        for xb, yb in universal_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = full_model(upsample(xb))
            loss = F.cross_entropy(logits, yb)
            head_opt.zero_grad(); loss.backward(); head_opt.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        if (ep+1) % 5 == 0:
            print(f"    Head epoch {ep+1}/{HEAD_EPOCHS}: acc={correct/total:.4f}")
    
    full_model.fc.weight.requires_grad = False
    full_model.fc.bias.requires_grad = False
    
    # Now add LoRA
    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_RANK * 2,
        target_modules=LORA_TARGETS, lora_dropout=0.05, bias="none",
    )
    uni_lora = get_peft_model(full_model, lora_config).to(DEVICE)
    
    optimizer = Adam(uni_lora.parameters(), lr=LORA_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LORA_EPOCHS)
    
    print("  Training universal LoRA...")
    uni_lora.train()
    for ep in range(LORA_EPOCHS):
        correct = total = 0
        for xb, yb in universal_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = uni_lora(upsample(xb))
            loss = F.cross_entropy(logits, yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            correct += (logits.argmax(1) == yb).sum().item()
            total += yb.size(0)
        scheduler.step()
        if (ep+1) % 5 == 0:
            print(f"    Epoch {ep+1}/{LORA_EPOCHS}: train_acc={correct/total:.4f}")
    
    # Eval universal on each client
    uni_lora.eval()
    universal_acc = {}
    for cid in range(5):
        cd = client_data[cid]
        offset = client_offsets[cid]
        correct = total = 0
        with torch.no_grad():
            for xb, yb in cd["test_loader"]:
                xb = xb.to(DEVICE)
                yb_offset = (yb + offset).to(DEVICE)
                logits = uni_lora(upsample(xb))
                correct += (logits.argmax(1) == yb_offset).sum().item()
                total += yb.size(0)
        universal_acc[cid] = correct / total
        print(f"    Client {cid} ({cd['name']}): universal acc = {universal_acc[cid]:.4f}")
    
    del uni_lora, full_model
    torch.cuda.empty_cache()
    
    # ── Summary ──
    print("\n" + "="*80)
    print("COMPARISON: Baseline (frozen) vs Universal LoRA vs Per-client LoRA")
    print("="*80)
    print(f"{'Client':>8s} {'Dataset':>15s} {'Baseline':>10s} {'Universal':>10s} {'Personal':>10s} {'U-B':>10s} {'P-B':>10s} {'P-U':>10s} {'Winner':>10s}")
    print("-"*95)
    
    for cid in range(5):
        cd = client_data[cid]
        b = baselines[cid]
        u = universal_acc[cid]
        p = personal_acc[cid]
        winner = "Personal" if p > u + 0.005 else ("Universal" if u > p + 0.005 else "Tie")
        print(f"{cid:>8d} {cd['name']:>15s} {b:10.4f} {u:10.4f} {p:10.4f} {u-b:+10.4f} {p-b:+10.4f} {p-u:+10.4f} {winner:>10s}")
    
    # Averages
    avg_b = np.mean([baselines[i] for i in range(5)])
    avg_u = np.mean([universal_acc[i] for i in range(5)])
    avg_p = np.mean([personal_acc[i] for i in range(5)])
    print("-"*95)
    print(f"{'AVG':>8s} {'':>15s} {avg_b:10.4f} {avg_u:10.4f} {avg_p:10.4f} {avg_u-avg_b:+10.4f} {avg_p-avg_b:+10.4f} {avg_p-avg_u:+10.4f}")
    
    print("\n✅ Heterogeneous client test complete!")

if __name__ == "__main__":
    main()
