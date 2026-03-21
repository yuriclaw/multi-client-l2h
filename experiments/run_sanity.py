"""Sanity check: 2 clients, few epochs, verify everything works end-to-end.

Usage on sarwate:
    conda activate l2h-b
    cd ~/multi-client-l2h
    python experiments/run_sanity.py
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import timm
from peft import LoraConfig, get_peft_model

# ── Config ──────────────────────────────────────────────────────────
NUM_CLIENTS = 5
NUM_CLASSES = 10
BATCH_SIZE = 128
ROUNDS = 20
ADAPTER_EPOCHS = 3
REJECTOR_EPOCHS = 3
LORA_RANK = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.path.expanduser("~/data")
CORRUPTIONS = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "fog"]
SEVERITY = 3

random.seed(42); np.random.seed(42); torch.manual_seed(42)


# ── Data ────────────────────────────────────────────────────────────
def load_cifar10c(data_dir, corruption, severity):
    """Load one corruption type from CIFAR-10-C."""
    folder = os.path.join(data_dir, "CIFAR-10-C")
    images = np.load(os.path.join(folder, f"{corruption}.npy"))
    labels = np.load(os.path.join(folder, "labels.npy"))
    start = (severity - 1) * 10000
    imgs = images[start:start+10000].astype(np.float32) / 255.0
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # NHWC -> NCHW
    # Normalize
    mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1,3,1,1)
    std = np.array([0.2470, 0.2435, 0.2616]).reshape(1,3,1,1)
    imgs = (imgs - mean) / std
    return torch.tensor(imgs, dtype=torch.float32), torch.tensor(labels[start:start+10000], dtype=torch.long)


def load_clean_cifar10(data_dir):
    """Load clean CIFAR-10 for server baseline eval."""
    import torchvision
    import torchvision.transforms as T
    transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    return ds


# ── Models ──────────────────────────────────────────────────────────
class LeNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Rejector(nn.Module):
    def __init__(self, input_dim=3072, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
    def forward(self, x):
        return torch.sigmoid(self.net(x.reshape(x.size(0), -1))).squeeze(-1)


# ── Pre-train client ───────────────────────────────────────────────
def pretrain_client(model, loader, epochs=10, lr=1e-3):
    """Train client on its corruption data."""
    model.train()
    opt = Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        total, correct = 0, 0
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
        if (ep+1) % 5 == 0:
            print(f"    Pretrain epoch {ep+1}/{epochs}: acc={correct/total:.4f}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ── Build server with LoRA ─────────────────────────────────────────
class ServerWrapper(nn.Module):
    """Wraps a timm model with input upsampling for CIFAR (32→224)."""
    def __init__(self, backbone, upsample_size=224):
        super().__init__()
        self.upsample = nn.Upsample(size=(upsample_size, upsample_size), mode='bilinear', align_corners=False)
        self.backbone = backbone
    def forward(self, x):
        x = self.upsample(x)
        return self.backbone(x)

def build_server(num_classes=10, lora_rank=4):
    """Create frozen ResNet18 + per-client LoRA adapters with input upsampling."""
    base = timm.create_model("resnet18", pretrained=True, num_classes=num_classes)
    for p in base.parameters():
        p.requires_grad = False
    
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["layer3.0.conv1", "layer3.0.conv2", "layer3.1.conv1", "layer3.1.conv2",
                        "layer4.0.conv1", "layer4.0.conv2", "layer4.1.conv1", "layer4.1.conv2",
                        "fc"],  # deeper conv layers + head
        bias="none",
    )
    model = get_peft_model(base, lora_config, adapter_name="default")
    return model, lora_config


# ── Main ───────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download CIFAR-10-C if needed
    cifar_c_dir = os.path.join(DATA_DIR, "CIFAR-10-C")
    if not os.path.exists(cifar_c_dir):
        print("Downloading CIFAR-10-C...")
        import urllib.request, tarfile
        url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
        tar_path = os.path.join(DATA_DIR, "CIFAR-10-C.tar")
        urllib.request.urlretrieve(url, tar_path)
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA_DIR)
        os.remove(tar_path)
        print("Done.")

    # ── Phase 1: Pre-train clients on their corruption ──
    print("\n=== Phase 1: Pre-training clients ===")
    clients = {}
    client_data = {}
    for i, corr in enumerate(CORRUPTIONS):
        cid = f"client_{i}"
        print(f"\n  [{cid}] corruption={corr}")
        imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
        ds = TensorDataset(imgs, labels)
        n_train = int(0.8 * len(ds))
        train_ds, val_ds = random_split(ds, [n_train, len(ds) - n_train])
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
        
        model = LeNet(NUM_CLASSES).to(DEVICE)
        model = pretrain_client(model, train_loader, epochs=15)
        
        # Eval on own data
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                correct += (model(x).argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"    [{cid}] val acc on own corruption: {correct/total:.4f}")
        
        clients[cid] = model
        client_data[cid] = {"train": train_loader, "val": val_loader, "imgs": imgs, "labels": labels}

    # ── Phase 2: Build server + LoRA adapters ──
    print("\n=== Phase 2: Building server with LoRA ===")
    server_peft, lora_config = build_server(NUM_CLASSES, LORA_RANK)
    # Wrap with upsampling for CIFAR 32x32 → 224x224
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(DEVICE)
    server_peft = server_peft.to(DEVICE)
    
    # Helper: forward with upsample
    def server_forward(x):
        return server_peft(upsample(x))
    
    server_base = server_peft  # keep reference for adapter switching
    
    # Linear probe: train the classification head on clean + corrupted CIFAR-10
    print("\nLinear probing server head on clean + corrupted CIFAR-10...")
    import torchvision, torchvision.transforms as T
    from torch.utils.data import ConcatDataset
    probe_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    clean_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=probe_transform)
    
    # Also add corrupted data from ALL corruption types for a robust head
    # Convert clean CIFAR-10 to tensors first
    print("  Loading clean CIFAR-10 as tensors...")
    clean_loader_tmp = DataLoader(clean_ds, batch_size=1000, num_workers=2)
    clean_imgs, clean_labels = [], []
    for xb, yb in clean_loader_tmp:
        clean_imgs.append(xb); clean_labels.append(yb)
    clean_tensor_ds = TensorDataset(torch.cat(clean_imgs), torch.cat(clean_labels))
    
    corrupt_datasets = [clean_tensor_ds]
    all_probe_corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "fog",
                              "motion_blur", "brightness", "contrast", "frost", "snow"]
    for corr in all_probe_corruptions:
        try:
            imgs, labels = load_cifar10c(DATA_DIR, corr, 3)
            corrupt_datasets.append(TensorDataset(imgs[:5000], labels[:5000]))
        except Exception as e:
            print(f"  Skipping {corr}: {e}")
    
    probe_ds = ConcatDataset(corrupt_datasets)
    print(f"  Probe dataset size: {len(probe_ds)} (clean + {len(corrupt_datasets)-1} corruption types)")
    probe_loader = DataLoader(probe_ds, batch_size=256, shuffle=True, num_workers=2)
    
    # Unfreeze only the final FC layer for probing
    # Access through PEFT wrapper
    base_model = server_peft.get_base_model()
    for name, p in base_model.named_parameters():
        if 'fc' in name:  # ResNet18 final FC
            p.requires_grad = True
    
    probe_opt = Adam([p for p in base_model.parameters() if p.requires_grad], lr=1e-3)
    server_peft.train()
    for ep in range(10):
        correct, total = 0, 0
        for x_b, y_b in probe_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            logits = server_forward(x_b)
            loss = F.cross_entropy(logits, y_b)
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
        if (ep+1) % 2 == 0:
            print(f"  Probe epoch {ep+1}/10: acc={correct/total:.4f}")
    
    # Re-freeze the head
    for p in base_model.parameters():
        p.requires_grad = False
    server_peft.eval()
    print("Linear probe done. Head frozen again.")
    
    # Add per-client adapters
    for cid in clients:
        server_base.add_adapter(cid, lora_config)
    
    # Eval server baseline (no LoRA, just pretrained ResNet)
    print("\nServer baseline (frozen ResNet18, no LoRA):")
    server_base.set_adapter("default")
    server_base.eval()
    for cid in clients:
        loader = client_data[cid]["val"]
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                # ResNet expects different input, but 32x32 should still work with timm
                logits = server_forward(x)
                correct += (logits.argmax(1) == y).sum().item()
                total += y.size(0)
        print(f"  [{cid}] server baseline acc: {correct/total:.4f}")

    # ── Phase 3: Per-client rejectors ──
    print("\n=== Phase 3: Building rejectors ===")
    rejectors = {}
    for cid in clients:
        rejectors[cid] = Rejector(input_dim=3*32*32, hidden_dim=128).to(DEVICE)

    # ── Phase 4: Alternating optimization ──
    print("\n=== Phase 4: Alternating L2H training ===")
    
    c_e = 1.0   # misclassification cost
    c_1 = 0.3   # deferral cost (higher = less defer)
    
    for rnd in range(1, ROUNDS + 1):
        print(f"\n--- Round {rnd}/{ROUNDS} ---")
        
        for cid in clients:
            server_base.set_adapter(cid)
            train_loader = client_data[cid]["train"]
            client_model = clients[cid]
            rejector = rejectors[cid]
            
            # Stage 1: Train LoRA adapter
            server_base.train()
            adapter_params = [p for p in server_base.parameters() if p.requires_grad]
            adapter_opt = Adam(adapter_params, lr=5e-5, weight_decay=1e-4)
            
            for ep in range(ADAPTER_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    logits = server_forward(x)
                    loss = F.cross_entropy(logits, y)
                    adapter_opt.zero_grad(); loss.backward(); adapter_opt.step()
            
            # Stage 2: Train rejector
            server_base.eval()
            rejector.train()
            rej_opt = Adam(rejector.parameters(), lr=1e-3)
            
            for ep in range(REJECTOR_EPOCHS):
                for x, y in train_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    with torch.no_grad():
                        server_logits = server_forward(x)
                        client_logits = client_model(x)
                    defer_p = rejector(x)
                    client_correct = (client_logits.argmax(1) == y).float()
                    server_correct = (server_logits.argmax(1) == y).float()
                    cost_local = c_e * (1.0 - client_correct)
                    cost_defer = c_1 + c_e * (1.0 - server_correct)
                    loss = (defer_p * cost_defer + (1 - defer_p) * cost_local).mean()
                    rej_opt.zero_grad(); loss.backward(); rej_opt.step()
            
            rejector.eval()
        
        # Evaluate every 2 rounds
        if rnd % 2 == 0 or rnd == ROUNDS:
            print(f"\n  Evaluation (round {rnd}):")
            for cid in clients:
                server_base.set_adapter(cid)
                server_base.eval()
                client_model = clients[cid]
                rejector = rejectors[cid]
                rejector.eval()
                val_loader = client_data[cid]["val"]
                
                sys_correct, total = 0, 0
                n_defer = 0
                client_only_correct = 0
                server_only_correct = 0
                
                with torch.no_grad():
                    for x, y in val_loader:
                        x, y = x.to(DEVICE), y.to(DEVICE)
                        defer_p = rejector(x)
                        defer_mask = defer_p > 0.5
                        
                        c_preds = client_model(x).argmax(1)
                        s_preds = server_forward(x).argmax(1)
                        
                        # System: use client where not deferred, server where deferred
                        sys_preds = torch.where(defer_mask, s_preds, c_preds)
                        sys_correct += (sys_preds == y).sum().item()
                        n_defer += defer_mask.sum().item()
                        client_only_correct += (c_preds == y).sum().item()
                        server_only_correct += (s_preds == y).sum().item()
                        total += y.size(0)
                
                defer_rate = n_defer / total
                print(f"    [{cid}] system_acc={sys_correct/total:.4f} | "
                      f"client_only={client_only_correct/total:.4f} | "
                      f"server_lora={server_only_correct/total:.4f} | "
                      f"defer_rate={defer_rate:.4f}")

    # ── Phase 5: Cross-corruption evaluation ──
    print("\n=== Phase 5: Cross-corruption test (each client on ALL corruptions) ===")
    all_corruptions = ["gaussian_noise", "shot_noise", "impulse_noise", "defocus_blur", "motion_blur"]
    
    for cid in clients:
        server_base.set_adapter(cid)
        server_base.eval()
        client_model = clients[cid]
        rejector = rejectors[cid]
        rejector.eval()
        own_corr = CORRUPTIONS[int(cid.split("_")[1])]
        
        print(f"\n  [{cid}] (trained on {own_corr}):")
        for corr in all_corruptions:
            imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
            # Use last 2000 as test
            x_test = imgs[-2000:].to(DEVICE)
            y_test = labels[-2000:].to(DEVICE)
            
            with torch.no_grad():
                c_preds = client_model(x_test).argmax(1)
                s_preds = server_forward(x_test).argmax(1)
                defer_p = rejector(x_test)
                defer_mask = defer_p > 0.5
                sys_preds = torch.where(defer_mask, s_preds, c_preds)
            
            c_acc = (c_preds == y_test).float().mean().item()
            s_acc = (s_preds == y_test).float().mean().item()
            sys_acc = (sys_preds == y_test).float().mean().item()
            d_rate = defer_mask.float().mean().item()
            marker = " ← own" if corr == own_corr else ""
            print(f"    {corr:20s}: client={c_acc:.4f} server_lora={s_acc:.4f} system={sys_acc:.4f} defer={d_rate:.4f}{marker}")

    print("\n✅ Sanity check complete!")


if __name__ == "__main__":
    main()
