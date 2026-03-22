"""Direction B: Prove LoRA adapter value independently of L2H.

For each corruption type:
1. Server baseline (ResNet-50 pretrained, clean-only probe) → acc on corruption
2. Server + LoRA trained on ALL client corruption data → acc on corruption
3. Compare: does LoRA improve server accuracy on unseen corruption?

No rejector, no L2H — pure LoRA evaluation.
"""
from __future__ import annotations
import os, sys, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import timm
from peft import LoraConfig, get_peft_model

# ── Config ──
NUM_CLASSES = 10
BATCH_SIZE = 128
LORA_RANK = 16
LORA_LR = 1e-4
LORA_EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = os.environ.get("DATA_DIR", os.path.expanduser("~/data"))
SEVERITY = 3

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "fog", "motion_blur",
    "brightness", "contrast", "frost", "snow",
]

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_cifar10c(data_dir, corruption, severity):
    base = os.path.join(data_dir, "CIFAR-10-C")
    imgs_all = np.load(os.path.join(base, f"{corruption}.npy"))
    labels_all = np.load(os.path.join(base, "labels.npy"))
    start = (severity - 1) * 10000
    imgs = imgs_all[start:start+10000]
    labels = labels_all[start:start+10000]
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2470, 0.2435, 0.2616])
    imgs = (imgs / 255.0 - mean) / std
    imgs = torch.tensor(imgs, dtype=torch.float32).permute(0, 3, 1, 2)
    labels = torch.tensor(labels, dtype=torch.long)
    return imgs, labels

def build_server(num_classes):
    base = timm.create_model("resnet50", pretrained=True, num_classes=num_classes)
    # Freeze everything
    for p in base.parameters():
        p.requires_grad = False
    return base

def main():
    set_seed(42)
    print(f"Device: {DEVICE}")
    
    # ── Build server ──
    server = build_server(NUM_CLASSES).to(DEVICE)
    upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False).to(DEVICE)
    
    def server_forward(model, x, bs=128):
        """Batched forward to avoid OOM."""
        preds = []
        for i in range(0, len(x), bs):
            xb = x[i:i+bs].to(DEVICE)
            with torch.no_grad():
                preds.append(model(upsample(xb)))
        return torch.cat(preds)
    
    # ── Phase 1: Linear probe on clean CIFAR-10 ──
    print("\n=== Phase 1: Linear probe on clean CIFAR-10 ===")
    import torchvision, torchvision.transforms as T
    probe_transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
    probe_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=probe_transform)
    probe_loader = DataLoader(probe_ds, batch_size=256, shuffle=True, num_workers=2)
    
    # Unfreeze only fc
    for name, p in server.named_parameters():
        if 'fc' in name:
            p.requires_grad = True
    
    probe_opt = Adam([p for p in server.parameters() if p.requires_grad], lr=1e-3)
    server.train()
    for ep in range(8):
        correct = total = 0
        for x_b, y_b in probe_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            logits = server(upsample(x_b))
            loss = F.cross_entropy(logits, y_b)
            probe_opt.zero_grad(); loss.backward(); probe_opt.step()
            correct += (logits.argmax(1) == y_b).sum().item()
            total += y_b.size(0)
        if (ep+1) % 2 == 0:
            print(f"  Probe epoch {ep+1}/8: acc={correct/total:.4f}")
    
    # Re-freeze everything
    for p in server.parameters():
        p.requires_grad = False
    server.eval()
    
    # Save probed weights
    probed_state = {k: v.clone() for k, v in server.state_dict().items()}
    
    # ── Phase 2: Baseline server accuracy on each corruption ──
    print("\n=== Phase 2: Server baseline (no LoRA) on each corruption ===")
    baselines = {}
    for corr in CORRUPTIONS:
        imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
        # Use last 2000 as test
        x_test, y_test = imgs[-2000:], labels[-2000:]
        logits = server_forward(server, x_test)
        acc = (logits.argmax(1).cpu() == y_test).float().mean().item()
        baselines[corr] = acc
        print(f"  {corr:20s}: {acc:.4f}")
    
    # ── Phase 3: For each corruption, train LoRA on that corruption's data, then eval ──
    print("\n=== Phase 3: LoRA trained on each corruption's data ===")
    
    results = {}
    for corr in CORRUPTIONS:
        print(f"\n  --- {corr} ---")
        
        # Reload probed weights (fresh start for each corruption)
        server.load_state_dict(probed_state)
        
        # Wrap with LoRA
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_RANK * 2,
            target_modules=["layer3.0.conv2", "layer3.1.conv2", "layer3.2.conv2",
                            "layer3.3.conv2", "layer3.4.conv2", "layer3.5.conv2",
                            "layer4.0.conv2", "layer4.1.conv2", "layer4.2.conv2"],
            lora_dropout=0.05,
            bias="none",
        )
        server_lora = get_peft_model(server, lora_config).to(DEVICE)
        
        # Load corruption data: first 8000 train, last 2000 test
        imgs, labels = load_cifar10c(DATA_DIR, corr, SEVERITY)
        x_train, y_train = imgs[:8000], labels[:8000]
        x_test, y_test = imgs[-2000:], labels[-2000:]
        
        train_ds = TensorDataset(x_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # Train LoRA on ALL corruption data
        optimizer = Adam(server_lora.parameters(), lr=LORA_LR, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=LORA_EPOCHS)
        
        server_lora.train()
        for ep in range(LORA_EPOCHS):
            correct = total = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = server_lora(upsample(xb))
                loss = F.cross_entropy(logits, yb)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                correct += (logits.argmax(1) == yb).sum().item()
                total += yb.size(0)
            scheduler.step()
            if (ep+1) % 5 == 0:
                print(f"    Epoch {ep+1}/{LORA_EPOCHS}: train_acc={correct/total:.4f}")
        
        # Eval on test set (same corruption)
        server_lora.eval()
        logits = server_forward(server_lora, x_test)
        lora_acc = (logits.argmax(1).cpu() == y_test).float().mean().item()
        
        # Also eval on clean CIFAR-10 test (check if LoRA hurts clean performance)
        clean_test_ds = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=probe_transform)
        clean_loader = DataLoader(clean_test_ds, batch_size=256, num_workers=2)
        clean_correct = clean_total = 0
        for xb, yb in clean_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with torch.no_grad():
                p = server_lora(upsample(xb)).argmax(1)
            clean_correct += (p == yb).sum().item()
            clean_total += yb.size(0)
        clean_acc = clean_correct / clean_total
        
        # Cross-corruption eval
        cross_results = {}
        for other_corr in CORRUPTIONS:
            if other_corr == corr:
                continue
            o_imgs, o_labels = load_cifar10c(DATA_DIR, other_corr, SEVERITY)
            o_test = o_imgs[-2000:]
            o_ytest = o_labels[-2000:]
            o_logits = server_forward(server_lora, o_test)
            o_acc = (o_logits.argmax(1).cpu() == o_ytest).float().mean().item()
            cross_results[other_corr] = o_acc
        
        results[corr] = {
            "baseline": baselines[corr],
            "lora": lora_acc,
            "clean_after_lora": clean_acc,
            "cross": cross_results,
        }
        
        delta = lora_acc - baselines[corr]
        marker = "📈" if delta > 0.01 else ("📉" if delta < -0.01 else "→")
        print(f"    Result: baseline={baselines[corr]:.4f} → LoRA={lora_acc:.4f} ({delta:+.4f}) {marker}")
        print(f"    Clean acc after LoRA: {clean_acc:.4f}")
        
        # Cleanup LoRA
        server = server_lora.merge_and_unload()
        server.load_state_dict(probed_state)  # reset
    
    # ── Summary ──
    print("\n" + "="*80)
    print("SUMMARY: LoRA value per corruption")
    print("="*80)
    print(f"{'Corruption':20s} {'Baseline':>10s} {'LoRA':>10s} {'Delta':>10s} {'Clean':>10s}")
    print("-"*60)
    for corr in CORRUPTIONS:
        r = results[corr]
        delta = r["lora"] - r["baseline"]
        print(f"{corr:20s} {r['baseline']:10.4f} {r['lora']:10.4f} {delta:+10.4f} {r['clean_after_lora']:10.4f}")
    
    # Cross-corruption transfer matrix
    print("\n" + "="*80)
    print("CROSS-CORRUPTION TRANSFER (LoRA trained on row, tested on col)")
    print("Showing DELTA vs baseline (positive = LoRA helps)")
    print("="*80)
    header = f"{'Trained on':20s}" + "".join(f"{c[:8]:>10s}" for c in CORRUPTIONS)
    print(header)
    print("-"*len(header))
    for corr in CORRUPTIONS:
        row = f"{corr:20s}"
        for other in CORRUPTIONS:
            if other == corr:
                delta = results[corr]["lora"] - results[corr]["baseline"]
                row += f"{delta:+10.4f}*"
            else:
                cross_acc = results[corr]["cross"][other]
                delta = cross_acc - baselines[other]
                row += f"{delta:+10.4f} "
        print(row)
    
    print("\n✅ LoRA value test complete!")

if __name__ == "__main__":
    main()
