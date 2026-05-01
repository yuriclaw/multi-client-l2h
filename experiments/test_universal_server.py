"""
Quick experiment: Train a universal server head (pooled data from all 5 clients)
and evaluate on each client's test set.
Compare with per-client server head accuracy.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, ConcatDataset
import timm
from collections import OrderedDict
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

def get_client_datasets(n_train=8000, n_test=2000):
    transform_32 = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_gray = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(3),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    clients = {}
    
    ds = torchvision.datasets.CIFAR10('/tmp/data', train=True, download=True, transform=transform_32)
    ds_test = torchvision.datasets.CIFAR10('/tmp/data', train=False, download=True, transform=transform_32)
    clients[0] = {'name': 'CIFAR-10', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_test, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.SVHN('/tmp/data', split='train', download=True, transform=transform_32)
    ds_test = torchvision.datasets.SVHN('/tmp/data', split='test', download=True, transform=transform_32)
    clients[1] = {'name': 'SVHN', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_test, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.FashionMNIST('/tmp/data', train=True, download=True, transform=transform_gray)
    ds_test = torchvision.datasets.FashionMNIST('/tmp/data', train=False, download=True, transform=transform_gray)
    clients[2] = {'name': 'FashionMNIST', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_test, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.STL10('/tmp/data', split='train', download=True, transform=transform_32)
    ds_test = torchvision.datasets.STL10('/tmp/data', split='test', download=True, transform=transform_32)
    clients[3] = {'name': 'STL-10', 'train': Subset(ds, range(min(n_train, len(ds)))), 'test': Subset(ds_test, range(n_test)), 'n_classes': 10}
    
    ds = torchvision.datasets.GTSRB('/tmp/data', split='train', download=True, transform=transform_32)
    ds_test = torchvision.datasets.GTSRB('/tmp/data', split='test', download=True, transform=transform_32)
    clients[4] = {'name': 'GTSRB', 'train': Subset(ds, range(n_train)), 'test': Subset(ds_test, range(n_test)), 'n_classes': 43}
    
    return clients

def extract_features(backbone, dataset, batch_size=128):
    """Extract frozen ResNet-50 features."""
    backbone.eval()
    transform_up = transforms.Resize((224, 224), antialias=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    all_feats, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = transform_up(x).to(device)
            feat = backbone(x)
            all_feats.append(feat.cpu())
            all_labels.append(y)
    return torch.cat(all_feats), torch.cat(all_labels)

def train_head(feats, labels, n_classes, epochs=200, lr=0.01):
    head = nn.Linear(feats.shape[1], n_classes).to(device)
    dataset = TensorDataset(feats, labels)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)
    optimizer = optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    head.train()
    for epoch in range(epochs):
        for f, y in loader:
            f, y = f.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(head(f), y)
            loss.backward()
            optimizer.step()
        scheduler.step()
    return head

def eval_head(head, feats, labels):
    head.eval()
    with torch.no_grad():
        preds = head(feats.to(device)).argmax(1).cpu()
    return (preds == labels).float().mean().item()

def main():
    clients = get_client_datasets()
    
    # Load frozen backbone
    print("Loading frozen ResNet-50...")
    backbone_full = timm.create_model('resnet50', pretrained=True).to(device)
    backbone = nn.Sequential(OrderedDict([
        (name, module) for name, module in backbone_full.named_children() if name != 'fc'
    ]))
    backbone.add_module('flatten', nn.Flatten())
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    # Extract features for all clients
    print("Extracting features...")
    train_feats = {}
    train_labels = {}
    test_feats = {}
    test_labels = {}
    for i, info in clients.items():
        print(f"  Client {i} ({info['name']})...")
        train_feats[i], train_labels[i] = extract_features(backbone, info['train'])
        test_feats[i], test_labels[i] = extract_features(backbone, info['test'])
    
    # --- Per-client server heads (200 epochs) ---
    print("\nTraining per-client server heads (200 epochs)...")
    per_client_acc = {}
    for i, info in clients.items():
        head = train_head(train_feats[i], train_labels[i], info['n_classes'], epochs=200)
        per_client_acc[i] = eval_head(head, test_feats[i], test_labels[i])
        print(f"  {info['name']}: {per_client_acc[i]:.4f}")
    
    # --- Universal server head ---
    # Build unified label space: offset labels for each client
    print("\nTraining universal server head (200 epochs, pooled data)...")
    label_offset = {}
    offset = 0
    for i, info in clients.items():
        label_offset[i] = offset
        offset += info['n_classes']
    total_classes = offset
    print(f"  Total unified classes: {total_classes}")
    
    all_feats_list = []
    all_labels_list = []
    for i in clients:
        all_feats_list.append(train_feats[i])
        all_labels_list.append(train_labels[i] + label_offset[i])
    
    uni_feats = torch.cat(all_feats_list)
    uni_labels = torch.cat(all_labels_list)
    
    uni_head = train_head(uni_feats, uni_labels, total_classes, epochs=200)
    
    universal_acc = {}
    for i, info in clients.items():
        # Offset test labels too
        offset_test_labels = test_labels[i] + label_offset[i]
        universal_acc[i] = eval_head(uni_head, test_feats[i], offset_test_labels)
        print(f"  {info['name']}: {universal_acc[i]:.4f}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"{'Client':<15} {'Per-Client':>10} {'Universal':>10} {'Delta':>10}")
    print("-" * 60)
    avg_per, avg_uni = 0, 0
    for i, info in clients.items():
        d = per_client_acc[i] - universal_acc[i]
        print(f"{info['name']:<15} {per_client_acc[i]:>10.4f} {universal_acc[i]:>10.4f} {d:>+10.4f}")
        avg_per += per_client_acc[i]
        avg_uni += universal_acc[i]
    n = len(clients)
    print("-" * 60)
    print(f"{'Average':<15} {avg_per/n:>10.4f} {avg_uni/n:>10.4f} {(avg_per-avg_uni)/n:>+10.4f}")

if __name__ == '__main__':
    main()
