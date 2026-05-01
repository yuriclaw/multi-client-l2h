# Multi-Client L2H 探索全过程报告

**项目：** Multi-Client One-Server Collaborative Inference  
**时间跨度：** 2026年3月22日 – 4月2日  
**研究人员：** Che.（NYUzi，研究方向决策）+ Muse（实验执行）  
**服务器：** sarwate GPU (`yw828@100.109.236.101`)，conda env `l2h-b`  
**代码仓库：** `~/multi-client-l2h/` (sarwate)，本地 `~/.openclaw/workspace-muse/experiments/`

---

## 1. 研究目标

构建一个 **Two-Stage Learning-to-Help (L2H)** 系统用于多客户端协作推理：
- **Client（边缘端）：** 轻量模型（AlexNet/ResNet-18/MobileNetV2/ShuffleNetV2/SqueezeNet），处理本地分类任务
- **Server（云端）：** 强力模型（DINOv2-B/14），作为 "helper"
- **Rejector（边缘端）：** 决定每个样本是 client 自己处理，还是 defer 给 server
- **核心问题：** L2H 训练的 rejector 能否比简单的 softmax confidence 阈值（ConfTh）做出更好的 defer 决策？

---

## 2. 系统架构

```
Input Image
    │
    ├──→ Client Model (edge) ──→ Client Prediction + Softmax Confidence
    │         │
    │         └──→ Hidden Features ──┐
    │                                 │
    ├──→ Rejector (edge) ────────────┤──→ Defer Decision
    │    (32×32 CNN + hidden concat)  │     (keep / defer)
    │                                 │
    └──→ Server (cloud) ─────────────┘──→ Server Prediction
         DINOv2 + MLP Head                 (if deferred)
```

**Rejector 架构（~200K 参数）：**
- CNN branch: Conv2d(3→32→64→128) on 32×32 image → 128-dim
- Concat with client hidden features (256-1280 dim)
- MLP: Linear(128+hidden → 128) → Linear(128 → 2)
- 输出: P(keep) vs P(defer)

**Deferral 方法对比：**
| 方法 | 决策依据 | 训练成本 |
|------|---------|---------|
| **L2H** | Rejector P(defer) 分数 | 需要训练 rejector（~30ep） |
| **ConfTh** | Client softmax max probability | **零训练成本** |
| **Gatekeeper** | Client confidence after finetune | 需要 finetune client |
| **Random** | 随机 defer | 无 |

---

## 3. 实验历程

### Phase 1: 基础架构搭建（v1-v6，3月22-24日）

**目标：** 确定 client/server/rejector 的最佳组合

| 版本 | Client | Server | 关键变化 | 文件 |
|-----|--------|--------|---------|------|
| v1 | AlexNet-style CNN | ResNet-50 | 初始系统 | `test_l2h_system.py` |
| v2 | AlexNet | ResNet-50 | Gatekeeper 对比 | `test_l2h_v2.py` |
| v3 | AlexNet | ResNet-50 | 5 种 loss 函数对比 | `test_l2h_v3_losses.py` |
| v4 | AlexNet | ResNet-50 | GK alpha 网格搜索 | `test_l2h_v4.py` |
| v5 | AlexNet | ResNet-50 | 修正 GK/Moz/OvA loss | `test_l2h_v5.py` |
| v6 | AlexNet | **ViT-B/16** | 换 server backbone | `test_l2h_v6_vit.py` |

**结论：** ConfTh 和 Gatekeeper 始终优于 L2H rejector。

---

### Phase 2: 扩展 Client 和 Server（v7 系列，3月25-26日）

**目标：** 换更强的 client/server 组合，测试多数据集

| 版本 | Client | Server | 数据集 | 文件 |
|-----|--------|--------|-------|------|
| v7.0 | AlexNet | ResNet-50/DINOv2/CLIP | 5 datasets | `test_l2h_v7_backbones.py` |
| v7.1 | **ResNet-18** | 3 backbones | 5 datasets | `test_l2h_v7.1_resnet18client.py` |
| v7.2 | **MobileNetV2** (frozen) | 3 backbones | 5 datasets | `test_l2h_v7.2_mobilenet_client.py` |
| v7.3 | **MobileNetV2** (finetune) | 3 backbones | 5 datasets | `test_l2h_v7.3_mobilenet_finetune.py` |
| v7.4 | MobileNetV2 FT | **DINOv2 + MLP** | 5 datasets | `test_l2h_v7.4_dinov2_mlphead.py` |
| v7.4b | MobileNetV2 FT | DINOv2 + MLP | 对比 Per/Uni head | `test_l2h_v7.4b_universal.py` |
| v7.5 | MobileNetV2 FT | DINOv2 + MLP | **Val→Test 阈值** | `test_l2h_v7.5_threshold.py` |
| v7.6 | MobileNetV2 FT | DINOv2 + MLP | AlexNet CNN rejector | `test_l2h_v7.6_alexnet_rejector.py` |

**v7.5 结果（5 数据集平均，Val→Test threshold）：**

| Deferral Rate | L2H | GK(0.8) | ConfTh | Random |
|--------------|-----|---------|--------|--------|
| 10% | 76.4% | 78.3% | **78.4%** | 75.5% |
| 30% | 81.7% | 87.5% | **87.6%** | 82.2% |
| 50% | 84.8% | **89.7%** | **89.7%** | 87.0% |
| 70% | 87.8% | 89.2% | **89.3%** | 89.7% |
| 90% | **89.8%** | 89.7% | 89.7% | 89.0% |

**结论：** ConfTh ≈ Gatekeeper >> L2H >> Random。只有 90% deferral 时 L2H 才勉强持平。

---

### Phase 3: 多 Client 异构场景（v8 系列，3月26日-4月1日）

**目标：** 5 个不同架构的 client 在同一 CIFAR-100 上训练，测试 per-client vs universal 策略

#### v8.1: Privacy 场景（3月27日）
- 测试 server 能获取多少 client 数据
- **结论：** Universal head 需要 ≥50% 数据才接近 Per-Client；DINOv2 对加噪极其鲁棒

#### v8.2b: Cost-sensitive Joint Training（3月29日）
- 5 个 client：AlexNet(256), ResNet-18(512), MobileNetV2(1280), ShuffleNetV2(1024), SqueezeNet(512)
- **结论：** ConfTh 全面赢 L2H

#### v8.2c: 改善 Calibration（3月30日）
- 加入 Label Smoothing 0.1, Mixup α=0.2, RandAugment, Dropout, 更长训练
- Client accuracy 变化：AlexNet 18.3%(-5.7), ResNet-18 **50.8%(+17.4)**, MobileNetV2 **67.5%(+6.5)**
- **结论：** Calibration 改善后 ConfTh 优势从 3.7pp 扩大到 4.6pp @50%

**v8.2c 结果（CIFAR-100，5 clients 平均）：**

| Deferral | L2H | ConfTh | Δ(L2H-CT) |
|----------|-----|--------|-----------|
| 10% | 54.8% | **56.2%** | -1.5 |
| 50% | 69.7% | **74.3%** | **-4.6** |
| 70% | 76.3% | **81.2%** | -4.9 |
| 90% | 82.5% | **84.8%** | -2.3 |

#### v8.2d: Per-Client vs Universal Head（3月30日）
- Per-Client(5K) ≈ Universal(5K)；Universal(25K) >> Per-Client(5K)
- **唯一 L2H 赢的场景：** 90% deferral + Universal(25K) → L2H 85.9% > ConfTh 84.8%（+1.0pp）

#### v8.2e: Equal Data Budget（3月30日）
- Per(5K) ≈ Uni(5K)（差 <1pp）
- 只有 Uni(25K) 在 90% 胜出 → 证明优势来自数据量非 rejector 质量

#### v8.2f: Error-only Server Training（3月31日）
- Server head 只用 client 预测错误的样本训练
- **大失败：** ResNet-18 error-only head 只有 1.7% accuracy（类别分布崩溃）

#### v8.2g: Weighted Server Training（4月1日）
- Error 样本 loss 权重 3x/5x/10x
- **无效果：** DINOv2 features 30ep 已经 100% train acc，加权无意义

**Phase 3 总结：** 在同分布（client 和 server 处理相同类别）场景下，**ConfTh 始终优于 L2H**。

---

### Phase 4: 辅助分析 — Calibration 研究（3月29日）

`confidence_calibration.py` — 分析 5 个 client 模型的 confidence 分布

| Client | 整体 Acc | 最佳校准 | 关键发现 |
|--------|---------|---------|---------|
| ShuffleNetV2 | 61% | ✅ 最佳 | (0.9,1.0] bin: 3.3% data, 99% acc |
| SqueezeNet | 58% | ❌ 最差 | (0.9,1.0] bin: 59% data, 仅 70% acc |
| AlexNet | 18% | 特殊 | 53% data conf<0.2（underconfident） |

**关键发现：** Calibration 越好 → ConfTh 越强。

---

### Phase 5: OOD 场景突破（4月1-2日）

**转折点：** 董事长提出新方向 — client 只见过部分类别，遇到新类时需要 defer 到 server

#### Step 1: OOD Confidence 分析（4月1日）

`ood_confidence_analysis.py`
- MobileNetV2 在 CIFAR-100 前 10 类训练，测试时面对全部 100 类
- **发现：** OOD 样本 mean conf=0.54 vs Known mean conf=0.83
- **但 25.6% 的 OOD 样本 conf > 0.7** — 这些高置信 OOD 是 ConfTh 抓不住的

#### Step 2: 单 Client OOD Deferral（4月1日）

`ood_rejector_vs_confth.py`
- **第一次 L2H 全面赢 ConfTh！**

| Rate | L2H(+OOD) | ConfTh | Δ |
|------|-----------|--------|---|
| 10% | **57.4%** | 56.2% | +1.2 |
| 30% | **73.9%** | 72.7% | +1.2 |
| 50% | **85.5%** | 84.6% | +0.9 |
| 70% | **89.8%** | 88.5% | +1.3 |
| 90% | **90.4%** | 90.0% | +0.4 |

AUROC: L2H **0.929** vs ConfTh 0.882（+4.7pp）

#### Step 3: Multi-Client OOD 实验（4月1-2日）

四组实验，统一结论 — **L2H 在 OOD 场景全面优于 ConfTh：**

| 实验 | 文件 | 每 client 类数 | 正则化 | L2H-CT @50% | AUROC gap |
|-----|------|--------------|--------|-------------|-----------|
| 10cls+LS | `ood_multi_client_diff_data.py` | 10 | ✅ LS+Mixup | **+2.3pp** | +6.5pp |
| 20cls noLS | `ood_multi_client_20class_nols.py` | 20 | ❌ 无 | **+3.5pp** | +8.5pp |
| 30cls random | `ood_multi_client_30class_random.py` | 30 | ❌ 无 | **+2.7pp** | +8.2pp |
| diff arch | `ood_multi_client_diff_arch.py` | 10 | ✅ LS+Mixup | **+2.0pp** | +5.7pp |

**20-class 无正则化实验详细结果（最佳条件）：**

| Deferral | L2H | ConfTh | Δ |
|----------|-----|--------|---|
| 10% | 52.4% | 51.8% | +0.6 |
| 30% | **68.3%** | 66.9% | +1.4 |
| 50% | **81.3%** | 77.8% | **+3.5** |
| 70% | **87.4%** | 86.2% | +1.2 |
| 90% | 88.6% | 88.4% | +0.2 |

AUROC 平均: L2H **0.893** vs ConfTh **0.808** (+8.5pp)

---

## 4. 技术细节

### 4.1 共同实验设置
| 参数 | 值 |
|------|-----|
| Dataset | CIFAR-100 (100 classes, 32×32 images) |
| SEED | 42 |
| Client 训练数据 | 5000 samples per client |
| Server 训练数据 | 25000 samples (all 100 classes) |
| Threshold 设定 | Val set 确定 → Test set 应用 |
| Deferral rates | 10%, 30%, 50%, 70%, 90% |

### 4.2 Client 训练配置
| 参数 | 同分布 (v8.x) | OOD (有正则) | OOD (无正则) |
|------|-------------|-------------|-------------|
| Backbone LR | 1e-4 | 1e-4 | 1e-4 |
| Head LR | 1e-3 | 1e-3 | 1e-3 |
| Epochs | 150 | 60 | 60 |
| Label Smoothing | 0.1 | 0.1 | ❌ |
| Mixup α | 0.2 | 0.2 | ❌ |
| Weight Decay | 5e-4 | 5e-4 | 5e-4 |

### 4.3 Server 配置
- **Backbone:** DINOv2-B/14 (ViT-Base, patch 14, frozen, 86M 参数)
- **Head:** MLP (768→512→ReLU→Dropout(0.3)→N_classes)
- **训练:** 30 epochs, Adam lr=1e-3, CosineAnnealing
- **同分布:** Head 输出维度 = client 的类别数（如 100）
- **OOD:** Head 输出维度 = 全部 100 类（universal）

### 4.4 Rejector 配置
- **参数量:** 177K-308K（取决于 client hidden dim）
- **训练:** 30 epochs, cost-sensitive loss, frozen server head
- **OOD 训练数据:** 5K known + 5K OOD 样本
- **Loss:** `L = -[e_server · log P(defer) + e_client · log P(keep)]`

---

## 5. 核心发现

### 5.1 同分布场景：ConfTh 是最优的

**为什么 Rejector 打不过 ConfTh：**
1. **信息瓶颈：** Rejector 输入 = 32×32 图片 + client hidden。ConfTh 用的 softmax confidence 直接来自 client 的 224×224 full representation，信息量更大。
2. **DINOv2 太强：** Server head 30ep 就 100% train acc，无论怎么调整训练策略都收敛到相同结果。
3. **Calibration 越好 → ConfTh 越强：** Label Smoothing 改善 calibration 后，softmax confidence 更可靠。

### 5.2 OOD 场景：L2H 胜出

**为什么 Rejector 能赢：**
1. Client 面对**从未见过的类**时，softmax 仍会给出 overconfident 的预测（归到某个已知类）
2. Rejector 能从 hidden features 中学到 "这个 pattern 不像训练分布" 的信号
3. 没有正则化（no Label Smoothing/Mixup）时，client 更 overconfident，ConfTh 更容易被骗

**但优势有限（+2-3.5pp）的原因：**
1. ImageNet pretrained backbone 泛化能力强，对 OOD 仍有一定区分力
2. 类别数少（10-30 类）时 softmax 分辨率高
3. ConfTh AUROC 仍有 0.80-0.87，并没有崩溃

### 5.3 Per-Client vs Universal Server Head

| 条件 | Per-Client | Universal |
|------|-----------|-----------|
| 等量数据 (5K) | ≈ 持平 | ≈ 持平 |
| Universal 数据更多 (25K vs 5K) | 落后 ~3pp | **更优** |
| 隐私受限（少量数据） | **更优** | 需要 ≥50% 数据 |

---

## 6. 失败实验记录

| 尝试 | 思路 | 结果 | 原因 |
|------|------|------|------|
| Error-only server (v8.2f) | 只用 client 错误样本训练 server | 严重失败 (1.7% acc) | 类别分布崩溃 |
| Weighted server (v8.2g) | Error 样本权重 3x/5x/10x | 无效果 | DINOv2 已收敛 |
| L2H(known only) | Rejector 训练不含 OOD 数据 | AUROC=0.25 | 学不到 OOD pattern |

---

## 7. 代码索引

### 服务器路径
```
~/multi-client-l2h/experiments/
├── test_l2h_system.py          # v1: 初始系统
├── test_l2h_v2.py              # v2: Gatekeeper 对比
├── test_l2h_v3_losses.py       # v3: 5 种 loss
├── test_l2h_v4.py              # v4: GK alpha grid
├── test_l2h_v5.py              # v5: 修正 loss
├── test_l2h_v6_vit.py          # v6: ViT server
├── test_l2h_v7_backbones.py    # v7: 多 backbone
├── test_l2h_v7.1~v7.6          # v7.x: Client/Server 变体
├── test_l2h_v8.1_privacy.py    # v8.1: Privacy scenarios
├── test_l2h_v8.2c~v8.2g        # v8.x: Multi-client 同分布
├── confidence_calibration.py   # Calibration 分析
├── ood_confidence_analysis.py  # OOD confidence 分布
├── ood_rejector_vs_confth.py   # 单 client OOD ⭐
├── ood_multi_client_diff_data.py      # 5 clients 不同数据 ⭐
├── ood_multi_client_diff_arch.py      # 5 clients 不同架构 ⭐
├── ood_multi_client_20class_nols.py   # 20 类无正则化 ⭐
└── ood_multi_client_30class_random.py # 30 类随机 ⭐
```

### Log 文件（服务器 `~/`）
每个实验对应 `l2h_v*.txt` 或 `ood_*.txt` 日志。

---

## 8. 结论与展望

### 确定结论
1. **同分布场景：** ConfTh（零成本 baseline）是最优的 deferral 方法
2. **OOD 场景：** L2H rejector 优于 ConfTh（+2-3.5pp accuracy, +5.7-8.5pp AUROC）
3. **Per-Client vs Universal head：** 等量数据下持平，Universal 有数据量优势
4. **DINOv2 server：** Features 极强，linear head 训练几乎不可能失败

### Paper 方向建议
- **聚焦 OOD 场景** — 这是 L2H 有明确优势的领域
- 可探索让 ConfTh 更容易失效的场景（near-OOD、更多类别、更强 finetune）
- 考虑 communication cost angle — L2H 可在 client 端做决策，不需要发送数据到 server 来判断是否 defer

---

*Report generated on 2026-04-18 by Muse 🎭*
