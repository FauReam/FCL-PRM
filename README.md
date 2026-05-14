# FCL-PRM

**Federated Continual Process Reward Model**
**联邦持续过程奖励模型 — for LLM step-level reasoning verification**

![Status](https://img.shields.io/badge/status-active-green)
![Stage](https://img.shields.io/badge/stage-M2_baseline-blue)
![License](https://img.shields.io/badge/license-MIT_pending-lightgrey)

---

## 一句话标语

> **在不共享 step-level 推理轨迹的前提下，跨机构协同训练并持续更新 step-level 过程奖励模型 (PRM)，并由此回答中心化研究无法触及的科学问题：同一 LLM 在不同推理域产生的 step 嵌入空间是否共享通用基底？**

---

## 1. 项目简介

### 1.1 PRM 是什么
Process Reward Model (PRM) 在 LLM 推理（CoT）的**每一步**给出标量奖励，相比仅评估最终答案的 Outcome Reward Model (ORM)：
- 信号密度高 10x-100x -> RLHF 数据效率显著提升
- 在 o1 / DeepSeek-R1 / Qwen-QwQ / Claude-Thinking / Gemini-2.5-Thinking 等推理模型的 RLVR / GRPO 训练中已成为关键组件
- 2025 年起爆发：PRM800K (OpenAI) / ProcessBench / VersaPRM / MedPRMBench / ThinkPRM / DreamPRM 等

### 1.2 为什么必须联邦化
高质量 step-level 标注高度敏感且分散：
- **医学/临床推理**：医院持有患者诊断 CoT，HIPAA / GDPR 禁止外传
- **代码/审计推理**：金融机构持有合规审查 CoT，含商业秘密
- **数学竞赛/教育**：教培机构持有学生解题轨迹，受未成年人数据保护
- **企业内部**：大厂偏好建设私有 step verifier，不愿混入公共池

中心化 PRM 必然遗漏这些数据 -> **联邦 PRM 是从 outcome-level 联邦 RLHF 自然推进的下一步**。

### 1.3 与 Federated RLHF 的根本差异（**必读**）

| 项 | Federated RLHF (FedBiscuit / APPA / FedPDPO) | **FCL-PRM (本项目)** |
|---|---|---|
| 监督粒度 | outcome-level（整段答案 1 个标量） | **step-level（每步 1 个标量）** |
| 单样本泄漏比特 | <= log2(K) = 1-3 bit | **O(T*log2(K))，T 为步数 -> 数量级差异** |
| 隐私威胁 | preference label leakage | **reasoning trace reconstruction（首次提出）** |
| 域异质性表现 | 偏好不一致（可平均） | **step 语义 polysemy（不可平均）** |
| 联邦聚合难点 | preference 加权 | **step embedding 对齐（创新点 A）** |
| DP 标定 | 已有标准方案 | **step-level 紧界尚无（创新点 B）** |

---

## 2. 核心思想（架构图）

```
+------------------------------------------------------------------+
|                        FCL-PRM 联邦架构                          |
+------------------------------------------------------------------+
|                                                                  |
|   +------------+  +------------+  +------------+  +------------+ |
|   | Client A   |  | Client B   |  | Client C   |  | Client D   | |
|   | (Math)     |  | (Code)     |  | (Medical)  |  | (General)  | |
|   |            |  |            |  |            |  |            | |
|   | CoT+step   |  | CoT+step   |  | CoT+step   |  | CoT+step   | |
|   | 标签       |  | 标签       |  | 标签       |  | 标签       | |
|   |     |      |  |     |      |  |     |      |  |     |      | |
|   |  Local PRM |  |  Local PRM |  |  Local PRM |  |  Local PRM | |
|   |  + DP-SGD  |  |  + DP-SGD  |  |  + DP-SGD  |  |  + DP-SGD  | |
|   +-----+------+  +-----+------+  +-----+------+  +-----+------+ |
|         | DeltaPRM_A        | DeltaPRM_B        | DeltaPRM_C        | DeltaPRM_D |
|         | + step embed  | + step embed  | + step embed  | + ...  |
|         +---------------+--------+--------------+        |
|                                  v                               |
|              +----------------------------------+                |
|              |   Server: Anchor-PRM Aggregator  |                |
|              |   1 公共 anchor step set 对齐    |                |
|              |   2 鲁棒聚合（创新点 P3）        |                |
|              |   3 持续更新机制（基座漂移）     |                |
|              +------------+---------------------+                |
|                           | Global PRM_t+1                       |
|                           v                                      |
|              +----------------------------------+                |
|              |   科学度量层（创新点 A）         |                |
|              |   CD-SPI: Cross-Domain Step      |                |
|              |   Polysemy Index                 |                |
|              +----------------------------------+                |
+------------------------------------------------------------------+
```

---

## 3. 与最近邻工作的边界

| 相邻工作 | 它做了什么 | FCL-PRM 的差异点 |
|---|---|---|
| **FedBiscuit** (ICLR 2025) | 联邦 binary preference selector | outcome-level -> FCL-PRM 是 step-level，每样本泄漏比特数差一个数量级 |
| **APPA** (arXiv:2604.04261, 2026/04) | 多群组 RLHF 公平聚合 | 仍是 outcome-level；FCL-PRM 解决 step 语义对齐问题 |
| **FedPDPO** (arXiv:2603.19741, 2026/03) | 联邦 DPO（无 reward model）| FCL-PRM 显式聚合 step-level 奖励 |
| **VersaPRM** (arXiv:2502.06737) | 14 域多域 PRM，**集中式** | 中心化混合数据；FCL-PRM 保留分布隔离揭示 step polysemy |
| **MedPRMBench** (arXiv:2604.17282) | 医疗 PRM 基准，**集中式** | 单域；FCL-PRM 是跨域联邦 |
| **DreamPRM** (arXiv:2505.20241) | 多模态 PRM + 域重加权，**集中式** | 中心化；FCL-PRM 解决联邦异质性 |
| **APRM** (arXiv:2511.22888) | 集中式对抗训练 PRM | FCL-PRM 解决联邦 + step-level 中毒攻防 |
| **FedGMKD** (Zhang et al., NeurIPS 2024) | 联邦图像分类原型蒸馏 + Discrepancy-Aware Aggregation (CKF+DAT) | 解决 *class-level* Non-IID；FCL-PRM 解决 *step-level* 语义 polysemy，二者异质性本质不同（见下） |
| **PFedDL** (arXiv:2509.20627) | 联邦字典学习 / fMRI | 非 LLM；FCL-PRM 解决 LLM step-level reward 特有问题 |
| **SALT** (arXiv:2511.07772) | 推理时 CoT 激活级隐私干预 | inference-time；FCL-PRM 解决 training-time step 标签隐私 |
| **Safer Reasoning Traces** (arXiv:2603.05618) | PII 在 CoT 步骤的泄漏 | 描述性；FCL-PRM 给出 step-level DP 紧界（创新点 B） |
| **When Reasoning Leaks Membership** (arXiv:2601.13607) | 首个 reasoning model MIA | 攻击端；FCL-PRM 同时给出防御端（联邦 + DP） |
| **Google Federated RLHF Patent** | 工业信号 | outcome-level 专利不覆盖 step-level |

### 与 FedGMKD（NeurIPS 2024）的精确边界

FedGMKD（Zhang et al., NeurIPS 2024）是联邦学习领域处理 Non-IID 异质性的代表作，但其问题设定、技术路线与 FCL-PRM 存在本质差异：

| 维度 | FedGMKD | FCL-PRM |
|---|---|---|
| **任务层级** | Image classification（*class-level*） | Step-level reasoning verification（*token-sequence-level*） |
| **异质性来源** | 客户端间类别分布偏移（covariate shift / label skew） | 同一文本 step 在不同推理域的 *语义 polysemy*（同形不同义） |
| **知识单元** | Class prototype（GMM 聚类特征 + soft prediction） | Step embedding（LLM hidden state 的隐空间方向） |
| **聚合对象** | Prototype features & soft predictions（公式 8） | PRM head 参数 + step embedding 对齐 |
| **蒸馏目标** | 全局原型 <- 本地原型（CKF，公式 5-7） | 无公共数据集蒸馏；锚点 step 对齐（Anchor-PRM） |
| **discrepancy 度量** | KL(prototype_local || prototype_global)（公式 10-11） | Cosine distance of step embeddings across domains（CD-SPI） |

**一句话边界**：FedGMKD 的 *class prototype* 无法直接迁移到 step-level PRM，因为 reasoning step 没有离散的 "class" 概念，其异质性表现为连续语义空间中的方向偏移，而非类别分布偏移。FCL-PRM 的 Anchor-PRM 聚合与 CD-SPI 度量是针对 step embedding 空间重新设计的。

---

## 4. 创新点

### 4.1 CD-SPI（Cross-Domain Step Polysemy Index）

**问题**：同一推理"步"（如「设变量 x 为...」）在 math / code / medical 三域 PRM 中是否共享同一隐空间方向？
**为什么必须联邦才能问**：中心化训练会强行把它们映射到同一空间（VersaPRM 做法），无法测得"原生差异"。
**度量定义**：

```
CD-SPI(s) = 1 - mean_{i,j} cos(h_i(s), h_j(s))
```

其中 h_i(s) 为客户端 i 上 PRM 对步 s 的隐表征。CD-SPI ~= 0 表示通用基底；接近 1 表示 step polysemy（同形不同义）。

**可证伪命题**：存在某些 step 范畴（如逻辑 connector）跨域 CD-SPI < 0.1，而另一些（如领域名词回指）CD-SPI > 0.5 -> 联邦 PRM 必须分层处理。

### 4.2 Step-Level DP 信息论紧界

**问题**：每个 CoT 包含 T 个 step 标签，对其加 DP 噪声时，单样本 epsilon 的最优分配是？
**核心命题（待证）**：

```
I(reasoning_trace ; PRM_gradient | DP_noise sigma) <= epsilon(sigma, T, label_complexity)
```

并证明 step-level PRM 训练所需 DP 强度**严格大于** outcome-level RM 训练（信息泄漏多 T 倍）。
**实践产出**：DP-SGD 标定表，告诉从业者「在 medical 域 step-level PRM 想达到 epsilon=4 时 sigma 至少要 X」。

---

## 5. 项目目录结构

```
FCL-PRM/
├── README.md                        # 本文件：项目入口
├── claude.md                        # 项目骨架总览
├── 项目介绍.txt                     # 详细中文版介绍
├── requirements.txt                 # Python 依赖
├── setup.py                         # 包安装配置
├── .gitignore                       # 实验输出隔离
│
├── docs/                            # 研究文档
│   ├── design/                      # 技术设计文档
│   │   ├── P1_federated_prm.md
│   │   ├── P2_cd_spi.md
│   │   ├── P3_poisoning.md
│   │   └── P4_dp_steplevel.md
│   ├── related_work.md              # 文献综述
│   ├── milestones.md                # 里程碑追踪
│   └── decisions.md                 # 关键设计决策记录（ADR）
│
├── src/fclprm/                      # 核心代码包
│   ├── __init__.py
│   ├── data/                        # 数据加载器
│   │   ├── __init__.py
│   │   ├── prm800k.py               # PRM800K 加载与 step 分割
│   │   ├── versa_loader.py          # VersaPRM 子集（4 域分流）
│   │   ├── med_loader.py            # MedPRMBench 加载
│   │   └── utils.py                 # 通用 tokenizer / collate / step 分割
│   ├── models/                      # PRM 模型定义
│   │   ├── __init__.py
│   │   ├── prm_head.py              # Step-level reward head（MLP -> 标量）
│   │   ├── base_wrapper.py          # Frozen LLM backbone + PRM head 包装
│   │   └── checkpoint.py            # 保存/加载工具
│   ├── federated/                   # 联邦学习核心
│   │   ├── __init__.py
│   │   ├── client.py                # 本地训练循环（Local PRM + 可选 DP-SGD）
│   │   ├── server.py                # 服务端聚合逻辑
│   │   ├── aggregators.py           # FedAvg-PRM / Anchor-PRM / 鲁棒聚合
│   │   ├── dp.py                    # Step-level DP-SGD 实现与标定
│   │   └── simulator.py             # 单机多进程仿真调度器
│   ├── metrics/                     # 评估与度量
│   │   ├── __init__.py
│   │   ├── cd_spi.py                # CD-SPI 计算
│   │   ├── prm_bench.py             # ProcessBench / PRMBench 评估接口
│   │   ├── bon.py                   # Best-of-N accuracy
│   │   └── privacy.py               # Privacy attack 评估（MIA / reconstruction）
│   ├── attacks/                     # 攻击方法
│   │   ├── __init__.py
│   │   ├── step_poisoning.py        # Step-level label poisoning
│   │   ├── gradient_recon.py        # Reasoning trace reconstruction attack
│   │   └── membership.py            # Membership inference on CoT
│   └── utils/                       # 通用工具
│       ├── __init__.py
│       ├── logging.py               # 结构化日志（JSONL）
│       ├── config.py                # YAML/JSON 配置解析
│       └── seed.py                  # 随机种子管理
│
├── configs/                         # 实验配置文件（YAML）
│   ├── m2_pythia_1b.yaml            # M2 中心化基线（batch 8, len 256）
│   ├── m3_verify_gpu.yaml           # M3 极简 GPU 验证（~20 min）
│   ├── m3_naive_fedavg.yaml         # M3 生产（50 轮, len 256, 4070 友好）
│   ├── m4_anchor_prm.yaml           # M4 生产（100 轮, len 256）
│   ├── m5_cd_spi.yaml               # M5 CD-SPI 测量
│   └── m6_dp_privacy.yaml           # M6 DP-SGD + 隐私攻击
│
├── scripts/                         # 可执行脚本
│   ├── train_centralized_prm.py     # 中心化训练
│   ├── run_federated.py             # 联邦仿真主入口
│   ├── evaluate_prm.py              # 独立评估脚本
│   └── compute_cd_spi.py            # CD-SPI 独立计算
│
├── experiments/                     # 实验目录
│   ├── M2_centralized_prm/
│   ├── M3_naive_fedavg_prm/
│   ├── M4_anchor_prm/
│   ├── M5_cd_spi/
│   └── M6_privacy/
│
└── tests/                           # 单元测试 + 集成测试
    ├── test_data.py
    ├── test_models.py
    ├── test_federated.py
    └── test_metrics.py
```

> **M2 完成**：中心化 PRM baseline 已训练（3 epochs, 17,500 steps, best val_loss 0.00317 / val_acc 99.83%）。详见下方 **M2 结果摘要**。

---

## 5.1 M2 结果摘要

**M2: Centralized PRM Baseline** -- 已于 2026-05-14 完成完整训练。

| 参数 | 值 |
|------|-----|
| Backbone | EleutherAI/pythia-1.4b (frozen) |
| PRM head | MLP(hidden_dim=256) |
| 数据 | PRM800K phase2_train / phase2_test |
| Epochs | 3 |
| Total steps | 17,500 |
| Batch size | 8 |
| Learning rate | 1e-4 (cosine decay) |

### 最佳指标

| Metric | Best Value | At Step | Checkpoint |
|--------|-----------|---------|------------|
| Val Loss | **0.00317** | 14,000 | `model_mM2_r14000_c-1.pt` |
| Val Accuracy | **99.83%** | 4,500 | `model_mM2_r4500_c-1.pt` |
| Val F1 | **0.99914** | 4,500 | `model_mM2_r4500_c-1.pt` |
| Val AUC | **0.97547** | 3,000 | `model_mM2_r3000_c-1.pt` |

**关键观察**：
- 收敛极快：首个 epoch 内 val loss 从 0.0295 降至 0.0034（**8.7x 降幅**）。
- 稳定平台期：step 4,500 后 val loss 在 0.0032-0.0049 区间振荡，无明显过拟合，说明 head 容量与任务匹配。
- AUC 波动较大（0.69-0.98），源于类别不平衡（205K correct vs 4.1K incorrect steps）及验证集较小（727 CoT samples）。
- 训练耗时：GPU 上约 8.5 小时。

完整日志与曲线见 `experiments/M2_centralized_prm/README.md`。

---

## 6. 快速开始

**开发硬件**：单张 RTX 4070 12GB（CUDA）即可跑通全部流程。`max_length` 默认 256，在 4070 上显存占用约 8-10GB；若使用 512，batch 32 会触发 OOM 或导致利用率断崖下跌。

```bash
git clone https://github.com/<user>/FCL-PRM.git
cd FCL-PRM
conda create -n fclprm python=3.11
pip install -r requirements.txt

# 复现 PRM800K 中心化 baseline（~8.5h）
python scripts/train_centralized_prm.py --config configs/m2_pythia_1b.yaml

# M3 极简 GPU 验证（4 客户端 x 50 样本，3 轮，~20 min）
python scripts/run_federated.py --config configs/m3_verify_gpu.yaml

# M3 生产规模（4 客户端 x 5,000 样本，50 轮，~1.5 天）
python scripts/run_federated.py --config configs/m3_naive_fedavg.yaml
```

### 实验数据收集

训练完成后，使用 `experiment_collector.py` 将 JSONL 日志汇总为单个 JSON：

```bash
python src/fclprm/utils/experiment_collector.py \
    --log experiments/M2_centralized_prm/results/logs/m2_centralized_prm_pythia_1b.jsonl \
    --config configs/m2_pythia_1b.yaml \
    --output experiments/experiment_summary.json
```

输出包含 `final_metrics`、`best_metrics`、`epoch_summary`、`raw_points`、`config_summary`，可直接用于 tex 表格 / matplotlib 绘图整理。

---

## 7. 模型 / 数据 / 评估矩阵（RTX 4070 12GB 适配版）

| 阶段 | 基座模型 | 数据 | 评估 | 4070 耗时 |
|---|---|---|---|---|
| M2 baseline | Pythia 1.4B | PRM800K (math) | ProcessBench, BoN@64 | **~8.5 h** |
| M3 verify | Pythia 1.4B | VersaPRM 4 域 x 50 | Val MSE (per-domain) | **~20 min** |
| M3 naive | Pythia 1.4B | VersaPRM 4 域 x 5,000 | ProcessBench (per-domain) | **~1.5 d** |
| M4 anchor | Pythia 1.4B | VersaPRM 4 域 x 5,000 | ProcessBench, CD-SPI | **~3 d** |
| M5 CD-SPI | Pythia 1.4B | VersaPRM 4 域 x 5,000 | CD-SPI taxonomy | **嵌入 M4** |
| M6 隐私 | Pythia 1.4B | VersaPRM 4 域 x 5,000 | DP epsilon-utility, MIA AUC | **~2 d** |

> **4070 效率说明**：`max_length` 从 512 降至 256 后，batch 32 在 4070 上利用率从 ~45% 提升至 ~85%，单 step 耗时从 ~5.5s 降至 ~2.5s。若显存仍吃紧，可再降至 batch 24。

### 实时速率参考（Pythia-1.4B frozen + MLP head，CUDA）

| 配置 | Step 耗时 | 每轮耗时 | 说明 |
|---|---|---|---|
| M2: batch 8, len 256 | **~1.7 s** | — | 单客户端中心化 |
| M3 verify: batch 16, len 256 | **~2.0 s** | **~10 min** | 4 客户端 x 50 样本 x 2 epoch |
| M3/M4 prod: batch 32, len 256 | **~2.5 s** | **~40 min** | 4 客户端 x 5,000 样本 x 2 epoch |
| M6 prod: batch 32, len 256 + DP | **~2.8 s** | **~45 min** | + 梯度裁剪与噪声开销 |

**显存占用**（实测估算）：
- batch 32, len 256, Pythia-1.4B frozen：~8.5 GB
- batch 32, len 512, Pythia-1.4B frozen：~11.5 GB（危险区，易 OOM）

---

## 8. 关键参考文献（按主题分类）

### 8.1 PRM 核心
- Lightman et al. **Let's Verify Step by Step** (arXiv:2305.20050) -- 奠基
- ThinkPRM (ICLR 2026 submission, OpenReview V727xqBYIW)
- VersaPRM (arXiv:2502.06737)
- MedPRMBench (arXiv:2604.17282)
- DreamPRM (arXiv:2505.20241)
- OmegaPRM (Google 2024, MCTS 自动标注)
- PRM Survey (arXiv:2510.08049)

### 8.2 PRM 鲁棒性 / 攻击
- APRM Adversarial Training (arXiv:2511.22888)
- Noise-aware PRM (arXiv:2601.12748)
- PRM-BiasBench
- Preference Poisoning Attacks (arXiv:2402.01920)

### 8.3 Federated RLHF
- FedBis / FedBiscuit (arXiv:2407.03038, ICLR 2025)
- APPA (arXiv:2604.04261)
- FedPDPO (arXiv:2603.19741)
- PluralLLM (arXiv:2503.09925)
- COBRA (Nature SR 2025)

### 8.4 CoT / Reasoning 隐私
- SALT (arXiv:2511.07772)
- Safer Reasoning Traces (arXiv:2603.05618)
- When Reasoning Leaks Membership (arXiv:2601.13607)

### 8.5 联邦学习基础
- FedAvg (McMahan et al., AISTATS 2017)
- SCAFFOLD (Karimireddy et al., ICML 2020)
- FedProx (Li et al., MLSys 2020)
- PFedDL (arXiv:2509.20627)
- **FedGMKD** (Zhang et al., NeurIPS 2024) -- 原型蒸馏 + Discrepancy-Aware Aggregation，class-level Non-IID 标杆

### 8.6 隐私 / 攻击基础
- DP-SGD (Abadi et al., CCS 2016)
- Deep Leakage from Gradients (Zhu et al., NeurIPS 2019)

详细文献见 `docs/related_work.md`。

---

## 9. 引用

```bibtex
@misc{fclprm2026,
  title  = {Federated Continual Process Reward Model: Step-Level Reasoning Verification under Privacy Constraints},
  author = {(TBD)},
  year   = {2026}
}
```

---

## 10. License

待定（计划 MIT / Apache-2.0）。
