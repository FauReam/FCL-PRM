# M2: Centralized PRM Baseline

## Objective
Reproduce PRM800K centralized training as the foundation for all subsequent experiments.

## Model
- Backbone: EleutherAI/pythia-1.4b
- Head: MLP (hidden_dim=256)
- Backbone frozen, only head trained

## Data
- PRM800K phase2_train / phase2_test
- Source: OpenAI GitHub (nested format: `question.problem`, `question.pre_generated_steps`, `label.steps[].completions[].rating`)
- `PRM800KLoader` adapted to parse nested OpenAI format (2026-05-11)
- Train: 18,227 samples | Test: 727 samples
- Validation: 10% holdout from train

## Metrics
- Step-level accuracy
- Best-of-N@64
- ProcessBench score

## Status
✅ M1 verification complete | ✅ Data ready | ✅ Loader adapted | ✅ **Full training complete**

---

## Full Training Results (2026-05-13 → 2026-05-14)

### Configuration
| Parameter | Value |
|-----------|-------|
| Backbone | EleutherAI/pythia-1.4b |
| PRM head | MLP(hidden_dim=256) |
| Epochs | 3 |
| Total steps | 17,500 |
| Batch size | 8 |
| Max length | 256 |
| Learning rate | 1e-4 (cosine decay) |
| Device | GPU |
| Config hash | `abb059108ea61052` |

### Convergence Curve

| Step | Epoch | Train Loss | Val Loss | Val Accuracy | Val F1 | Val AUC |
|------|-------|------------|----------|--------------|--------|---------|
| 500 | 1 | 0.0150 | 0.0295 | 97.55% | 0.9876 | 0.9108 |
| 1,000 | 1 | 0.0342 | 0.0115 | 99.81% | 0.9990 | 0.7506 |
| 2,000 | 1 | 0.0102 | 0.0085 | 99.22% | 0.9961 | 0.9487 |
| 3,000 | 1 | 0.0098 | 0.0051 | 99.76% | 0.9988 | **0.9755** |
| 4,500 | 1 | 0.0021 | 0.0034 | **99.83%** | **0.9991** | 0.9507 |
| 6,000 | 2 | 0.0007 | 0.0041 | 99.78% | 0.9989 | 0.9073 |
| 9,000 | 2 | 0.0024 | 0.0045 | 99.64% | 0.9982 | 0.9463 |
| 12,000 | 3 | 0.0342 | 0.0038 | 99.72% | 0.9986 | 0.9412 |
| 14,000 | 3 | 0.0007 | **0.0032** | 99.78% | 0.9989 | 0.8451 |
| 17,500 | 3 | 0.0005 | 0.0036 | 99.69% | 0.9985 | 0.8948 |

### Best Checkpoints

| Metric | Best Value | At Step | Checkpoint |
|--------|-----------|---------|------------|
| Val Loss | **0.00317** | 14,000 | `model_mM2_r14000_c-1.pt` |
| Val Accuracy | **99.83%** | 4,500 | `model_mM2_r4500_c-1.pt` |
| Val F1 | **0.99914** | 4,500 | `model_mM2_r4500_c-1.pt` |
| Val AUC | **0.97547** | 3,000 | `model_mM2_r3000_c-1.pt` |

### Key Observations

1. **Fast convergence**: Val loss drops from 0.0295 (step 500) to 0.0034 (step 4,500) — **8.7× reduction within the first epoch**.
2. **Stable plateau**: After step 4,500, val loss oscillates in the 0.0032–0.0049 band without clear overfitting, suggesting the head capacity is well-matched to the task.
3. **AUC volatility**: AUC fluctuates significantly (0.69–0.98) across steps, likely due to class imbalance (205K correct vs 4.1K incorrect steps) and small validation set (727 CoT samples).
4. **Training time**: ~8.5 hours for 3 epochs on GPU.

### Comparison with Related Work

| Work | Setting | Data | Best Metric | Notes |
|------|---------|------|-------------|-------|
| **FCL-PRM (M2)** | Centralized PRM | PRM800K phase2 | Val Acc 99.83% / Loss 0.0032 | Frozen Pythia-1.4b + MLP head |
| Lightman et al. (PRM800K) | Centralized PRM | PRM800K | Human-level step verification | Full finetune, larger model |
| VersaPRM | Centralized multi-domain | 14 domains | Domain-avg F1 0.89 | LLaMA-3.1 8B, full finetune |
| FedGMKD (Zhang et al., NeurIPS 2024) | Federated image cls | CIFAR-10/100, SVHN | +3.2% global acc vs FedAvg | Prototype KD + discrepancy-aware aggregation; **not step-level** |

> **Boundary with FedGMKD**: FedGMKD addresses Non-IID heterogeneity in *image classification* via prototype-based knowledge distillation and discrepancy-aware aggregation (Equations 8–11 in their paper). FCL-PRM differs fundamentally: (1) we operate on *step-level* rewards, not class predictions; (2) our heterogeneity is *semantic polysemy* of reasoning steps, not covariate shift; (3) our aggregation challenge is *embedding space alignment*, not prototype clustering. FedGMKD is cited as a methodological neighbor in `docs/related_work.md`.

---

## Historical Verification Runs

### Mini Verification Results
Quick validation with reduced data to verify code-path integrity:

| Step | Validation MSE |
|------|---------------|
| 10   | 0.2112        |
| 20   | 0.3009        |
| 30   | 0.1576        |

- **Best val_loss**: 0.1576 at step 30
- **Config hash**: `0d5f11b1892044a6`
- **Run date**: 2026-05-06

### Post-Loader-Fix Mini Run (2026-05-11)
验证修复后的 `PRM800KLoader` 能正确解析 OpenAI 嵌套格式并端到端训练：

| 参数 | 值 |
|------|-----|
| Data | mini_train (200 CoT → 448 step samples) |
| Epochs | 1 |
| Batch size | 8 |
| Max length | 256 |
| Device | CPU |
| Avg train loss | **0.1401** |
| Runtime | ~2 min |

✅ 数据流验证通过：嵌套格式解析 → step-level 展开 → collate → 训练 → loss 收敛

---

## Artifacts

```
experiments/M2_centralized_prm/results/
├── checkpoints/
│   ├── model_mM2_r1_c-1.pt        # epoch 1 end
│   ├── model_mM2_r2_c-1.pt        # epoch 2 end
│   ├── model_mM2_r3_c-1.pt        # epoch 3 end
│   ├── model_mM2_r14000_c-1.pt    # best val_loss
│   └── model_mM2_r17655_c-1.pt    # final step
├── logs/
│   └── m2_centralized_prm_pythia_1b.jsonl   # 35 records
└── cache/
    └── step_dataset_90e28e5e14b93694b3d0491da02b5a56.pt
```

## Next Steps
- [ ] Run ProcessBench evaluation on `model_mM2_r14000_c-1.pt`
- [ ] Compute Best-of-N@64 on GSM8K-style candidates
- [ ] Use M2 checkpoint to initialize M3 (naive FedAvg-PRM) global model
