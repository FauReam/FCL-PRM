# M3: Naive FedAvg-PRM

## Objective
Establish the failure mode of naive parameter aggregation for step-level PRM.

## Setup
- 4 clients: math, code, medical, general
- Aggregation: FedAvg on PRM head only

## Expected Failure
Step semantic misalignment across domains causes aggregated model to underperform.

## 4070 三档配置

| 档位 | 每客户端样本 | 轮数 | Batch | Length | 设备 | 预估耗时 |
|---|---|---|---|---|---|---|
| **Verify** | 50 | 3 | 16 | 256 | CUDA | **~20 min** |
| **Mini** | 5 | 2 | 16 | 128 | CPU | **~2 min** |
| **Production** | 5,000 | 50 | 32 | 256 | CUDA | **~1.5 天** |

> **为什么 length 从 512 降到 256**：RTX 4070 12GB 在 batch 32 / length 512 下利用率仅 ~45%，单 step ~5.5s，且显存处于 OOM 边缘。降至 256 后利用率恢复至 ~85%，单 step ~2.5s，效率翻倍以上。

## 验证结果

### Verify 档（GPU, 3 轮）
⏳ 待运行。预期 3 轮内 val MSE 跨域 gap 即可显现。

### Mini 档（CPU, 2 轮）
| Metric | Value |
|--------|-------|
| Final avg MSE | 0.6860 |
| Num rounds | 2 |
| Config hash | `c4aea84aa71cad16` |

⏳ Production 档（50 轮）待 verify 确认信号后启动。
