# M1 代码完善进度 — 2026/05/11

> 状态：33/33 测试通过 | 2/3 任务完成 | 剩余 Task 3 待处理

---

## 已完成的任务

### Task 2: data 模块重构与测试 (completed)

**修改文件：**
- `src/fclprm/data/utils.py` — 新增 `_load_jsonl_or_json()`、`_normalize_dataset()` 公共函数
- `src/fclprm/data/prm800k.py` — 使用公共加载函数，消除重复代码；修复 `tokenize_steps()` 实现与 docstring 不一致
- `src/fclprm/data/versa_loader.py` — 移除硬编码 `DOMAINS`，改为动态发现；使用公共加载函数
- `src/fclprm/data/med_loader.py` — 使用公共加载函数
- `src/fclprm/data/__init__.py` — 补充 `split_cot_into_steps`、`collate_step_batch` 导出
- `tests/test_data.py` — 新增 7 个测试，覆盖 label normalization、文件加载、三个 loader

**测试结果：** 9/9 passed

---

### Task 1: 联邦聚合器测试与 utils (completed)

**修改文件：**
- `src/fclprm/utils/seed.py` — 添加 `deterministic` 参数，设置 `cudnn.deterministic` 和 `cudnn.benchmark`
- `src/fclprm/utils/config.py` — 新增 `require()`（缺失时报错）、`validate_keys()`（批量检查必填键）
- `src/fclprm/utils/__init__.py` — 补充 `ExperimentConfig`、`ExperimentLogger`、`set_seed` 导出
- `tests/test_federated.py` — 新增 4 个 Anchor-PRM 测试：
  - `test_anchor_prm_alignment` — identity permutation 验证函数等价性
  - `test_anchor_prm_insufficient_clients` — <2 clients 时 graceful degradation 到 FedAvg
  - `test_anchor_prm_reference_client_fallback` — reference client 缺失时 fallback
  - `test_anchor_prm_shape_mismatch` — embedding shape 不匹配时抛出 ValueError
- `tests/test_smoke.py` — 新增 5 个端到端 smoke 测试：
  - `test_end_to_end_federated_smoke` — tiny-gpt2 + 2 clients + FedAvg，1 round 完整训练
  - `test_end_to_end_anchor_prm_smoke` — tiny-gpt2 + 2 clients + Anchor-PRM，含 anchor embedding 提取
  - `test_collate_integration` — DataLoader + collate_step_batch 端到端
  - `test_checkpoint_save_load` — checkpoint 命名规则和 round-trip 验证
  - `test_config_hash_reproducibility` — 配置哈希可复现性 + validate_keys

**测试结果：** 8/8 (federated) + 5/5 (smoke) = 13/13 passed

---

## 总体测试状态

```
pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-9.0.3
collected 33 items

tests/test_data.py     :: 9 passed
tests/test_federated.py:: 8 passed
tests/test_metrics.py  :: 6 passed
tests/test_models.py   :: 5 passed
tests/test_smoke.py    :: 5 passed

======================= 33 passed, 2 warnings in 14.25s ========================
```

---

## 已完成的任务

### Task 3: scripts 修复与包导出 (completed)

**scripts/ 修复：**
- `scripts/train_centralized_prm.py` — `local_files_only=True` 改为默认 `False`，添加下载失败时的友好错误提示
- `scripts/run_federated.py` — 同上
- `scripts/evaluate_prm.py` — 添加 `--dataset` 参数（支持 `prm800k` / `versaprm`）和 `--domain` 参数，扩展 VersaPRM 多域评估能力
- `scripts/compute_cd_spi.py` — 支持从 `checkpoint_dir` 自动加载各客户端最新 checkpoint；anchor steps 可配置化；添加下载失败错误处理

**包导出完善：**
- `src/fclprm/metrics/__init__.py` — 补充 `ProcessBenchEvaluator`、`best_of_n_accuracy`、`evaluate_reconstruction_attack`、`evaluate_membership_inference`
- `src/fclprm/attacks/__init__.py` — 导出 `StepPoisoningAttack`、`GradientReconstructionAttack`、`MembershipInferenceAttack`
- `src/fclprm/federated/__init__.py` — 补充 `fedavg_prm`、`anchor_prm_aggregate`、`robust_aggregate_trimmed_mean`
- `src/fclprm/models/__init__.py` — 补充 `save_checkpoint`、`load_checkpoint`

---

## 总体测试状态

```
pytest tests/ -v
============================= test session starts ==============================
platform darwin -- Python 3.10.11, pytest-9.0.3
collected 33 items

tests/test_data.py     :: 9 passed
tests/test_federated.py:: 8 passed
tests/test_metrics.py  :: 6 passed
tests/test_models.py   :: 5 passed
tests/test_smoke.py    :: 5 passed

======================= 33 passed, 2 warnings in 17.34s ========================
```

---

## 回归检查点

以下数值来自 commit `a9a2a38`，任何代码修改导致以下值偏移 >1% 需排查：

| 配置 | 指标 | 值 |
|---|---|---|
| M3 smoke (2 clients, FedAvg) | Final avg MSE | 0.3483 |
| M4 smoke (2 clients, Anchor-PRM) | Final avg MSE | 0.3483 |
| M2 mini (centralized) | Validation MSE | 0.1576 |
| M3 mini (4 clients, FedAvg) | Final avg MSE | 0.6860 |
| M4 mini (4 clients, Anchor-PRM) | Final avg MSE | 0.4747 |
| DLG reconstruction | Cosine grad-dist | 0.9713 |
| MIA member | Score | 0.9945 |
| MIA non-member | Score | 0.8247 |

---

## M1 代码完善已完成

全部 3 个 Task 完成，33/33 测试通过，无回归。
