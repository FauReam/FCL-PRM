# M6: Privacy Analysis

## Objective
Evaluate step-level privacy leakage and calibrate DP-SGD.

## Attacks
1. Reasoning trace reconstruction from gradients
2. Membership inference on CoT data

## Defense
Step-level DP-SGD with calibrated noise multiplier.

## Status
✅ M1 verification complete (unit tests)

### Unit Test Verification Results
Privacy attack modules validated via pytest:

| Attack | Metric | Value | Source |
|--------|--------|-------|--------|
| Gradient Reconstruction (DLG) | Cosine grad-dist | 0.9713 | `test_metrics.py` |
| Membership Inference (member) | Score | 0.9945 | `test_metrics.py` |
| Membership Inference (non-member) | Score | 0.8247 | `test_metrics.py` |

**Note**: These are synthetic test values from unit tests with dummy data.
Real attack evaluation requires trained model checkpoints from M4.

⏳ Full privacy evaluation pending M4 checkpoint availability.
