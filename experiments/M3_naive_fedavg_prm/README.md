# M3: Naive FedAvg-PRM

## Objective
Establish the failure mode of naive parameter aggregation for step-level PRM.

## Setup
- 4 clients: math, code, medical, general
- 5000 samples per client
- Aggregation: FedAvg on PRM head only

## Expected Failure
Step semantic misalignment across domains causes aggregated model to underperform.

## Status
✅ M1 verification complete (mini run)

### Mini Verification Results
Quick validation with 2 rounds, 4 clients:

| Metric | Value |
|--------|-------|
| Final avg MSE | 0.6860 |
| Num rounds | 2 |
| Config hash | `c4aea84aa71cad16` |

⏳ Full simulation (50 rounds) pending GPU allocation.
