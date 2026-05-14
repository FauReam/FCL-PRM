# M4: Anchor-PRM with Step Embedding Alignment

## Objective
Propose and validate Anchor-PRM: align client step embeddings before aggregation.

## Key Idea
Use a small shared anchor step set to calibrate cross-client embedding spaces.

## Innovation
First attempt at cross-domain step embedding alignment in federated PRM.

## Status
✅ M1 verification complete (mini run)

### Mini Verification Results
Quick validation with 2 rounds, 4 clients, Anchor-PRM aggregation:

| Metric | Value |
|--------|-------|
| Final avg MSE | 0.4747 |
| Num rounds | 2 |
| Config hash | `15f7dbb30c782a82` |

**Note**: Anchor-PRM (0.4747) vs FedAvg (0.6860) shows ~31% loss reduction
in mini validation, suggesting alignment is beneficial.

⏳ Full simulation (100 rounds) pending GPU allocation.
