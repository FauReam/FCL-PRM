#!/usr/bin/env bash
# M3: Naive FedAvg-PRM, 4-client federated simulation.
# Prereq: VersaPRM data at ./data/versaprm/versa_prm.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python scripts/run_federated.py \
    --config "$HERE/config.yaml" \
    "$@"
