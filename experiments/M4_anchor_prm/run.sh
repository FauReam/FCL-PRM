#!/usr/bin/env bash
# M4: Anchor-PRM with cross-client step embedding alignment.
# Prereq: VersaPRM data at ./data/versaprm/versa_prm.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python scripts/run_federated.py \
    --config "$HERE/config.yaml" \
    "$@"
