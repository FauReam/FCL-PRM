#!/usr/bin/env bash
# M2: Centralized PRM baseline on PRM800K.
# Prereq: PRM800K data downloaded to ./data/prm800k/{train,val,test}.jsonl
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python scripts/train_centralized_prm.py \
    --config "$HERE/config.yaml" \
    "$@"
