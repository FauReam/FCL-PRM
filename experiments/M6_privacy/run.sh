#!/usr/bin/env bash
# M6: Step-level DP-SGD + privacy attacks (MIA + gradient reconstruction).
# Prereq: opacus installed (`pip install -e .[privacy]`), VersaPRM data ready.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python scripts/run_federated.py \
    --config "$HERE/config.yaml" \
    "$@"
