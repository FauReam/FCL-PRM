#!/usr/bin/env bash
# M5: CD-SPI cross-domain step polysemy measurement.
# Prereq: per-client checkpoints from M4 in experiments/M4_anchor_prm/results/checkpoints/
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

python scripts/compute_cd_spi.py \
    --config "$HERE/config.yaml" \
    "$@"
