#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/data/checkpoints/diffsbdd"
URL="https://zenodo.org/record/8183747/files/crossdocked_fullatom_cond.ckpt?download=1"

mkdir -p "$OUT"
if command -v aria2c >/dev/null 2>&1; then
  aria2c -c -x 8 -s 8 -k 1M \
    --retry-wait=5 --max-tries=20 --timeout=60 --connect-timeout=30 \
    --summary-interval=30 -d "$OUT" -o crossdocked_fullatom_cond.ckpt "$URL"
else
  curl -L --retry 10 --retry-delay 5 -C - -o "$OUT/crossdocked_fullatom_cond.ckpt" "$URL"
fi
ls -lh "$OUT/crossdocked_fullatom_cond.ckpt"
