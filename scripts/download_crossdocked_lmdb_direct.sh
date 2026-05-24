#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUT="$ROOT/data/raw/if3-crossdocked2020"
BASE="https://hf-mirror.com/datasets/Yukk1Zz/if3-crossdocked2020/resolve/main"
mkdir -p "$OUT/train.lmdb" "$OUT/val.lmdb"

download_split() {
  local split="$1"
  local url="$BASE/${split}.lmdb/data.mdb"
  local dir="$OUT/${split}.lmdb"
  echo "Downloading ${split}.lmdb/data.mdb"
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 16 -s 16 -k 1M \
      --retry-wait=5 --max-tries=20 --timeout=60 --connect-timeout=30 \
      --summary-interval=30 -d "$dir" -o data.mdb "$url"
  else
    curl -L --retry 10 --retry-delay 5 -C - -o "$dir/data.mdb" "$url"
  fi
  touch "$dir/lock.mdb"
  ls -lh "$dir"
}

for split in ${SPLITS:-train val}; do
  download_split "$split"
done
