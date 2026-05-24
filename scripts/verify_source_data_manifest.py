"""Verify checksums for the lightweight paper source-data snapshots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "paper_source_data"
MANIFEST = DATA / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = manifest.get("source_data_hashes", {})
    if not expected:
        raise SystemExit("manifest.json does not contain source_data_hashes")

    failures: list[str] = []
    for name, meta in sorted(expected.items()):
        path = DATA / name
        if not path.exists():
            failures.append(f"missing: {name}")
            continue
        size = path.stat().st_size
        sha = _sha256(path)
        if size != int(meta["bytes"]):
            failures.append(f"{name}: bytes {size} != {meta['bytes']}")
        if sha != str(meta["sha256"]):
            failures.append(f"{name}: sha256 {sha} != {meta['sha256']}")

    if failures:
        raise SystemExit("\n".join(failures))

    print(json.dumps({"checked_files": len(expected), "status": "ok"}, indent=2))


if __name__ == "__main__":
    main()
