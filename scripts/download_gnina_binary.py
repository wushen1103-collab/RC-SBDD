import argparse
import json
import os
import time
import urllib.request
from pathlib import Path


GNINA_URLS = [
    "https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8",
    "https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2",
    "https://gh.llkk.cc/https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8",
    "https://gh.llkk.cc/https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2",
    "https://gh-proxy.com/https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2.cuda12.8",
    "https://gh-proxy.com/https://github.com/gnina/gnina/releases/download/v1.3.2/gnina.1.3.2",
]


def try_download(url, out_path, connect_timeout, max_seconds):
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    start = tmp.stat().st_size if tmp.exists() else 0
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    if start:
        req.add_header("Range", f"bytes={start}-")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=connect_timeout) as response, open(tmp, "ab" if start else "wb") as handle:
        total = response.headers.get("Content-Length")
        total = int(total) + start if total else None
        done = start
        while True:
            if time.time() - t0 > max_seconds:
                raise TimeoutError(f"download exceeded {max_seconds} seconds")
            chunk = response.read(1024 * 1024 * 8)
            if not chunk:
                break
            handle.write(chunk)
            done += len(chunk)
            if total and done >= total:
                break
    tmp.rename(out_path)
    os.chmod(out_path, 0o755)
    return out_path.stat().st_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="tools/gnina")
    ap.add_argument("--audit-json", default="logs/gnina_download_audit.json")
    ap.add_argument("--connect-timeout", type=int, default=90)
    ap.add_argument("--max-seconds-per-url", type=int, default=900)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    attempts = []
    if out_path.exists() and out_path.stat().st_size > 100_000_000:
        result = {"ok": True, "path": str(out_path), "size": out_path.stat().st_size, "attempts": attempts, "reused": True}
    else:
        result = {"ok": False, "path": str(out_path), "attempts": attempts, "reused": False}
        for url in GNINA_URLS:
            try:
                size = try_download(url, out_path, args.connect_timeout, args.max_seconds_per_url)
                result.update({"ok": True, "url": url, "size": size})
                break
            except Exception as exc:
                attempts.append({"url": url, "error": f"{type(exc).__name__}: {exc}"})
    Path(args.audit_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.audit_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
