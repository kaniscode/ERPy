#!/usr/bin/env python3
"""Download public OpenNeuro ds004774 objects from a saved inventory.

Downloads are atomic, verified for byte count and simple-object ETag, and
recorded with SHA-256. No protected ERPy illustration data are used.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import quote
from urllib.request import Request, urlopen


def download(item: dict, destination: Path) -> dict:
    key = item["key"]
    if not key.startswith("ds004774/") or ".." in Path(key).parts:
        raise ValueError("Unexpected dataset key")
    target = destination / key
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = int(item["bytes"])
    etag = item.get("etag", "").strip('"')
    url = "https://s3.amazonaws.com/openneuro.org/" + quote(key, safe="/")
    for attempt in range(4):
        try:
            if not target.is_file() or target.stat().st_size != expected:
                partial = target.with_name(target.name + ".partial")
                request = Request(url, headers={"User-Agent": "ERPy-public-validation/1.0"})
                with urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                if partial.stat().st_size != expected:
                    raise ValueError(f"Size mismatch for {key}")
                payload = partial.read_bytes()
                if len(etag) == 32 and hashlib.md5(payload).hexdigest() != etag:
                    raise ValueError(f"ETag mismatch for {key}")
                partial.replace(target)
            payload = target.read_bytes()
            if len(etag) == 32 and hashlib.md5(payload).hexdigest() != etag:
                raise ValueError(f"Cached ETag mismatch for {key}")
            return {"key": key, "url": url, "bytes": expected, "etag": etag,
                    "sha256": hashlib.sha256(payload).hexdigest()}
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("Unreachable download state")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inventory", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--subject", action="append", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--archived-calls", action="store_true", help="Download only each subject’s archived erdetect_data.mat comparator output")
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    subjects = set(args.subject)
    if args.archived_calls:
        selected = [x for x in inventory if x["key"].endswith("/erdetect_data.mat") and any(x["key"].startswith(f"ds004774/derivatives/app_detect_output/sub-{s}_") for s in subjects)]
    else:
        selected = [x for x in inventory if any(x["key"].startswith(f"ds004774/sub-{s}/") for s in subjects)]
    if not selected:
        parser.error("No objects matched the supplied subjects")
    args.destination.mkdir(parents=True, exist_ok=True)
    manifest = args.destination / ("manifest_" + ("archived_" if args.archived_calls else "") + "_".join(sorted(subjects)) + ".jsonl")
    print(f"Downloading {len(selected)} objects, {sum(x['bytes'] for x in selected)/1e6:.1f} MB", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, manifest.open("w") as out:
        tasks = [pool.submit(download, x, args.destination) for x in selected]
        for count, future in enumerate(as_completed(tasks), 1):
            out.write(json.dumps(future.result()) + "\n")
            out.flush()
            if count % 25 == 0 or count == len(tasks):
                print(f"Verified {count}/{len(tasks)} objects", flush=True)
