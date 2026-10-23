"""Parallel, resuming downloader for MIT-BIH records from PhysioNet.

PhysioNet throttles *per connection*, so this module is built around
small range requests instead of whole files:

1. probe each file's total size with a 1-byte range GET;
2. split it into ~256 KB chunks and fetch every chunk of every record in one
   shared thread pool (chunk-level retries are cheap, unlike whole-file ones);
3. assemble the chunks in order and verify the final size byte-for-byte.

Returns the record ids that could not be completed, so callers can fail loudly
instead of silently training on truncated signals.
"""

from __future__ import annotations

import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import config

BASE_URLS = [
    "https://physionet.org/files/mitdb/1.0.0",
    "https://www.physionet.org/physiobank/database/mitdb",
]
EXTENSIONS = ("hea", "dat", "atr")
WORKERS = 48
CHUNK = 256 * 1024


def _unlink(path: Path) -> None:
    """Remove a part file, retrying while Windows still holds the handle open.

    A worker thread may not have flushed/closed its handle when the main
    thread assembles a record; Windows then refuses the unlink, so retry
    briefly and leave the file behind rather than crashing the run.
    """
    for delay in (0.0, 0.1, 0.5, 1.0, 2.0):
        try:
            path.unlink(missing_ok=True)
            return
        except PermissionError:
            if delay:
                time.sleep(delay)


def _open(url: str, headers: dict) -> urllib.request.Response:
    req = urllib.request.Request(url, headers=headers)
    return urllib.request.urlopen(req, timeout=300)


def remote_total(url: str) -> int | None:
    """Total size in bytes, read from a 1-byte range request."""
    try:
        with _open(url, {"Range": "bytes=0-0"}) as r:
            content_range = r.headers.get("Content-Range", "")
            if "/" in content_range:
                return int(content_range.split("/")[-1])
            size = int(r.headers.get("Content-Length", 0))
            return size or None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def fetch_range(url: str, dest: Path, start: int, end: int, retries: int = 8) -> bool:
    """Fetch bytes [start, end) into dest; retries on a dropped connection."""
    if dest.exists() and dest.stat().st_size == end - start:
        return True
    for attempt in range(retries):
        try:
            with _open(url, {"Range": f"bytes={start}-{end - 1}"}) as resp, open(dest, "wb") as fh:
                shutil.copyfileobj(resp, fh, length=1 << 20)
            if dest.stat().st_size == end - start:
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(1.0 * (attempt + 1))
    return False


def _chunks(total: int) -> list[tuple[int, int]]:
    return [(i, min(i + CHUNK, total)) for i in range(0, total, CHUNK)]


def fetch_file(url: str, dest: Path, workers: int = 4) -> bool:
    """Complete one file via parallel range chunks, then verify its size."""
    total = remote_total(url)
    if total is None:
        return False
    if dest.exists() and dest.stat().st_size == total:
        return True  # already complete
    ranges = _chunks(total)
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(len(ranges))]
    results = [fetch_range(url, part, s, e) for part, (s, e) in zip(parts, ranges)]
    ok = all(results)
    if not all(ok):
        for part in parts:
            part.unlink(missing_ok=True)
        return False
    with open(dest, "wb") as out:
        for part in parts:
            with open(part, "rb") as fh:
                shutil.copyfileobj(fh, out)
            part.unlink()
    return dest.stat().st_size == total


def download_records(record_ids: list[str], dest_dir: Path = config.RAW_DIR,
                     workers: int = WORKERS) -> list[str]:
    """Download every record's .hea/.dat/.atr; returns ids that failed."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    ids = sorted(set(record_ids))
    failed: set[str] = set()

    for base in BASE_URLS:
        # Probe all remote sizes in parallel: a single range GET costs seconds
        # on PhysioNet, so a sequential probe of 132 files alone takes ~20 min.
        urls = [(rid, ext, f"{base}/{rid}.{ext}") for rid in ids for ext in EXTENSIONS]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            sizes = list(pool.map(lambda t: remote_total(t[2]), urls))
        print(f"  probed {sum(s is not None for s in sizes)}/{len(urls)} sizes on {base}",
              flush=True)

        tasks, owners = [], []
        for (rid, ext, url), total in zip(urls, sizes):
            if total is None:
                continue
            dest = dest_dir / f"{rid}.{ext}"
            if dest.exists() and dest.stat().st_size == total:
                continue  # already complete
            for i, (s, e) in enumerate(_chunks(total)):
                part = dest.with_name(f"{dest.name}.part{i}")
                if part.exists() and part.stat().st_size == e - s:
                    continue
                tasks.append((url, part, s, e))
                owners.append((rid, ext))
        if not tasks:
            print("  nothing left to fetch", flush=True)
            continue
        print(f"  {len(tasks)} chunks to fetch ({workers} workers)", flush=True)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch_range, url, part, s, e): (rid, ext, part)
                       for (url, part, s, e), (rid, ext) in zip(tasks, owners)}
            done = 0
            for fut in as_completed(futures):
                rid, ext, part = futures[fut]
                done += 1
                if not fut.result():
                    _unlink(part)
                if done % 100 == 0 or done == len(tasks):
                    print(f"    {done}/{len(tasks)} chunks", flush=True)

    # Assemble and validate every record.
    for rid in ids:
        for ext in EXTENSIONS:
            dest = dest_dir / f"{rid}.{ext}"
            total = remote_total(f"{BASE_URLS[0]}/{rid}.{ext}")
            if total is None:
                continue
            parts = sorted(dest.parent.glob(f"{dest.name}.part*"),
                           key=lambda p: int(p.name.rsplit("part", 1)[1]))
            if parts:
                with open(dest, "wb") as out:
                    for part in parts:
                        with open(part, "rb") as fh:
                            shutil.copyfileobj(fh, out)
                        _unlink(part)
            if not (dest.exists() and dest.stat().st_size == total):
                failed.add(rid)
    if failed:
        print(f"  WARNING incomplete: {sorted(failed)}")
    return sorted(failed)
