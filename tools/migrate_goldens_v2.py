#!/usr/bin/env python3
"""One-time migration: golden-file format v1 -> v2 (FINDING 18).

v1 header: magic(u32) | version=1(u32) | dtype(u32) | rank(u32) | shape | payload
v2 header: magic(u32) | version=2(u32) | recorded_at_epoch(u64) | dtype(u32) | rank(u32) | shape | payload

recorded_at_epoch is backfilled from `git log -1 --format=%ct -- <path>` (the
most recent commit that touched the file) for tracked files, or the current
wall-clock time for untracked (newly-added, uncommitted) files -- both
correctly reflect "when was this content actually captured."

Run once from the repo root: python3 tools/migrate_goldens_v2.py
Safe to re-run: files already at v2 are skipped.
"""
import glob
import os
import struct
import subprocess
import sys
import time

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "backend_parity", "golden")
MAGIC = 0x444C4754


def git_last_touch_times(golden_dir):
    """One `git log` invocation over the whole golden dir; newest-first, so
    the first epoch time seen per filename is its most recent commit."""
    out = subprocess.run(
        ["git", "log", "--format=%x00%ct", "--name-only", "--", golden_dir],
        cwd=os.path.join(os.path.dirname(__file__), ".."),
        capture_output=True, text=True, check=True,
    ).stdout
    times = {}
    current_ct = None
    for line in out.splitlines():
        if line.startswith("\x00"):
            current_ct = int(line[1:])
        elif line.strip().endswith(".gold"):
            fname = os.path.basename(line.strip())
            if fname not in times:
                times[fname] = current_ct
    return times


def migrate_file(path, recorded_at):
    with open(path, "rb") as f:
        data = f.read()
    magic, version = struct.unpack_from("<II", data, 0)
    if magic != MAGIC:
        print(f"SKIP (bad magic): {path}", file=sys.stderr)
        return False
    if version == 2:
        return False  # already migrated
    if version != 1:
        print(f"SKIP (unknown version {version}): {path}", file=sys.stderr)
        return False
    rest = data[8:]  # everything after magic+version: dtype|rank|shape|payload
    new_header = struct.pack("<IIQ", MAGIC, 2, recorded_at)
    with open(path + ".tmp", "wb") as f:
        f.write(new_header + rest)
    os.replace(path + ".tmp", path)
    return True


def main():
    files = sorted(glob.glob(os.path.join(GOLDEN_DIR, "*.gold")))
    print(f"Found {len(files)} .gold files")
    touch_times = git_last_touch_times(GOLDEN_DIR)
    print(f"git log covers {len(touch_times)} distinct filenames")
    now = int(time.time())
    migrated = 0
    untracked = 0
    for path in files:
        fname = os.path.basename(path)
        recorded_at = touch_times.get(fname)
        if recorded_at is None:
            recorded_at = now
            untracked += 1
        if migrate_file(path, recorded_at):
            migrated += 1
    print(f"Migrated {migrated} files ({untracked} untracked -> stamped with current time)")


if __name__ == "__main__":
    main()
