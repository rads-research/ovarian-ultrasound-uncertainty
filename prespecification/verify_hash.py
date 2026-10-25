#!/usr/bin/env python3
"""Verify the prespecification file against its recorded SHA-256 hash.

    python verify_hash.py            # check, exit 1 on mismatch
    python verify_hash.py --write    # record the current hash (first time only)

Call check() before loading any contrast-enhanced ultrasound data.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = HERE / "prespecification.yaml"
HASH_FILE = HERE / "prespecification.sha256"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(strict: bool = True) -> bool:
    if not PLAN.exists():
        raise FileNotFoundError(f"Prespecification file missing: {PLAN}")
    if not HASH_FILE.exists():
        raise FileNotFoundError(
            f"No recorded hash at {HASH_FILE}. Run with --write once, then "
            "confirm the printed value matches the manuscript.")

    recorded = HASH_FILE.read_text().split()[0].strip()
    actual = sha256(PLAN)
    if actual != recorded:
        message = (
            "PRESPECIFICATION HASH MISMATCH\n"
            f"  recorded: {recorded}\n"
            f"  actual:   {actual}\n"
            "The locked analysis plan has changed since it was recorded. "
            "Do not proceed to the cross-modality data.")
        if strict:
            raise RuntimeError(message)
        print(message, file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="record the current hash (refuses to overwrite)")
    args = ap.parse_args()

    if args.write:
        if HASH_FILE.exists():
            print(f"{HASH_FILE.name} already exists; refusing to overwrite. "
                  "Delete it deliberately if you really mean to re-lock.",
                  file=sys.stderr)
            return 1
        digest = sha256(PLAN)
        HASH_FILE.write_text(f"{digest}  {PLAN.name}\n")
        print(f"Recorded {digest}")
        print("Confirm this matches the hash quoted in the manuscript.")
        return 0

    try:
        check()
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Prespecification hash verified: {sha256(PLAN)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
