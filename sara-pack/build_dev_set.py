"""
Carve out a frozen dev set from binary/train, stratified by section.

The dev set is used as a regression check during iteration on the domain YAML
and primitive prompts. It must never be tuned against directly. To enforce
that operationally, we move the case files into a separate sibling directory
(cases/binary/dev/) so that any code that iterates "all train cases" naturally
excludes them.

Strategy: take ~12% of train (≈ 20 cases) stratified by SARA section.
Fixed random seed for reproducibility.
"""

from __future__ import annotations
import json, random, re, shutil
from collections import defaultdict
from pathlib import Path

CASES_DIR = Path("/home/claude/sara-pack/cases")
TRAIN_DIR = CASES_DIR / "binary" / "train"
DEV_DIR   = CASES_DIR / "binary" / "dev"

DEV_FRACTION = 0.12
SEED = 7

def section_of(case_id: str) -> str:
    # SARA-S151-D-1-NEG  →  151
    m = re.match(r"SARA-S(\d+)-", case_id)
    return m.group(1) if m else "unknown"

def main():
    DEV_DIR.mkdir(parents=True, exist_ok=True)

    # If dev already populated, abort (idempotent guard)
    existing_dev = list(DEV_DIR.glob("*.json"))
    if existing_dev:
        print(f"Dev set already has {len(existing_dev)} cases. Skipping.")
        return

    rng = random.Random(SEED)

    # Group train cases by section
    by_section = defaultdict(list)
    for case_file in sorted(TRAIN_DIR.glob("*.json")):
        by_section[section_of(case_file.stem)].append(case_file)

    print(f"Train cases by section:")
    for sec, files in sorted(by_section.items(), key=lambda x: -len(x[1])):
        print(f"  §{sec}: {len(files)} cases")

    # Stratified sample
    moved = []
    for section, files in by_section.items():
        n_dev = max(1, round(len(files) * DEV_FRACTION))
        sample = rng.sample(files, n_dev)
        for src in sample:
            dst = DEV_DIR / src.name
            shutil.move(str(src), str(dst))
            moved.append((section, src.name))

    print(f"\nMoved {len(moved)} cases from train → dev:")
    for sec, name in sorted(moved):
        print(f"  §{sec}: {name}")
    print(f"\nRemaining train: {len(list(TRAIN_DIR.glob('*.json')))}")
    print(f"Dev:             {len(list(DEV_DIR.glob('*.json')))}")

    # Update split manifest
    manifest_path = CASES_DIR / "splits.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["binary"]["train"] = sorted([
        p.stem for p in TRAIN_DIR.glob("*.json")
    ])
    manifest["binary"]["dev"] = sorted([
        p.stem for p in DEV_DIR.glob("*.json")
    ])
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nUpdated manifest: {manifest_path}")

if __name__ == "__main__":
    main()
