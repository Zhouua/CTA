"""Patch data/product_registry.json to populate mid_weekly_files from xlsx.

Scope: T4.1 of docs/mid_weekly_integration_plan.md.

Convention: each file in ``--cleaned-dir`` is named ``<PID>.xlsx`` and
the matching registry entry receives ``mid_weekly_files = ["<PID>.xlsx"]``.
The ``_cleaned/`` prefix is *not* written into the registry — at read time
the pipeline expects ``paths.mid_weekly_dir`` to be pointed at
``data/mid_weekly/_cleaned``.

Products without a cleaned xlsx keep ``mid_weekly_files = []``. The
script prints a diff summary and writes the registry back in-place.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="data/product_registry.json")
    parser.add_argument("--cleaned-dir", default="data/mid_weekly/_cleaned")
    parser.add_argument("--dry-run", action="store_true", help="Print diff without writing")
    args = parser.parse_args(argv)

    registry_path = Path(args.registry)
    cleaned_dir = Path(args.cleaned_dir)
    if not registry_path.exists():
        print(f"ERROR: registry {registry_path} missing", file=sys.stderr)
        return 2
    if not cleaned_dir.exists():
        print(f"ERROR: cleaned dir {cleaned_dir} missing", file=sys.stderr)
        return 2

    available = {p.stem.upper(): p.name for p in sorted(cleaned_dir.glob("*.xlsx")) if not p.name.startswith("~")}
    if not available:
        print(f"ERROR: no xlsx files under {cleaned_dir}", file=sys.stderr)
        return 2

    with registry_path.open(encoding="utf-8") as f:
        registry = json.load(f)
    if not isinstance(registry, list):
        print("ERROR: registry is not a list", file=sys.stderr)
        return 2

    assigned: list[str] = []
    unchanged: list[str] = []
    cleared: list[str] = []
    orphan_xlsx: set[str] = set(available.keys())

    for entry in registry:
        pid = str(entry.get("product_id", "")).upper()
        current = list(entry.get("mid_weekly_files", []) or [])
        target: list[str]
        if pid in available:
            target = [available[pid]]
            orphan_xlsx.discard(pid)
        else:
            target = []
        if target != current:
            if target:
                assigned.append(f"{pid}: {current} -> {target}")
            else:
                cleared.append(f"{pid}: {current} -> []")
            entry["mid_weekly_files"] = target
        else:
            unchanged.append(pid)

    print(f"Assigned mid_weekly_files ({len(assigned)}):")
    for line in assigned:
        print(f"  + {line}")
    if cleared:
        print(f"\nCleared stale mid_weekly_files ({len(cleared)}):")
        for line in cleared:
            print(f"  - {line}")
    print(f"\nUnchanged: {len(unchanged)} entries")
    if orphan_xlsx:
        print(f"\nWARNING: xlsx files with no matching registry entry: {sorted(orphan_xlsx)}")

    if args.dry_run:
        print("\n[dry-run] registry not modified")
        return 0

    with registry_path.open("w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {registry_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
