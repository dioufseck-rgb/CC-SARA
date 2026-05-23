#!/usr/bin/env python3
"""
Rebuild a summary from the per-case JSONs in an output directory.

Useful for:
- Mid-run progress checks (summary.json hasn't been written yet)
- Recovering from a crashed run (summary.json never got written)
- Verifying an existing summary.json against the on-disk JSONs

Usage:
    python rebuild_summary.py output/cc_v6_test_n100
"""

import json
import sys
from pathlib import Path


def rebuild(out_dir: Path) -> dict:
    case_files = sorted(f for f in out_dir.glob("*.json") if f.name != "summary.json")

    results = []
    for cf in case_files:
        try:
            data = json.loads(cf.read_text())
            results.append({
                "case_id": data.get("case_id") or cf.stem,
                "determination": data.get("determination") or data.get("result_summary", {}).get("determination"),
                "tier_applied": data.get("tier_applied"),
                "elapsed_s": data.get("elapsed_total_s"),
                "status": data.get("status"),
            })
        except Exception as e:
            results.append({"case_id": cf.stem, "status": f"parse_error: {e!r}"})

    n_completed = sum(
        1 for r in results
        if r.get("determination") in ("ENTAILMENT", "CONTRADICTION")
    )

    elapsed_values = [r.get("elapsed_s", 0) for r in results if r.get("elapsed_s")]
    avg_elapsed = sum(elapsed_values) / max(len(elapsed_values), 1)

    tier_counts = {}
    for r in results:
        t = r.get("tier_applied") or "none"
        tier_counts[t] = tier_counts.get(t, 0) + 1

    return {
        "label": out_dir.name,
        "n_cases": len(results),
        "n_completed": n_completed,
        "tier_distribution": tier_counts,
        "avg_elapsed_s": avg_elapsed,
        "case_results": results,
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python rebuild_summary.py <output_dir>")
        return 1

    out_dir = Path(sys.argv[1])
    if not out_dir.is_dir():
        print(f"Not a directory: {out_dir}")
        return 1

    summary = rebuild(out_dir)

    # Write to summary_rebuilt.json so we don't clobber the real one
    out_path = out_dir / "summary_rebuilt.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"Rebuilt summary from {summary['n_cases']} per-case JSONs")
    print(f"  Completed:    {summary['n_completed']}/{summary['n_cases']}")
    print(f"  Avg latency:  {summary['avg_elapsed_s']:.1f}s/case")
    print(f"  Tier distribution: {summary['tier_distribution']}")
    print(f"  Written to:   {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())