"""
Score CC and baseline runs against SARA ground truth and produce a report.

Designed to be runnable on any output directory (from run.py or
run_baseline.py). Two-run comparison if --compare is given.

Usage:
    # Score a single run
    python score.py --run output/cc_20260510_123000

    # Compare a CC run against a baseline run
    python score.py --run output/cc_20260510_123000 \
                    --compare output/baseline_20260510_124500

Outputs (printed to stdout, also saved):
    - Overall accuracy, with Wilson 95% confidence interval
    - Confusion matrix (Entailment / Contradiction)
    - Per-section accuracy breakdown
    - Tier distribution (CC only)
    - Latency stats (median, p90)
    - If --compare: head-to-head case-by-case win/loss table
"""

from __future__ import annotations
import argparse, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent

# ── Wilson confidence interval ──────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for a binomial proportion. Robust for small n."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

# ── Result loading ──────────────────────────────────────────────────────────

def load_run(run_dir: Path) -> tuple[list[dict], dict]:
    """Return (per-case results, summary)."""
    summary = {}
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())

    cases = []
    for f in sorted(run_dir.glob("SARA-*.json")):
        cases.append(json.loads(f.read_text()))
    return cases, summary

def section_of(case_id: str) -> str:
    m = re.match(r"SARA-S(\d+)-", case_id)
    return m.group(1) if m else "unknown"

# ── Core scoring ────────────────────────────────────────────────────────────

def correct(det: str | None, gt: str) -> bool | None:
    """None if we can't determine correctness (unparseable output)."""
    if det is None:
        return None
    return det.upper() == gt.upper()

def score(results: list[dict]) -> dict:
    n_total = len(results)
    n_parseable = sum(1 for r in results if r.get("determination"))
    n_correct = sum(1 for r in results
                    if correct(r.get("determination"), r["ground_truth"]) is True)
    n_wrong = sum(1 for r in results
                  if correct(r.get("determination"), r["ground_truth"]) is False)
    n_unparseable = n_total - n_parseable

    # Accuracy among all cases (treating unparseable as wrong)
    acc_all = n_correct / n_total if n_total else 0.0
    # Accuracy among parseable only
    acc_parseable = n_correct / n_parseable if n_parseable else 0.0

    ci_all = wilson_ci(n_correct, n_total)
    ci_parseable = wilson_ci(n_correct, n_parseable)

    # Confusion matrix
    cm = defaultdict(int)
    for r in results:
        det = (r.get("determination") or "UNPARSEABLE").upper()
        gt = r["ground_truth"].upper()
        cm[(gt, det)] += 1

    # Per-section accuracy
    per_section = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in results:
        sec = section_of(r["case_id"])
        per_section[sec]["n"] += 1
        if correct(r.get("determination"), r["ground_truth"]) is True:
            per_section[sec]["correct"] += 1
    section_breakdown = {
        sec: {
            "n": d["n"],
            "correct": d["correct"],
            "accuracy": d["correct"] / d["n"] if d["n"] else 0.0,
            "ci": wilson_ci(d["correct"], d["n"]),
        }
        for sec, d in sorted(per_section.items(), key=lambda x: -x[1]["n"])
    }

    # Latency
    elapsed = [r.get("elapsed_total_s") or r.get("elapsed_s") or 0
               for r in results]
    elapsed_sorted = sorted(elapsed)
    def percentile(xs, p):
        if not xs:
            return 0.0
        idx = max(0, min(len(xs) - 1, int(round(p * (len(xs) - 1)))))
        return xs[idx]

    # Tier distribution (CC only — baselines don't have tiers)
    tier_dist = Counter(r.get("tier_applied") or "none" for r in results)

    return {
        "n_total": n_total,
        "n_parseable": n_parseable,
        "n_unparseable": n_unparseable,
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy_all": acc_all,
        "accuracy_all_ci": ci_all,
        "accuracy_parseable": acc_parseable,
        "accuracy_parseable_ci": ci_parseable,
        "confusion_matrix": dict(cm),
        "per_section": section_breakdown,
        "latency_median_s": percentile(elapsed_sorted, 0.5),
        "latency_p90_s": percentile(elapsed_sorted, 0.9),
        "latency_max_s": max(elapsed) if elapsed else 0.0,
        "tier_distribution": dict(tier_dist),
    }

# ── Head-to-head ────────────────────────────────────────────────────────────

def head_to_head(cc_results: list[dict],
                 baseline_results: list[dict]) -> dict:
    cc_by_id = {r["case_id"]: r for r in cc_results}
    bl_by_id = {r["case_id"]: r for r in baseline_results}
    shared = sorted(set(cc_by_id) & set(bl_by_id))

    cc_wins = []     # CC right, baseline wrong
    baseline_wins = []  # baseline right, CC wrong
    both_right = []
    both_wrong = []
    cc_only_parseable = []
    bl_only_parseable = []

    for case_id in shared:
        cc = cc_by_id[case_id]
        bl = bl_by_id[case_id]
        cc_ok = correct(cc.get("determination"), cc["ground_truth"])
        bl_ok = correct(bl.get("determination"), bl["ground_truth"])

        # Both unparseable → skip from win/loss
        if cc_ok is None and bl_ok is None:
            continue
        if cc_ok and bl_ok:
            both_right.append(case_id)
        elif cc_ok is False and bl_ok is False:
            both_wrong.append(case_id)
        elif cc_ok and not bl_ok:
            cc_wins.append(case_id)
        elif bl_ok and not cc_ok:
            baseline_wins.append(case_id)
        elif cc_ok is None:
            bl_only_parseable.append(case_id)
        elif bl_ok is None:
            cc_only_parseable.append(case_id)

    return {
        "n_shared": len(shared),
        "both_right": len(both_right),
        "both_wrong": len(both_wrong),
        "cc_only_right": len(cc_wins),
        "baseline_only_right": len(baseline_wins),
        "cc_only_parseable": len(cc_only_parseable),
        "baseline_only_parseable": len(bl_only_parseable),
        "cc_win_cases": cc_wins[:20],
        "baseline_win_cases": baseline_wins[:20],
    }

# ── Reporting ───────────────────────────────────────────────────────────────

def print_report(label: str, scored: dict):
    print(f"\n{'═' * 70}")
    print(f"  {label}")
    print(f"{'═' * 70}")
    print(f"  Total cases:       {scored['n_total']}")
    print(f"  Parseable:         {scored['n_parseable']} "
          f"(unparseable: {scored['n_unparseable']})")
    print(f"  Correct:           {scored['n_correct']}")
    print(f"  Accuracy (all):    {scored['accuracy_all']:.1%} "
          f"[{scored['accuracy_all_ci'][0]:.1%}, "
          f"{scored['accuracy_all_ci'][1]:.1%}]  Wilson 95% CI")
    print(f"  Accuracy (parse):  {scored['accuracy_parseable']:.1%} "
          f"[{scored['accuracy_parseable_ci'][0]:.1%}, "
          f"{scored['accuracy_parseable_ci'][1]:.1%}]")
    print()
    print(f"  Latency median:    {scored['latency_median_s']:.1f}s")
    print(f"  Latency p90:       {scored['latency_p90_s']:.1f}s")
    print(f"  Latency max:       {scored['latency_max_s']:.1f}s")

    print(f"\n  Confusion matrix (rows = ground truth, cols = prediction):")
    cm = scored["confusion_matrix"]
    labels = ["ENTAILMENT", "CONTRADICTION", "UNPARSEABLE"]
    print(f"    {'':<15}" + "".join(f"{l:>15}" for l in labels))
    for gt in ["ENTAILMENT", "CONTRADICTION"]:
        row = [cm.get((gt, l), 0) for l in labels]
        print(f"    {gt:<15}" + "".join(f"{v:>15}" for v in row))

    print(f"\n  Per-section accuracy:")
    for sec, d in scored["per_section"].items():
        ci = d["ci"]
        print(f"    §{sec:<6} {d['correct']:>3}/{d['n']:<3}  "
              f"{d['accuracy']:>5.1%}  "
              f"[{ci[0]:.1%}, {ci[1]:.1%}]")

    if any(t != "none" for t in scored["tier_distribution"]):
        print(f"\n  Tier distribution:")
        for tier, n in sorted(scored["tier_distribution"].items(),
                              key=lambda x: -x[1]):
            print(f"    {tier:<15} {n:>4}")

def print_h2h(h2h: dict):
    print(f"\n{'═' * 70}")
    print(f"  HEAD-TO-HEAD")
    print(f"{'═' * 70}")
    print(f"  Shared cases:                {h2h['n_shared']}")
    print(f"  Both right:                  {h2h['both_right']}")
    print(f"  Both wrong:                  {h2h['both_wrong']}")
    print(f"  CC only right (CC wins):     {h2h['cc_only_right']}")
    print(f"  Baseline only right:         {h2h['baseline_only_right']}")
    if h2h['cc_only_parseable']:
        print(f"  Only CC parseable:           {h2h['cc_only_parseable']}")
    if h2h['baseline_only_parseable']:
        print(f"  Only baseline parseable:     {h2h['baseline_only_parseable']}")

    # McNemar-style net win (correctly handling discordant pairs)
    discordant = h2h['cc_only_right'] + h2h['baseline_only_right']
    if discordant > 0:
        cc_share = h2h['cc_only_right'] / discordant
        print(f"\n  Of discordant cases: CC wins {cc_share:.1%}, "
              f"baseline wins {1-cc_share:.1%}")

    if h2h["cc_win_cases"]:
        print(f"\n  Sample CC-wins:")
        for cid in h2h["cc_win_cases"][:10]:
            print(f"    {cid}")
    if h2h["baseline_win_cases"]:
        print(f"\n  Sample baseline-wins:")
        for cid in h2h["baseline_win_cases"][:10]:
            print(f"    {cid}")

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Path to a run output directory")
    parser.add_argument("--compare",
                        help="Path to a comparison run (typically baseline)")
    parser.add_argument("--out",
                        help="Save the scored report as JSON to this path")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cc_results, cc_summary = load_run(run_dir)
    cc_scored = score(cc_results)
    print_report(f"RUN: {run_dir.name}", cc_scored)

    full_report = {"primary": {"path": str(run_dir), "scored": cc_scored}}

    if args.compare:
        cmp_dir = Path(args.compare)
        cmp_results, _ = load_run(cmp_dir)
        cmp_scored = score(cmp_results)
        print_report(f"COMPARISON: {cmp_dir.name}", cmp_scored)
        h2h = head_to_head(cc_results, cmp_results)
        print_h2h(h2h)
        full_report["comparison"] = {"path": str(cmp_dir), "scored": cmp_scored}
        full_report["head_to_head"] = h2h

    if args.out:
        Path(args.out).write_text(json.dumps(full_report, indent=2, default=str))
        print(f"\nSaved report → {args.out}")

if __name__ == "__main__":
    main()
