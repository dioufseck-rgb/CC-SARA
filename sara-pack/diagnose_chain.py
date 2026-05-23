"""
Diagnostic: extract the key chain-state fields from a CC run's outputs
so failure patterns can be read side-by-side rather than buried in full traces.

For each case in a run directory, print:
  - Disposition (CC) vs ground_truth
  - retrieve_statute: identified subsections
  - investigate: rule-output finding (the model's stated derived value)
  - verify: conforms boolean + any violations
  - challenge: survives boolean + any vulnerabilities
  - generate.artifact: disposition + subsections_cited + key reasoning
  - govern: tier + rationale (first sentence)

Usage:
    python diagnose_chain.py --run output/cc_hard_batch_n8
    python diagnose_chain.py --run output/cc_hard_batch_n8 --case SARA-S1-A-1-III-NEG
"""

from __future__ import annotations
import argparse, json, sys, textwrap
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
CASES_DIR = PACK_DIR / "cases"


def find_step(records: list, primitive: str) -> dict | None:
    for rec in records:
        if rec.get("primitive") == primitive:
            return rec.get("output", {})
    return None


def find_step_from_summary(summary: dict, primitive: str) -> dict | None:
    for step in summary.get("steps", []) or []:
        if step.get("primitive") == primitive:
            return step
    return None


def short(text: str, n: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = " ".join(text.split())
    if len(text) <= n:
        return text
    return text[:n] + "…"


def diagnose_case(cc_result: dict, gt_case: dict) -> None:
    cid = cc_result.get("case_id", "?")
    gt = gt_case.get("ground_truth", "?")
    det = cc_result.get("determination", "?")
    tier = cc_result.get("tier_applied", "?")

    correct_marker = "✓" if det and gt and det[0].upper() == gt[0].upper() else "✗"
    print()
    print("=" * 78)
    print(f"  {cid}  ({correct_marker})")
    print(f"  question     : {gt_case.get('question_text', '')}")
    print(f"  narrative    : {short(gt_case.get('case_narrative', ''), 200)}")
    print(f"  ground_truth : {gt}")
    print(f"  CC disposition: {det}   tier: {tier}")
    print("-" * 78)

    records = cc_result.get("step_records") or []
    summary = cc_result.get("result_summary") or {}

    # retrieve_statute
    rs = find_step(records, "retrieve") or {}
    # In our pack there are 3 retrieve steps; find the statute one by source
    rs_statute = None
    for rec in records:
        if rec.get("primitive") == "retrieve":
            data = rec.get("output", {}).get("data", {}) or {}
            if "statute_corpus" in data:
                rs_statute = rec.get("output", {})
                break
    if rs_statute:
        # The structured digest is the LLM-emitted summary, accessible via
        # reasoning/data; we want the structured subsection identifiers
        data = rs_statute.get("data", {}).get("statute_corpus") or {}
        # The LLM may emit primary_subsections or similar in its reasoning
        # data; show whatever structure is there.
        reasoning = rs_statute.get("reasoning", "")
        print(f"  retrieve_statute reasoning: {short(reasoning, 250)}")

    # investigate
    inv = find_step(records, "investigate") or {}
    finding = inv.get("finding", "")
    print(f"\n  INVESTIGATE finding ({inv.get('confidence', '?')}):")
    print(textwrap.fill(short(finding, 500), width=76, initial_indent="    ", subsequent_indent="    "))
    hyps = inv.get("hypotheses_tested") or []
    if hyps:
        print(f"  hypotheses tested: {len(hyps)}")
        for h in hyps[:6]:
            hyp_text = h.get("hypothesis") if isinstance(h, dict) else str(h)
            status = h.get("status", "?") if isinstance(h, dict) else "?"
            print(f"    [{status}] {short(hyp_text, 110)}")

    # verify
    ver = find_step(records, "verify") or {}
    print(f"\n  VERIFY conforms={ver.get('conforms', '?')}  conf={ver.get('confidence', '?')}")
    print(f"    rules_checked: {ver.get('rules_checked', [])}")
    for v in (ver.get("violations") or []):
        sev = v.get("severity", "?") if isinstance(v, dict) else "?"
        desc = v.get("description", "") if isinstance(v, dict) else str(v)
        print(f"    violation [{sev}]: {short(desc, 150)}")

    # challenge
    ch = find_step(records, "challenge") or {}
    print(f"\n  CHALLENGE survives={ch.get('survives', '?')}  conf={ch.get('confidence', '?')}")
    for v in (ch.get("vulnerabilities") or []):
        sev = v.get("severity", "?") if isinstance(v, dict) else "?"
        desc = v.get("description", "") if isinstance(v, dict) else str(v)
        print(f"    vuln [{sev}]: {short(desc, 150)}")

    # generate
    gen = find_step(records, "generate") or {}
    art = gen.get("artifact") if isinstance(gen.get("artifact"), dict) else {}
    print(f"\n  GENERATE disposition={art.get('disposition', '?')}  conf={gen.get('confidence', '?')}")
    print(f"    primary_reasoning: {short(art.get('primary_reasoning', ''), 400)}")
    print(f"    subsections_cited: {art.get('subsections_cited', [])}")
    print(f"    case_facts_cited:  {art.get('case_facts_cited', [])}")

    # govern
    gov = find_step(records, "govern") or {}
    rat = gov.get("tier_rationale", "")
    print(f"\n  GOVERN tier_applied={gov.get('tier_applied', '?')}  conf={gov.get('confidence', '?')}")
    print(f"    rationale: {short(rat, 200)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="CC run directory under output/")
    ap.add_argument("--case", help="Optional single case_id to inspect")
    ap.add_argument("--wrong-only", action="store_true",
                    help="Only show cases where CC disagreed with ground_truth")
    args = ap.parse_args()

    run_dir = Path(args.run)
    files = sorted(run_dir.glob("SARA-*.json"))
    if not files:
        print(f"No SARA-*.json in {run_dir}", file=sys.stderr)
        return 1

    for f in files:
        cc = json.loads(f.read_text())
        cid = cc.get("case_id", "")
        if args.case and cid != args.case:
            continue
        # Load ground truth from the case file
        gt_path = None
        for sub in ["train", "dev", "test"]:
            p = CASES_DIR / "binary" / sub / f"{cid}.json"
            if p.exists():
                gt_path = p
                break
        if not gt_path:
            print(f"No case file for {cid}", file=sys.stderr)
            continue
        gt_case = json.loads(gt_path.read_text())

        if args.wrong_only:
            gt = (gt_case.get("ground_truth") or "").lower()
            det = (cc.get("determination") or "").lower()
            if gt and det and gt.startswith(det[:3]):
                continue

        diagnose_case(cc, gt_case)
    return 0


if __name__ == "__main__":
    sys.exit(main())
