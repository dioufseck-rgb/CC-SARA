"""
Run a monolithic chain-of-thought baseline on SARA Binary cases.

This is the fair comparison point for CC. Same model, same input
(statute_corpus + case_narrative + question_text), single LLM call with
explicit chain-of-thought, structured final answer.

The point of having this in the same pack with the same case selection
mechanics is to ensure we never accidentally compare runs on different
case sets, different model versions, or different prompt structures.

Usage:
    python run_baseline.py SARA-S151-A-NEG
    python run_baseline.py --split test --all
    python run_baseline.py --split dev --section 151
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
CASES_DIR = PACK_DIR / "cases"
OUTPUT_DIR = PACK_DIR / "output"

# Reuse path setup from main runner
REPO_CANDIDATE = PACK_DIR.parent / "cognitive-core-main"
if REPO_CANDIDATE.exists():
    sys.path.insert(0, str(REPO_CANDIDATE))

from cognitive_core.engine.llm import create_llm

BASELINE_PROMPT_TEMPLATE = """You are evaluating a statutory entailment claim under US Federal Tax Law.

Below are the relevant statute sections, the case facts, and the claim. Determine whether the claim is ENTAILMENT (follows from the statute applied to the facts) or CONTRADICTION (does not follow).

Think step by step:
  1. Identify the IRC section(s) the claim references.
  2. Identify the case facts relevant to those section(s).
  3. Apply the section(s) to the facts and derive the rule's output.
  4. Compare the rule's output to the claim's assertion.
  5. Consider any subsection exceptions or cross-references that could change the result.
  6. Conclude with your determination.

End your response with a final line of EXACTLY this form (no other text on that line):
Answer: ENTAILMENT
or
Answer: CONTRADICTION

─── STATUTES ─────────────────────────────────────────────────────────────────
{statute_corpus}

─── CASE FACTS ───────────────────────────────────────────────────────────────
{case_narrative}

─── CLAIM ────────────────────────────────────────────────────────────────────
{question_text}
"""

ANSWER_RE = re.compile(r"^\s*Answer\s*:\s*(ENTAILMENT|CONTRADICTION)\b",
                       re.IGNORECASE | re.MULTILINE)

def section_of(case_id: str) -> str:
    m = re.match(r"SARA-S(\d+)-", case_id)
    return m.group(1) if m else "unknown"

def select_cases(args) -> list[Path]:
    """Identical selection logic to run.py — keep in sync."""
    if args.case_id:
        for split in ("train", "dev", "test"):
            candidate = CASES_DIR / "binary" / split / f"{args.case_id}.json"
            if candidate.exists():
                return [candidate]
        raise FileNotFoundError(f"Case not found: {args.case_id}")

    task = args.task or "binary"
    if not args.split:
        raise ValueError("Must specify --split or a case_id")
    case_dir = CASES_DIR / task / args.split
    candidates = sorted(case_dir.glob("*.json"))
    if args.section:
        candidates = [c for c in candidates if section_of(c.stem) == args.section]
    if args.limit:
        candidates = candidates[: args.limit]
    return candidates

def parse_answer(text: str) -> str | None:
    """Extract the final 'Answer: X' from the model output."""
    matches = ANSWER_RE.findall(text)
    if not matches:
        # Fallback: look for the word anywhere in the last 200 chars
        tail = text[-200:].upper()
        if "ENTAILMENT" in tail and "CONTRADICTION" not in tail:
            return "ENTAILMENT"
        if "CONTRADICTION" in tail and "ENTAILMENT" not in tail:
            return "CONTRADICTION"
        return None
    # Use the last occurrence — model may discuss both before committing
    return matches[-1].upper()

def run_case(case_path: Path, llm, out_dir: Path,
             verbose: bool = True) -> dict:
    case = json.loads(case_path.read_text())
    case_id = case["case_id"]

    if verbose:
        print(f"\n  Case: {case_id}")
        print(f"    Q: {case['question_text'][:120]}")

    prompt = BASELINE_PROMPT_TEMPLATE.format(
        statute_corpus=case["statute_corpus"],
        case_narrative=case["case_narrative"],
        question_text=case["question_text"],
    )

    t0 = time.time()
    try:
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content=prompt)])
        # Some providers wrap content in lists; normalize.
        raw = response.content if isinstance(response.content, str) \
              else " ".join(b.get("text", "") if isinstance(b, dict) else str(b)
                            for b in response.content)
        elapsed = time.time() - t0
        determination = parse_answer(raw)
        status = "ok" if determination else "unparseable"
    except Exception as e:
        elapsed = time.time() - t0
        raw = ""
        determination = None
        status = f"error: {e!r}"

    result = {
        "case_id": case_id,
        "source_id": case["source_id"],
        "elapsed_s": round(elapsed, 2),
        "determination": determination,
        "status": status,
        "ground_truth": case["ground_truth"],
        "raw_response": raw,
        "prompt_chars": len(prompt),
    }

    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(result, indent=2))

    if verbose:
        print(f"    → {determination}  ({elapsed:.1f}s)")
    return result

def main():
    parser = argparse.ArgumentParser(description="Baseline monolithic LLM on SARA")
    parser.add_argument("case_id", nargs="?")
    parser.add_argument("--task", choices=["binary", "numeric"], default="binary")
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--section")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--label", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if not args.case_id and not args.split:
        parser.error("Provide a case_id or --split")
    if args.split and not args.all and not args.limit and not args.case_id:
        parser.error("With --split, also provide --all or --limit")

    cases = select_cases(args)
    if not cases:
        print("No cases selected.")
        return 1

    label = args.label or datetime.now().strftime("baseline_%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = create_llm()  # uses the same llm_config.yaml as CC; matched model
    print(f"Baseline run: {len(cases)} case(s) → {out_dir}")
    print(f"Model: {getattr(llm, 'model', getattr(llm, 'model_name', 'unknown'))}")

    results = []
    for i, case_path in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}]", end="")
        try:
            r = run_case(case_path, llm, out_dir, verbose=not args.quiet)
            results.append(r)
        except Exception as e:
            print(f"\n  ✗ {e!r}")
            results.append({"case_id": case_path.stem,
                            "status": f"runner_error: {e!r}"})

    summary = {
        "label": label,
        "n_cases": len(results),
        "n_completed": sum(1 for r in results
                           if r.get("determination") in
                           ("ENTAILMENT", "CONTRADICTION")),
        "avg_elapsed_s": (
            sum(r.get("elapsed_s", 0) for r in results) / max(len(results), 1)
        ),
        "case_results": [
            {
                "case_id": r["case_id"],
                "determination": r.get("determination"),
                "elapsed_s": r.get("elapsed_s"),
                "status": r.get("status"),
            }
            for r in results
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'─' * 70}")
    print(f"Completed: {summary['n_completed']}/{summary['n_cases']}")
    print(f"Avg:       {summary['avg_elapsed_s']:.1f}s/case")
    return 0

if __name__ == "__main__":
    sys.exit(main())
