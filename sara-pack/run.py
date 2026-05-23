"""
Run Cognitive Core against SARA Binary cases.

Usage:
    # Single case by ID
    python run.py SARA-S151-A-NEG

    # Named subset
    python run.py --split test --section 151
    python run.py --split dev --all
    python run.py --split train --section 7703 --limit 5

    # Full subset
    python run.py --split test --all

Outputs:
    output/<run_label>/<case_id>.json  — full trace + final disposition per case
    output/<run_label>/summary.json    — per-run summary (counts, latency, accuracy)

The runner does NOT score against ground truth during execution. Scoring is
a separate step (score.py) to enforce the discipline that we never look at
ground truth while CC is running. Ground truth is in the case files (since
we have to write it somewhere) but the workflow does not read it — only
case_narrative, question_text, and statute_corpus.
"""

from __future__ import annotations
import argparse, json, os, re, sys, time
from datetime import datetime
from pathlib import Path

# ── Path setup ──────────────────────────────────────────────────────────────
PACK_DIR = Path(__file__).resolve().parent
CASES_DIR = PACK_DIR / "cases"
OUTPUT_DIR = PACK_DIR / "output"

# Locate the cognitive_core install — assume it's a sibling of this pack
# or pip-installed. Try sibling first, fall back to system.
REPO_CANDIDATE = PACK_DIR.parent / "cognitive-core-main"
if REPO_CANDIDATE.exists():
    sys.path.insert(0, str(REPO_CANDIDATE))

from cognitive_core.coordinator.runtime import Coordinator
from cognitive_core.engine.trace import TraceCallback, set_trace, NullTrace

# ── Case selection ──────────────────────────────────────────────────────────

def section_of(case_id: str) -> str:
    m = re.match(r"SARA-S(\d+)-", case_id)
    return m.group(1) if m else "unknown"

def select_cases(args) -> list[Path]:
    if args.case_id:
        # Find the case file across all binary directories
        for split in ("train", "dev", "test"):
            candidate = CASES_DIR / "binary" / split / f"{args.case_id}.json"
            if candidate.exists():
                return [candidate]
        for split in ("train", "test"):
            candidate = CASES_DIR / "numeric" / split / f"{args.case_id}.json"
            if candidate.exists():
                return [candidate]
        raise FileNotFoundError(f"Case not found: {args.case_id}")

    # Filter by split and optionally section
    task = args.task or "binary"
    split = args.split
    if not split:
        raise ValueError("Must specify --split or a case_id")
    case_dir = CASES_DIR / task / split
    if not case_dir.exists():
        raise FileNotFoundError(f"No such split: {case_dir}")

    candidates = sorted(case_dir.glob("*.json"))
    if args.section:
        candidates = [c for c in candidates if section_of(c.stem) == args.section]
    if args.limit:
        candidates = candidates[: args.limit]
    return candidates

# ── Case input construction ─────────────────────────────────────────────────

def build_case_input(case: dict) -> dict:
    """
    Build the case_input dict for the coordinator.

    Keys here become tools in the case registry. The workflow's retrieve
    steps reference them via the `sources:` param:
      retrieve_question_structure  → sources: question_text
      retrieve_statute             → sources: statute_corpus
      retrieve_case_facts          → sources: case_narrative

    Ground truth (`ground_truth`, `prolog_*`) is intentionally excluded.
    """
    return {
        "case_id":        case["case_id"],
        # The three retrievable surfaces — these become tools the workflow reads
        "question_text":  case["question_text"],
        "statute_corpus": case["statute_corpus"],
        "case_narrative": case["case_narrative"],
    }

# ── Trace callback for per-step logging ─────────────────────────────────────

ICONS = {
    "retrieve":    "📥",
    "classify":    "🏷 ",
    "investigate": "🔍",
    "verify":      "✅",
    "deliberate":  "🤔",
    "generate":    "📝",
    "challenge":   "⚔️ ",
    "reflect":     "🪞",
    "govern":      "⚖️ ",
}

class SaraTrace(TraceCallback):
    """Capture step outputs via on_parse_result; engine doesn't emit on_step_end."""
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.step_records: list[dict] = []
        self._step_start_times: dict[str, float] = {}
        self._current_step = ""
        self._current_primitive = ""

    def on_step_start(self, step_name, primitive, loop_iteration):
        self._current_step = step_name
        self._current_primitive = primitive
        self._step_start_times[step_name] = time.time()
        if self.verbose:
            icon = ICONS.get(primitive, "  ")
            print(f"  {icon} {step_name:<35} [{primitive}] ...", flush=True)

    def on_parse_result(self, step_name, primitive, output):
        elapsed = (time.time() - self._step_start_times.get(step_name, time.time())) * 1000
        conf = output.get("confidence") if isinstance(output, dict) else None
        rec = {
            "step_name": step_name,
            "primitive": primitive,
            "confidence": conf,
            "elapsed_ms": elapsed,
            "output": output if isinstance(output, dict) else {"raw": str(output)},
        }
        self.step_records.append(rec)
        if self.verbose:
            conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "?"
            print(f"     ✓ {step_name} conf={conf_str} ({elapsed/1000:.1f}s)")
            self._print_summary(primitive, output)

    def on_parse_error(self, step_name, error):
        if self.verbose:
            print(f"     ✗ {step_name} parse error: {str(error)[:200]}")
        self.step_records.append({
            "step_name": step_name,
            "primitive": self._current_primitive,
            "parse_error": str(error)[:500],
        })

    def on_llm_start(self, step_name, prompt_chars):
        if self.verbose:
            print(f"       [llm in]  {prompt_chars:,} chars", flush=True)

    def on_llm_end(self, step_name, response_chars, elapsed):
        if self.verbose:
            print(f"       [llm out] {response_chars:,} chars  {elapsed:.1f}s")

    def on_retrieve_start(self, step_name, source_name):
        if self.verbose:
            print(f"       [retrieve→] {source_name}")

    def on_retrieve_end(self, step_name, source_name, status, latency_ms):
        if self.verbose:
            print(f"       [retrieve←] {source_name} {status} {latency_ms:.0f}ms")

    def on_route_decision(self, *a, **kw): pass

    def _print_summary(self, primitive, output):
        if not isinstance(output, dict):
            return
        if primitive == "retrieve":
            srcs = output.get("sources_queried", [])
            ok = sum(1 for s in srcs if s.get("status") == "success")
            print(f"       → {ok}/{len(srcs)} sources")
        elif primitive == "investigate":
            f = (output.get("finding") or "")[:120]
            print(f"       → {f}")
        elif primitive == "verify":
            print(f"       → conforms={output.get('conforms')}  "
                  f"violations={len(output.get('violations', []))}")
        elif primitive == "challenge":
            print(f"       → survives={output.get('survives')}  "
                  f"vulns={len(output.get('vulnerabilities', []))}")
        elif primitive == "generate":
            art = output.get("artifact")
            if isinstance(art, dict):
                print(f"       → {art.get('disposition', '?')}")
            else:
                print(f"       → {str(art)[:120]}")
        elif primitive == "govern":
            tier = str(output.get("tier_applied", "?")).replace("GovernanceTier.", "")
            print(f"       → tier={tier}")

# ── Single-case execution ───────────────────────────────────────────────────

def run_case(case_path: Path, coordinator: Coordinator,
             out_dir: Path, verbose: bool = True) -> dict:
    case = json.loads(case_path.read_text())
    case_id = case["case_id"]

    if verbose:
        print(f"\n{'─' * 70}")
        print(f"Case: {case_id} ({case['source_id']})")
        print(f"  Facts:    {case['case_narrative'][:120]}...")
        print(f"  Question: {case['question_text'][:120]}")
        print(f"{'─' * 70}")

    case_input = build_case_input(case)
    trace = SaraTrace(verbose=verbose)
    set_trace(trace)

    t0 = time.time()
    instance_id = None
    status = "unknown"
    determination = None
    tier = None
    result_summary = None
    error = None

    try:
        instance_id = coordinator.start(
            workflow_type="sara_binary",
            domain="sara_us_federal_tax",
            case_input=case_input,
        )
        elapsed_total = time.time() - t0

        # Retrieve final state via store
        instance = coordinator.store.get_instance(instance_id)
        if instance:
            status = str(instance.status.value if hasattr(instance.status, "value")
                         else instance.status).lower()
            result_summary = instance.result
            tier = str(getattr(instance, "governance_tier", "")).lower() \
                       .replace("governancetier.", "")
            # Determination should live in result.determination or in the
            # generate step's artifact
            if isinstance(result_summary, dict):
                # Top-level (_extract_result_summary surfaces it)
                determination = result_summary.get("determination") or \
                                result_summary.get("disposition")
                # If artifact is a dict with 'disposition', pull from there
                for s in result_summary.get("steps", []):
                    if s.get("primitive") == "generate":
                        art = s.get("artifact")
                        if isinstance(art, dict) and "disposition" in art:
                            determination = art["disposition"]
                        break
            # Normalize to canonical ENTAILMENT/CONTRADICTION if present in text
            if isinstance(determination, str):
                up = determination.upper()
                if "ENTAILMENT" in up and "CONTRADICTION" not in up:
                    determination = "ENTAILMENT"
                elif "CONTRADICTION" in up and "ENTAILMENT" not in up:
                    determination = "CONTRADICTION"
    except Exception as e:
        elapsed_total = time.time() - t0
        status = f"error: {type(e).__name__}"
        error = repr(e)
        if verbose:
            print(f"  ✗ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    result = {
        "case_id": case_id,
        "source_id": case["source_id"],
        "instance_id": instance_id,
        "status": status,
        "error": error,
        "elapsed_total_s": round(elapsed_total, 2),
        "tier_applied": tier,
        "determination": determination,
        "ground_truth": case["ground_truth"],   # for the scorer; NOT used by CC
        "step_records": trace.step_records,
        "result_summary": result_summary,
    }

    out_path = out_dir / f"{case_id}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))

    # Detach trace so it doesn't leak into the next case
    set_trace(NullTrace())

    if verbose:
        print(f"\n  → determination: {determination}  tier: {tier}  "
              f"total: {elapsed_total:.1f}s")
        print(f"  → saved: {out_path}")

    return result

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run CC on SARA cases")
    parser.add_argument("case_id", nargs="?",
                        help="A single case ID to run (e.g. SARA-S151-A-NEG)")
    parser.add_argument("--task", choices=["binary", "numeric"], default="binary")
    parser.add_argument("--split", choices=["train", "dev", "test"])
    parser.add_argument("--section", help="Restrict to cases for IRC section (e.g. 151)")
    parser.add_argument("--all", action="store_true",
                        help="Run all matching cases (no limit)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap number of cases to run")
    parser.add_argument("--label", default=None,
                        help="Label this run (default: timestamp)")
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

    label = args.label or datetime.now().strftime("cc_%Y%m%d_%H%M%S")
    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    coordinator = Coordinator(
        config_path=str(PACK_DIR / "coordinator_config.yaml"),
        db_path=str(PACK_DIR / "sara_cc.db"),
        verbose=not args.quiet,
    )
    # The Coordinator derives workflow_dir/domain_dir/case_dir from the
    # directory of config_path. Since coordinator_config.yaml lives at the
    # pack root and references workflows/, domains/, this is correct as-is.

    # ── LLM preflight ────────────────────────────────────────────────────
    # The coordinator silently falls back to simulation if no LLM is
    # configured. We want a loud, early failure instead.
    try:
        from cognitive_core.engine.llm import create_llm
        _ = create_llm(model="default", temperature=0.1)
        print(f"✓ LLM preflight OK")
    except Exception as e:
        print(f"\n✗ LLM PREFLIGHT FAILED — coordinator would fall back to "
              f"simulation, which doesn't actually exercise reasoning.")
        print(f"  Error: {type(e).__name__}: {e}")
        print(f"\n  Set one of:")
        print(f"    export GOOGLE_API_KEY=...     (Gemini, primary provider)")
        print(f"    export ANTHROPIC_API_KEY=...  (Claude)")
        print(f"    export OPENAI_API_KEY=...     (GPT)")
        print(f"  And ensure llm_config.yaml in the cognitive-core root "
              f"points to your chosen provider.")
        return 2

    print(f"Running {len(cases)} case(s). Output → {out_dir}")
    results = []
    for i, case_path in enumerate(cases, 1):
        print(f"\n[{i}/{len(cases)}]", end="")
        try:
            r = run_case(case_path, coordinator, out_dir, verbose=not args.quiet)
            results.append(r)
        except Exception as e:
            print(f"\n  ✗ case-level error: {e!r}")
            results.append({
                "case_id": case_path.stem,
                "status": f"runner_error: {e!r}",
            })

    # Write per-run summary
    summary = {
        "label": label,
        "started_at": label,  # timestamp is embedded in label by default
        "n_cases": len(results),
        "n_completed": sum(1 for r in results if r.get("determination") in
                           ("ENTAILMENT", "CONTRADICTION")),
        "tier_distribution": {},
        "avg_elapsed_s": (
            sum(r.get("elapsed_total_s", 0) for r in results) / max(len(results), 1)
        ),
        "case_results": [
            {
                "case_id": r["case_id"],
                "determination": r.get("determination"),
                "tier_applied": r.get("tier_applied"),
                "elapsed_s": r.get("elapsed_total_s"),
                "status": r.get("status"),
            }
            for r in results
        ],
    }
    # Tier breakdown
    tier_counts = {}
    for r in results:
        t = r.get("tier_applied") or "none"
        tier_counts[t] = tier_counts.get(t, 0) + 1
    summary["tier_distribution"] = tier_counts

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'─' * 70}")
    print(f"Run summary: {out_dir / 'summary.json'}")
    print(f"  Completed:    {summary['n_completed']}/{summary['n_cases']}")
    print(f"  Avg latency:  {summary['avg_elapsed_s']:.1f}s/case")
    print(f"  Tier distribution: {tier_counts}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
