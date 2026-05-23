"""
Prolog-faithfulness analysis for CC runs on SARA cases.

This is Layer 2 of the evaluation: not "did CC reach the right disposition"
(Layer 1, handled by score.py), but "did CC's cited subsections and facts
correspond to the Prolog ground truth's predicates and asserted facts?"

The Prolog ground truth in each SARA case file (prolog_facts + prolog_test)
is the symbolic encoding of:
  - What the case asserts as fact (e.g., s151_c(alice,_,2000,2015) asserts
    Alice has a §151(c) exemption of $2,000 for 2015)
  - What the test checks (e.g., :- \+ s151_a(alice,6000,2015) asserts the
    Prolog engine should fail to prove s151_a(alice,6000,2015) — Contradiction)

CC's run output contains:
  - investigate.evidence_used: cited subsections and case facts
  - generate.artifact.subsections_cited: final disposition's subsection citations
  - generate.artifact.case_facts_cited: final disposition's fact citations

We compare these vocabularies after normalization. Both express the same
statutory content; comparing them gives us a faithfulness measure.

Output metrics, per case:
  - subsection_precision: of subsections CC cited, fraction that correspond
    to predicates appearing in the Prolog
  - subsection_recall:    of predicates appearing in the Prolog, fraction
    CC also cited
  - over_attribution:     subsections CC cited that have no Prolog correlate
  - under_attribution:    Prolog predicates CC failed to cite

Aggregate across run:
  - mean precision/recall with Wilson CIs
  - distribution of over- and under-attribution counts

Usage:
    python score_prolog.py --run output/cc_<label>
    python score_prolog.py --run output/cc_<label> --out report.json
"""

from __future__ import annotations
import argparse, json, math, re, sys
from collections import Counter, defaultdict
from pathlib import Path

PACK_DIR = Path(__file__).resolve().parent
CASES_DIR = PACK_DIR / "cases"


# ── Prolog vocabulary parsing ─────────────────────────────────────────────────

# Prolog statute predicates have the form s<num>(_<letter_or_num>)*
# Examples: s151_a, s151_d_3, s7703_b, s3306_a_1_A, s151_c_applies
# We extract these from prolog_facts and prolog_test and normalize to
# subsection identifiers.
_PREDICATE_RE = re.compile(r'\bs(\d+(?:_[a-zA-Z0-9]+)*)\b')

# Suffixes that indicate the predicate is a derived/auxiliary form, not a
# direct subsection reference. We strip these for normalization.
_DERIVED_SUFFIXES = {"applies", "satisfied", "holds"}


def predicate_to_subsection(predicate: str) -> str | None:
    """
    Normalize a Prolog predicate to a canonical subsection identifier.

    Examples:
        s151_a           → §151(a)
        s151_d_3         → §151(d)(3)
        s151_b_applies   → §151(b)              (strip 'applies' suffix)
        s3306_a_1_A      → §3306(a)(1)(A)
        s7703_b          → §7703(b)
        s63              → §63
        s2_a             → §2(a)

    Returns None for non-statute predicates (which shouldn't appear, but
    defensive).
    """
    m = _PREDICATE_RE.fullmatch("s" + predicate.lstrip("s"))
    if not m:
        return None
    parts = m.group(1).split("_")
    if not parts:
        return None

    # First part is the section number (digits only)
    if not parts[0].isdigit():
        return None
    section = parts[0]

    # Strip derived suffixes
    while parts and parts[-1].lower() in _DERIVED_SUFFIXES:
        parts.pop()

    if len(parts) == 1:
        # Just the section, no subsection
        return f"§{section}"

    # Remaining parts are subsection levels; wrap each in parens
    levels = "".join(f"({p})" for p in parts[1:])
    return f"§{section}{levels}"


def extract_prolog_predicates(prolog_text: str) -> set[str]:
    """Return the set of canonical subsection identifiers in a Prolog blob."""
    predicates = set()
    for m in _PREDICATE_RE.finditer(prolog_text):
        canonical = predicate_to_subsection(m.group(0))
        if canonical:
            predicates.add(canonical)
    return predicates


def case_prolog_predicates(case: dict) -> set[str]:
    """All canonical subsection identifiers across a case's Prolog (facts + test)."""
    blob = case.get("prolog_facts", "") + "\n" + case.get("prolog_test", "")
    return extract_prolog_predicates(blob)


# ── CC trace parsing ──────────────────────────────────────────────────────────

# Match an explicit subsection citation. A citation must be marked by one of:
#   - the § symbol (optionally followed by whitespace)
#   - the word "section" or "Section" (whole-word, followed by whitespace)
#   - the abbreviation "sec." or "Sec." (followed by optional whitespace)
# Bare digits (dollar amounts, years) never qualify.
#
# After the marker, capture: <digits>[(<alphanum>)]+...  e.g. 151(a)(1)
#
# Examples that MATCH:
#   "§151(a)", "§ 151(a)", "§151", "Section 151(d)(3)", "section 7703(b)",
#   "Sec. 63", "§ 63(c)(2)(C)"
#
# Examples that DO NOT match (correctly):
#   "$2,000", "in 2015", "100000", "$250,000"
_SUBSECTION_RE = re.compile(
    r"""
    (?:                              # citation marker (required):
        §\s*                         #   § symbol with optional whitespace, or
        |
        \b[Ss]ection\s+              #   the word "Section" / "section", or
        |
        \b[Ss]ec\.\s*                #   abbreviation "Sec." / "sec."
    )
    (?P<section>\d+)                 # section number (digits)
    (?P<subs>                        # optional subsection chain:
        (?:\([a-zA-Z0-9]+\))*        #   any number of (a), (1), (A), etc.
    )
    """,
    re.VERBOSE,
)


def normalize_citation(text: str) -> str | None:
    """
    Normalize a citation string to canonical form §<num>(...)(...)

    Returns None if `text` doesn't begin with a citation marker.

    Examples:
        "§151(a)"           → §151(a)
        "section 151(a)"    → §151(a)
        "Section 151(d)(3)" → §151(d)(3)
        "§151"              → §151
        "Sec. 63"           → §63
        "151(a)"            → None      (bare; needs § or "section")
        "$2,000"            → None
        "in 2015"           → None
    """
    m = _SUBSECTION_RE.match(text.strip())
    if not m:
        return None
    return f"§{m.group('section')}{m.group('subs')}"


# Permissive variant for contexts where every string IS a citation (e.g.,
# generate.artifact.subsections_cited is an explicit list of citations).
# Bare entries like "151(a)" are accepted; dollar amounts and bare years
# are still rejected (require at least one subsection paren OR a marker).
_AUTHORITATIVE_CITATION_RE = re.compile(
    r"""
    ^\s*
    (?:§\s*|[Ss]ection\s+|[Ss]ec\.\s*)?  # optional citation marker
    (?P<section>\d+)                     # section number
    (?P<subs>(?:\([a-zA-Z0-9]+\))+)?     # optional subsection chain
    \s*$
    """,
    re.VERBOSE,
)


def normalize_authoritative_citation(text: str) -> str | None:
    """
    Normalize a citation in a context where the surrounding structure
    guarantees the string IS a citation (e.g. an entry in an explicit
    citations list).

    Requires EITHER a citation marker (§, "section") OR at least one
    subsection paren. Bare digits alone return None — that's a dollar
    amount or year, not a citation.
    """
    m = _AUTHORITATIVE_CITATION_RE.match(text.strip())
    if not m:
        return None
    section = m.group("section")
    subs = m.group("subs") or ""
    raw = text.strip()
    has_marker = bool(re.match(r'^\s*(?:§\s*|[Ss]ection\s+|[Ss]ec\.\s*)', raw))
    if not has_marker and not subs:
        return None
    return f"§{section}{subs}"


def extract_cc_subsection_citations(cc_result: dict) -> set[str]:
    """
    Collect all subsection identifiers cited by CC's run on a case.
    Looks at: generate.artifact.subsections_cited (primary, authoritative),
              investigate.evidence_used (secondary, free-form prose).
    """
    citations = set()

    # From step_records — find the generate step
    for rec in cc_result.get("step_records", []):
        if rec.get("primitive") == "generate":
            artifact = rec.get("output", {}).get("artifact") or {}
            if isinstance(artifact, dict):
                # subsections_cited is an explicit list — every entry IS
                # meant to be a citation. Use the authoritative normalizer
                # which accepts bare "151(a)" alongside marked "§151(a)".
                for cite in artifact.get("subsections_cited", []) or []:
                    if isinstance(cite, str):
                        norm = normalize_authoritative_citation(cite)
                        if norm:
                            citations.add(norm)
        if rec.get("primitive") == "investigate":
            ev = rec.get("output", {}).get("evidence_used", []) or []
            for entry in ev:
                if isinstance(entry, dict):
                    desc = entry.get("description", "")
                    # Free-form prose — use the strict normalizer that
                    # requires an explicit citation marker. Bare digits
                    # (dollar amounts, years) must NOT match.
                    for m in _SUBSECTION_RE.finditer(desc):
                        norm = f"§{m.group('section')}{m.group('subs') or ''}"
                        citations.add(norm)

    # Also try the result_summary in case step_records didn't capture it
    rs = cc_result.get("result_summary") or {}
    for step in rs.get("steps", []):
        if step.get("primitive") == "generate":
            art = step.get("artifact") or {}
            if isinstance(art, dict):
                for cite in art.get("subsections_cited", []) or []:
                    if isinstance(cite, str):
                        norm = normalize_authoritative_citation(cite)
                        if norm:
                            citations.add(norm)

    return citations


def is_descendant_or_equal(citation: str, predicate: str) -> bool:
    """
    Returns True if `citation` is the same as or a descendant of `predicate`.

    Examples:
        §151(a)    matches  §151             (descendant)
        §151       matches  §151             (equal)
        §151(d)(3) matches  §151(d)          (descendant)
        §151(d)(3) matches  §151             (descendant)
        §151(a)    does not match  §151(b)
        §152       does not match  §151
    """
    if citation == predicate:
        return True
    return citation.startswith(predicate + "(")


# ── Per-case faithfulness scoring ─────────────────────────────────────────────

def score_case(cc_result: dict, case: dict) -> dict:
    """
    Compare CC's citations against the Prolog predicates for one case.

    A CC citation is considered grounded if it matches (equal or descendant
    of) at least one Prolog predicate. This is a permissive match — citing
    §151(a) when the Prolog references §151(d)(3) is NOT grounded, but
    citing §151(a) when the Prolog references §151(a) IS grounded.

    Conversely, a Prolog predicate is "covered" if at least one CC citation
    matches it.
    """
    prolog_preds = case_prolog_predicates(case)
    cc_cites = extract_cc_subsection_citations(cc_result)

    # Which CC citations are grounded in some Prolog predicate?
    grounded_cc = set()
    for cite in cc_cites:
        for pred in prolog_preds:
            if is_descendant_or_equal(cite, pred) or is_descendant_or_equal(pred, cite):
                grounded_cc.add(cite)
                break

    # Which Prolog predicates are covered by some CC citation?
    covered_pred = set()
    for pred in prolog_preds:
        for cite in cc_cites:
            if is_descendant_or_equal(cite, pred) or is_descendant_or_equal(pred, cite):
                covered_pred.add(pred)
                break

    over_attr = cc_cites - grounded_cc
    under_attr = prolog_preds - covered_pred

    precision = (len(grounded_cc) / len(cc_cites)) if cc_cites else None
    recall = (len(covered_pred) / len(prolog_preds)) if prolog_preds else None

    return {
        "case_id":          case["case_id"],
        "prolog_predicates": sorted(prolog_preds),
        "cc_citations":     sorted(cc_cites),
        "grounded_cc":      sorted(grounded_cc),
        "covered_pred":     sorted(covered_pred),
        "over_attribution": sorted(over_attr),
        "under_attribution": sorted(under_attr),
        "precision":        precision,
        "recall":           recall,
        "n_cc_citations":   len(cc_cites),
        "n_prolog_pred":    len(prolog_preds),
    }


# ── Aggregate scoring ─────────────────────────────────────────────────────────

def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate(per_case: list[dict]) -> dict:
    # Macro-averaged precision/recall (mean across cases, ignoring None)
    precisions = [c["precision"] for c in per_case if c["precision"] is not None]
    recalls    = [c["recall"]    for c in per_case if c["recall"]    is not None]

    # Micro-averaged: sum across cases
    total_cites = sum(c["n_cc_citations"] for c in per_case)
    total_grounded = sum(len(c["grounded_cc"]) for c in per_case)
    total_preds = sum(c["n_prolog_pred"] for c in per_case)
    total_covered = sum(len(c["covered_pred"]) for c in per_case)

    micro_prec = (total_grounded / total_cites) if total_cites else None
    micro_rec  = (total_covered / total_preds)  if total_preds  else None

    # Distributions
    over_counts = Counter(len(c["over_attribution"]) for c in per_case)
    under_counts = Counter(len(c["under_attribution"]) for c in per_case)

    # All over-attributed citations across cases
    all_over = Counter()
    for c in per_case:
        for o in c["over_attribution"]:
            all_over[o] += 1

    # All under-attributed predicates across cases
    all_under = Counter()
    for c in per_case:
        for u in c["under_attribution"]:
            all_under[u] += 1

    return {
        "n_cases": len(per_case),
        "macro": {
            "precision_mean": sum(precisions) / len(precisions) if precisions else None,
            "recall_mean":    sum(recalls) / len(recalls) if recalls else None,
            "n_precision":    len(precisions),
            "n_recall":       len(recalls),
        },
        "micro": {
            "precision":         micro_prec,
            "precision_ci":      wilson_ci(total_grounded, total_cites) if total_cites else None,
            "recall":            micro_rec,
            "recall_ci":         wilson_ci(total_covered, total_preds) if total_preds else None,
            "total_cites":       total_cites,
            "total_grounded":    total_grounded,
            "total_preds":       total_preds,
            "total_covered":     total_covered,
        },
        "over_attribution_count_dist":  dict(over_counts),
        "under_attribution_count_dist": dict(under_counts),
        "most_common_over_attr":  all_over.most_common(10),
        "most_common_under_attr": all_under.most_common(10),
    }


# ── Reporting ─────────────────────────────────────────────────────────────────

def print_report(per_case: list[dict], agg: dict) -> None:
    print("=" * 70)
    print(f"  PROLOG FAITHFULNESS  ({agg['n_cases']} cases)")
    print("=" * 70)

    mp = agg["micro"]["precision"]
    mr = agg["micro"]["recall"]
    print()
    print("  Micro-averaged (each citation/predicate counted equally):")
    if mp is not None:
        ci = agg["micro"]["precision_ci"]
        print(f"    Precision: {mp:.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]   "
              f"({agg['micro']['total_grounded']}/{agg['micro']['total_cites']} citations grounded)")
    else:
        print("    Precision: N/A (no citations to score)")
    if mr is not None:
        ci = agg["micro"]["recall_ci"]
        print(f"    Recall:    {mr:.1%}  [{ci[0]:.1%}, {ci[1]:.1%}]   "
              f"({agg['micro']['total_covered']}/{agg['micro']['total_preds']} predicates cited)")
    else:
        print("    Recall:    N/A (no Prolog predicates to score)")

    mac_p = agg["macro"]["precision_mean"]
    mac_r = agg["macro"]["recall_mean"]
    print()
    print("  Macro-averaged (mean across cases):")
    if mac_p is not None:
        print(f"    Precision: {mac_p:.1%}  (n={agg['macro']['n_precision']})")
    if mac_r is not None:
        print(f"    Recall:    {mac_r:.1%}  (n={agg['macro']['n_recall']})")

    print()
    print("  Over-attribution distribution (citations not in Prolog):")
    for n_over, count in sorted(agg["over_attribution_count_dist"].items()):
        print(f"    {n_over:>2} over-attr citations: {count} cases")
    if agg["most_common_over_attr"]:
        print()
        print("  Most common over-attributed citations:")
        for cite, n in agg["most_common_over_attr"][:5]:
            print(f"    {cite}: {n} cases")

    print()
    print("  Under-attribution distribution (Prolog predicates CC missed):")
    for n_under, count in sorted(agg["under_attribution_count_dist"].items()):
        print(f"    {n_under:>2} missed predicates: {count} cases")
    if agg["most_common_under_attr"]:
        print()
        print("  Most common missed predicates:")
        for pred, n in agg["most_common_under_attr"][:5]:
            print(f"    {pred}: {n} cases")

    print()
    print("  Per-case detail:")
    for c in per_case:
        prec_str = f"{c['precision']:.1%}" if c["precision"] is not None else " N/A "
        rec_str  = f"{c['recall']:.1%}"    if c["recall"]    is not None else " N/A "
        flags = []
        if c["over_attribution"]:
            flags.append(f"OVER:{','.join(c['over_attribution'])}")
        if c["under_attribution"]:
            flags.append(f"UNDER:{','.join(c['under_attribution'])}")
        flag_str = "  " + "  ".join(flags) if flags else ""
        print(f"    {c['case_id']:<32}  P={prec_str:>6}  R={rec_str:>6}{flag_str}")


# ── Main ──────────────────────────────────────────────────────────────────────

def find_case_file(case_id: str) -> Path | None:
    for sub in [
        CASES_DIR / "binary" / "train",
        CASES_DIR / "binary" / "dev",
        CASES_DIR / "binary" / "test",
        CASES_DIR / "numeric" / "train",
        CASES_DIR / "numeric" / "test",
    ]:
        p = sub / f"{case_id}.json"
        if p.exists():
            return p
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True,
                        help="Path to a CC run output directory")
    parser.add_argument("--out",
                        help="Save the report as JSON to this path")
    args = parser.parse_args()

    run_dir = Path(args.run)
    cc_files = sorted(run_dir.glob("SARA-*.json"))
    if not cc_files:
        print(f"No CC result files in {run_dir}", file=sys.stderr)
        return 1

    per_case = []
    for f in cc_files:
        cc_result = json.loads(f.read_text())
        case_path = find_case_file(cc_result["case_id"])
        if not case_path:
            print(f"Warning: no case file for {cc_result['case_id']}",
                  file=sys.stderr)
            continue
        case = json.loads(case_path.read_text())
        per_case.append(score_case(cc_result, case))

    agg = aggregate(per_case)
    print_report(per_case, agg)

    if args.out:
        report = {"per_case": per_case, "aggregate": agg}
        Path(args.out).write_text(json.dumps(report, indent=2, default=str))
        print(f"\nSaved report → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())