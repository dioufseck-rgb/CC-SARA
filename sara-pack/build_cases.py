"""
Convert the SARA HuggingFace dataset (sara_train.json, sara_test.json) into
Cognitive Core case JSON files, one per case.

We separate binary cases from numeric:
  - sara-pack/cases/binary/train/    176 cases
  - sara-pack/cases/binary/test/     100 cases
  - sara-pack/cases/numeric/train/    80 cases  (not used in first pass)
  - sara-pack/cases/numeric/test/     20 cases  (not used in first pass)

Each case JSON has:
  case_id          — derived from SARA id
  source_split     — "train" | "test"
  task_type        — "binary" | "numeric"
  case_narrative   — SARA's "text" field (the facts)
  question_text    — SARA's "question" field (the claim)
  ground_truth     — SARA's "answer" field
  statute_corpus   — concatenated nine IRC sections (loaded once, embedded)
  prolog_facts     — SARA's "facts" field (the symbolic ground truth, for analysis only)
  prolog_test      — SARA's "test" field (the symbolic ground truth, for analysis only)
"""

from __future__ import annotations
import json, re
from pathlib import Path

PACK_DIR = Path("/home/claude/sara-pack")
DOCS_DIR = PACK_DIR / "documents"
CASES_DIR = PACK_DIR / "cases"

# Build the statute corpus once. We pass it embedded in each case_input so
# retrieve_statute has direct access via the tool registry.
def load_statute_corpus() -> str:
    sections = []
    for f in sorted(DOCS_DIR.glob("irc_section_*.txt"),
                    key=lambda p: int(re.search(r"_(\d+)\.txt", p.name).group(1))):
        sections.append(f.read_text())
    return "\n\n".join(sections)

def is_binary(answer: str) -> bool:
    return answer in ("Entailment", "Contradiction")

def case_id_from_sara(sara_id: str, split: str) -> str:
    # s151_a_neg → SARA-S151-A-NEG (train).  Add split suffix for clarity.
    parts = sara_id.upper().replace("_", "-")
    return f"SARA-{parts}"

def convert_case(raw: dict, split: str, statute_corpus: str) -> dict:
    answer = raw["answer"]
    return {
        "case_id":        case_id_from_sara(raw["id"], split),
        "source_split":   split,
        "source_id":      raw["id"],
        "task_type":      "binary" if is_binary(answer) else "numeric",
        # The two pieces of case data the workflow reads via retrieve steps
        "case_narrative": raw["text"],
        "question_text":  raw["question"],
        "statute_corpus": statute_corpus,
        # Ground truth — NOT visible to the workflow; used only by scorer
        "ground_truth":   answer,
        # Symbolic representations — held aside for step-faithfulness analysis
        "prolog_facts":   raw.get("facts", ""),
        "prolog_test":    raw.get("test", ""),
    }

def main():
    statute_corpus = load_statute_corpus()
    print(f"Statute corpus: {len(statute_corpus)} chars from "
          f"{len(list(DOCS_DIR.glob('irc_section_*.txt')))} sections")

    with open("/mnt/user-data/uploads/sara_train.json") as f:
        train = json.load(f)
    with open("/mnt/user-data/uploads/sara_test.json") as f:
        test = json.load(f)

    counts = {"binary": {"train": 0, "test": 0},
              "numeric": {"train": 0, "test": 0}}

    for raw_cases, split in [(train, "train"), (test, "test")]:
        for raw in raw_cases:
            case = convert_case(raw, split, statute_corpus)
            task = case["task_type"]
            out_dir = CASES_DIR / task / split
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{case['case_id']}.json"
            with open(out_path, "w") as f:
                json.dump(case, f, indent=2)
            counts[task][split] += 1

    print(f"\nWritten case files:")
    for task, splits in counts.items():
        for split, n in splits.items():
            print(f"  {task}/{split}: {n} files")

    # Also dump a split manifest — which case IDs are in each split.
    manifest = {
        "binary": {
            "train": sorted([
                p.stem for p in (CASES_DIR / "binary" / "train").glob("*.json")
            ]),
            "test": sorted([
                p.stem for p in (CASES_DIR / "binary" / "test").glob("*.json")
            ]),
        },
        "numeric": {
            "train": sorted([
                p.stem for p in (CASES_DIR / "numeric" / "train").glob("*.json")
            ]),
            "test": sorted([
                p.stem for p in (CASES_DIR / "numeric" / "test").glob("*.json")
            ]),
        },
    }
    with open(CASES_DIR / "splits.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nWrote split manifest: {CASES_DIR / 'splits.json'}")

if __name__ == "__main__":
    main()
