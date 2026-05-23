# CC × SARA — Domain Pack for Statutory Reasoning

A Cognitive Core domain pack for the SARA (StAtutory Reasoning Assessment) Binary task. The architectural claim being tested: decomposed reasoning with typed primitives and input/output contracts produces more grounded determinations than monolithic chain-of-thought, on a benchmark with external validity (US Federal Tax statutes, published evaluations).

## Status

First-pass implementation. Workflow-mode (fixed primitive sequence), sequential execution (no parallelism in this pass), six distinct primitives, eight invocations. Two primitives intentionally dropped from the chain for this domain (Classify, Deliberate) — see "Design decisions" below.

## Layout

```
sara-pack/
├── workflows/sara_binary.yaml          The 8-step fixed sequence
├── domains/sara_us_federal_tax.yaml    Domain knowledge + primitive configs
├── documents/                          Nine SARA-edited IRC sections
│   ├── irc_section_1.txt   (tax imposed)
│   ├── irc_section_2.txt   (filing status definitions)
│   ├── irc_section_63.txt  (taxable income)
│   ├── irc_section_68.txt  (overall limitation)
│   ├── irc_section_151.txt (exemptions)
│   ├── irc_section_152.txt (dependent defined)
│   ├── irc_section_3301.txt (FUTA tax)
│   ├── irc_section_3306.txt (FUTA definitions)
│   └── irc_section_7703.txt (marital status)
├── cases/
│   ├── binary/{train,dev,test}/        SARA NLI cases as CC-format JSON
│   ├── numeric/{train,test}/           SARA QA cases (not used in first pass)
│   └── splits.json                     Manifest of case IDs per split
├── coordinator_config.yaml
├── build_cases.py                      Converts SARA HF JSON → CC case format
├── build_dev_set.py                    Carves a stratified dev set from train
├── run.py                              CC runner
├── run_baseline.py                     Monolithic LLM baseline runner
└── score.py                            Scoring + head-to-head comparison
```

## Case counts

| Task    | Train | Dev | Test |
|---------|-------|-----|------|
| Binary  | 155   | 21  | 100  |
| Numeric | 80    | —   | 20   |

Dev set is carved from train, stratified by IRC section, fixed random seed (7).

## Design decisions

The full reasoning is in the conversation log; this is the summary.

**Workflow mode, not agentic.** SARA Binary cases all have the same epistemic structure: facts + statute + claim → Entailment/Contradiction. A fixed sequence is the right way to test whether decomposition works on this task; agentic mode would add an orchestrator-reasoning confound. If workflow underperforms in a way that suggests sequence problems, fall back to agentic as a learning instrument.

**Six primitives, not eight.** Two primitives are not invoked for SARA Binary:
- **Classify** — the claim type is structurally evident in the question after `retrieve_question_structure`. No categorical ambiguity to resolve.
- **Deliberate** — SARA produces a truth-value judgment, not a recommended action. Deliberate's "warranted action" framing doesn't fit.

This is a finding, not a limitation. The architecture's eight primitives compose selectively per domain.

**Three Retrieve calls upfront.** `retrieve_question_structure`, `retrieve_statute`, `retrieve_case_facts` each surface one orientation surface. Failures are cleanly attributable to a specific surface.

**Sequential execution.** First pass is worst-case latency. Parallel composition of the three independent Retrieves is a follow-up optimization.

**Verify is reframed as an anti-hallucination check.** Rules R1–R5 in the verify config check that Investigate's reasoning is sound and grounded (rule-output type matches claim type, no hallucinated subsections, no unsupported facts). This is what makes Verify earn its place: it catches the silent-error mode that monolithic chains miss.

**HOLD tier on hallucinated subsections.** If Generate cites a subsection that was not in Retrieve's output, Govern routes to HOLD (compliance review), not GATE (analyst review). This is structural — it converts a hallucination into a non-release condition.

## Train/dev/test discipline

- **Train (155)**: free to look at cases, examine traces, iterate on YAML.
- **Dev (21)**: frozen regression check. Run after each YAML change; never look at trace details case-by-case to tune.
- **Test (100)**: looked at exactly once, after iteration converges. The reported number.

Violating this discipline invalidates the result.

## Baseline protocol

The monolithic baseline uses the same model as CC, same input (statute corpus + case narrative + question), single chain-of-thought call. Prompt structure is fixed in `run_baseline.py` and not tuned per case.

## Usage

```bash
# Build cases (one-time, after uploading SARA HF JSON files)
python build_cases.py
python build_dev_set.py

# Run CC on a single case
python run.py SARA-S151-A-NEG

# Run CC on a section subset
python run.py --split dev --section 151 --all

# Full test set (NOT to be run during iteration — only after convergence)
python run.py --split test --all

# Baseline
python run_baseline.py --split dev --all

# Score and compare
python score.py --run output/cc_20260510_120000 \
                --compare output/baseline_20260510_121500
```

## Reference baselines for comparison

From the published literature on SARA:

- Random/chance: 50.0% (binary balanced)
- GPT-4 chain-of-thought: ~67% on 276 binary cases (Blair-Stanek, Holzenberger, Van Durme, Tax Notes 2023)
- DeonticBench frontier-LLM hard subset: 44.4% on SARA Numeric (not directly comparable to binary)
- Symbolic solver with hand-translated Prolog (Holzenberger 2020): ~100% — but requires costly human translation

CC should be measured against the GPT-4 reference point for the strongest comparison.

## Source

Cases: `jhu-clsp/SARA` on HuggingFace (SARA v1, NLI subset).
Statutes: `SgfdDttt/sara` on GitHub, `statutes/source/`.
