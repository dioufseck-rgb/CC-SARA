#!/usr/bin/env bash
# resume_v7_run.sh — resume the v7 SARA Binary run from wherever it left off.
#
# What this does:
#   1. Find the SARA test-split file (sara/splits/test — one snake_case id per line).
#   2. Map each id to its expected output filename: s152_d_2_C_pos -> SARA-S152-D-2-C-POS.json
#   3. Scan the output dir; treat a case as "done" if the JSON exists, parses, and has
#      a non-empty `determination` field OR a terminal `status` (completed/suspended).
#   4. Re-launch the runner on remaining cases only, one at a time (resilient to crashes).
#
# Environment overrides (all optional):
#   OUTPUT_DIR    output directory of per-case JSONs       (default: output/cc_v7_test_n100)
#   SPLITS_FILE   path to sara/splits/test                 (default: autodetect)
#   RUNNER        command to invoke your runner            (default: "python run.py")
#   WORKFLOW      workflow id                               (default: sara_binary)
#   DOMAIN        domain id                                 (default: sara_us_federal_tax)
#   CASE_FLAG     CLI flag name for case id                 (default: --cases)
#   OUT_FLAG      CLI flag name for output dir              (default: --output-dir)
#
# Usage:
#   bash resume_v7_run.sh           # interactive (asks before launching)
#   bash resume_v7_run.sh -y        # non-interactive
#   bash resume_v7_run.sh --dry     # show what would run, don't launch
#   bash resume_v7_run.sh --list    # just list completed/remaining and exit

set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-output/cc_v7_test_n100}"
SPLITS_FILE="${SPLITS_FILE:-}"
RUNNER="${RUNNER:-python run.py}"
WORKFLOW="${WORKFLOW:-sara_binary}"
DOMAIN="${DOMAIN:-sara_us_federal_tax}"
CASE_FLAG="${CASE_FLAG:---cases}"
OUT_FLAG="${OUT_FLAG:---output-dir}"
INCLUDE_NUMERIC="${INCLUDE_NUMERIC:-0}"   # set to 1 to include SARA Numeric (tax_case_NN) cases

# ---- arg parsing ----
DRY=0; LIST_ONLY=0; YES=0
for arg in "$@"; do
  case "$arg" in
    -y|--yes)  YES=1 ;;
    --dry)     DRY=1 ;;
    --list)    LIST_ONLY=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    *)         echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# ---- locate splits file ----
if [[ -z "$SPLITS_FILE" ]]; then
  d="$PWD"
  while [[ "$d" != "/" ]]; do
    for cand in "$d/sara/splits/test" "$d/splits/test" "$d/sara_test_case_ids.txt"; do
      if [[ -f "$cand" ]]; then
        SPLITS_FILE="$cand"
        break 2
      fi
    done
    d="$(dirname "$d")"
  done
fi

if [[ -z "$SPLITS_FILE" || ! -f "$SPLITS_FILE" ]]; then
  cat <<EOF >&2
ERROR: could not locate the SARA test split file.

Tried walking up from $(pwd) looking for:
  sara/splits/test
  splits/test
  sara_test_case_ids.txt

Fix one of:
  (a) cd into a directory that contains sara/splits/test
  (b) set SPLITS_FILE=/path/to/splits/test
  (c) create sara_test_case_ids.txt — one snake_case id per line (e.g. s1_b_neg)
EOF
  exit 1
fi

echo "Using splits file: $SPLITS_FILE"

# ---- check output dir ----
if [[ ! -d "$OUTPUT_DIR" ]]; then
  echo "ERROR: output dir does not exist: $OUTPUT_DIR" >&2
  echo "       create it with: mkdir -p $OUTPUT_DIR" >&2
  exit 1
fi
echo "Output dir:        $OUTPUT_DIR"

# ---- build case list ----
mapfile -t RAW_IDS_ALL < <(grep -v '^[[:space:]]*$' "$SPLITS_FILE" | grep -v '^[[:space:]]*#')

if [[ ${#RAW_IDS_ALL[@]} -eq 0 ]]; then
  echo "ERROR: splits file is empty: $SPLITS_FILE" >&2
  exit 1
fi

# Filter to SARA Binary (POS/NEG) cases unless INCLUDE_NUMERIC=1
RAW_IDS=()
NUMERIC_DROPPED=0
for raw in "${RAW_IDS_ALL[@]}"; do
  if [[ "$raw" == *_pos || "$raw" == *_neg ]]; then
    RAW_IDS+=("$raw")
  elif [[ "$INCLUDE_NUMERIC" == "1" ]]; then
    RAW_IDS+=("$raw")
  else
    NUMERIC_DROPPED=$((NUMERIC_DROPPED+1))
  fi
done

if [[ $NUMERIC_DROPPED -gt 0 ]]; then
  echo "Filtered out:      $NUMERIC_DROPPED SARA Numeric case(s) (set INCLUDE_NUMERIC=1 to include)"
fi

TOTAL=${#RAW_IDS[@]}
echo "Total cases:       $TOTAL"

SNAKE=()
OUTID=()
for raw in "${RAW_IDS[@]}"; do
  SNAKE+=("$raw")
  cid="SARA-$(echo "$raw" | tr '[:lower:]' '[:upper:]' | tr '_' '-')"
  OUTID+=("$cid")
done

# ---- scan completed ----
DONE_SNAKE=()
TODO_SNAKE=()
TODO_OUTID=()
for i in "${!SNAKE[@]}"; do
  raw="${SNAKE[$i]}"
  cid="${OUTID[$i]}"
  out="$OUTPUT_DIR/$cid.json"
  if [[ -s "$out" ]] && python3 -c "
import json, sys
try:
    d = json.load(open('$out'))
    ok = bool(d.get('determination')) or d.get('status') in ('completed','suspended')
    sys.exit(0 if ok else 1)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
    DONE_SNAKE+=("$raw")
  else
    TODO_SNAKE+=("$raw")
    TODO_OUTID+=("$cid")
  fi
done

echo "Completed:         ${#DONE_SNAKE[@]}"
echo "Remaining:         ${#TODO_SNAKE[@]}"

if [[ $LIST_ONLY -eq 1 ]]; then
  if [[ ${#TODO_SNAKE[@]} -gt 0 ]]; then
    echo
    echo "Remaining cases:"
    for i in "${!TODO_SNAKE[@]}"; do
      printf '  %3d. %-25s -> %s\n' "$((i+1))" "${TODO_SNAKE[$i]}" "${TODO_OUTID[$i]}"
    done
  fi
  exit 0
fi

if [[ ${#TODO_SNAKE[@]} -eq 0 ]]; then
  echo "All cases already complete. Nothing to do."
  exit 0
fi

echo
echo "Next 5 to run:"
for i in "${!TODO_SNAKE[@]}"; do
  [[ $i -lt 5 ]] && printf '  %s\n' "${TODO_SNAKE[$i]}"
done

echo
echo "Runner invocation:"
echo "  $RUNNER --workflow $WORKFLOW --domain $DOMAIN $CASE_FLAG <id> $OUT_FLAG $OUTPUT_DIR"

if [[ $DRY -eq 1 ]]; then
  echo
  echo "(--dry: nothing launched)"
  exit 0
fi

if [[ $YES -eq 0 ]]; then
  echo
  read -r -p "Proceed? [y/N] " ans
  [[ "$ans" == "y" || "$ans" == "Y" ]] || { echo "Aborted."; exit 0; }
fi

# ---- run remaining cases ----
n=0
fail=0
for raw in "${TODO_SNAKE[@]}"; do
  n=$((n+1))
  echo
  echo "──────────────────────────────────────────────────"
  echo "[$n/${#TODO_SNAKE[@]}] $raw"
  echo "──────────────────────────────────────────────────"
  if ! $RUNNER --workflow "$WORKFLOW" --domain "$DOMAIN" \
               $CASE_FLAG "$raw" \
               $OUT_FLAG "$OUTPUT_DIR"; then
    fail=$((fail+1))
    echo "WARN: runner failed on $raw (continuing)"
  fi
done

echo
echo "Finished. ran=$n  failed=$fail"
echo "Score with:"
echo "  python3 score_v7.py $OUTPUT_DIR"