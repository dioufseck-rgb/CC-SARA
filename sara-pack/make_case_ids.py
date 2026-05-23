#!/usr/bin/env python3
"""Generate sara_test_case_ids.txt — one case_id per line, in test-set order.

Usage:
  python make_case_ids.py sara_test.json > sara_test_case_ids.txt
"""
import json, sys

if len(sys.argv) != 2:
    print("usage: make_case_ids.py <sara_test.json>", file=sys.stderr)
    sys.exit(1)

data = json.load(open(sys.argv[1]))
# sara_test.json is a list of {case_id, ...}
for c in data:
    print(c["case_id"])