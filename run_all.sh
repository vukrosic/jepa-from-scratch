#!/bin/sh
# Runs every step and the tests. The first run trains every model and saves it in files/.
PY="${PYTHON:-python3}"
cd "$(dirname "$0")"
for f in step1_pictures step2_pixels step3_meaning step4_slow_copy step5_spread step6_decide tests; do
  echo "=== $f"
  "$PY" "$f.py" || exit 1
done
