#!/bin/bash
# INTEGRITY GATE -- run BEFORE reading any accuracy number.
#   ./check_v12.sh 3 smoke     (after the smoke jobs)
#   ./check_v12.sh 3 grid      (after the full grid)
# Checks every job log for this K: rc=0, loss=ce, right n_classes, params
# VERIFIED OK, determinism enabled, and the right class list; then counts
# result JSONs. Accuracy is printed ONLY for the smoke, and ONLY if the
# gate passed.
cd "$(dirname "$0")"
source ./subsets.sh
K="${1:?usage: ./check_v12.sh K smoke|grid}"
MODE="${2:?usage: ./check_v12.sh K smoke|grid}"
if [ "$MODE" = "smoke" ]; then LOGS=$(ls v12s_k${K}?.o* 2>/dev/null); EXPECT=1
else LOGS=$(ls v12_k${K}?.o* 2>/dev/null); EXPECT=40; fi
[ -z "$LOGS" ] && { echo "No logs found for K=$K $MODE yet (still queued/running? check qstat)"; exit 1; }

FAIL=0
bad() { echo "  FAIL: $1"; FAIL=1; }
NLOG=0
for f in $LOGS; do
  NLOG=$((NLOG+1))
  grep -q 'Job end.*rc=0'             "$f" || bad "$f  no 'Job end rc=0'"
  grep -q 'PROTOCOL RESOLVED.*loss=ce' "$f" || bad "$f  loss is not ce"
  grep -q "PROTOCOL RESOLVED.*n_classes=$K " "$f" || bad "$f  n_classes != $K"
  grep -q 'VERIFIED.*OK'              "$f" || bad "$f  parameter check not OK"
  grep -qi 'DETERMINISM.*ENABLED'     "$f" || bad "$f  determinism not enabled"
  grep -q 'CONT=[1-9]'                "$f" && bad "$f  was a continuation (not bit-reproducible)"
  TAG=$(grep -o 'TAG=[AB]' "$f" | head -1 | cut -d= -f2)
  WANT=$(subset_classes "$K" "$TAG" | sed 's/,/, /g')
  grep -q "PROTOCOL RESOLVED.*classes=\[$WANT\]" "$f" || bad "$f  class list != [$WANT]"
done
echo "Checked $NLOG log(s) for K=$K ($MODE)."

echo "Parameter counts seen (expect ONE line per model, same across subsets/seeds):"
grep -h 'VERIFIED' $LOGS | sed 's/^.*\] //' | sort | uniq -c

for TAG in $SUBSET_TAGS; do
  D="../$(results_dir $K $TAG)"
  N=$(ls "$D"/result_*.json 2>/dev/null | wc -l)
  echo "  subset $TAG: $N result JSON(s) in $D (expect $EXPECT)"
  [ "$N" -ge "$EXPECT" ] || bad "subset $TAG has $N results, expected $EXPECT"
done

if [ $FAIL -ne 0 ]; then
  echo "GATE FAILED -- do not read accuracies. Paste this output."; exit 1
fi
echo "ALL CHECKS PASSED"
if [ "$MODE" = "smoke" ]; then
  for TAG in $SUBSET_TAGS; do
    D="../$(results_dir $K $TAG)"
    python3 -c "import json;r=json.load(open('$D/result_M2_mnist_seed42_alpha0.0.json'));print('  smoke K=$K $TAG: test_acc',r['test_acc'],'best_epoch',r.get('best_epoch'),'stop_epoch',r.get('stop_epoch'))"
  done
  echo "Sanity: accuracy must be well above chance (1/$K). Then: ./submit_v12_grid.sh $K"
else
  echo "Next: cd .. && python scripts/collect_v12.py && python scripts/plot_v12.py"
fi
