#!/bin/bash
# INTEGRITY GATE -- run BEFORE reading any accuracy number.
#   ./check_v13.sh 5 smoke | ./check_v13.sh 5 grid | (same for 10)
# Every job log: rc=0, loss=ce, n_classes=K, n_obs=W, class list, params
# VERIFIED OK, determinism on, no continuation. Then result counts per cell.
# Smoke also runs the REPRODUCTION test against v12/v10 (exact match) and
# prints accuracy only for the smoke and only if everything passed.
cd "$(dirname "$0")"
source ./config13.sh
K="${1:?usage: ./check_v13.sh K smoke|grid}"; MODE="${2:?usage: ./check_v13.sh K smoke|grid}"
if [ "$MODE" = "smoke" ]; then LOGS=$(ls v13s_k${K}[A-Za-z]*.o* 2>/dev/null)
else LOGS=$(ls v13_k${K}[A-Za-z]*.o* 2>/dev/null); fi
[ -z "$LOGS" ] && { echo "No logs for K=$K $MODE yet (check qstat)"; exit 1; }

FAIL=0; bad() { echo "  FAIL: $1"; FAIL=1; }
NLOG=0
for f in $LOGS; do
  NLOG=$((NLOG+1))
  L=$(grep -m1 '^\[V13\] K=' "$f") || { bad "$f  no [V13] header"; continue; }
  TAG=$(sed -n 's/.* TAG=\([A-Za-z]*\) .*/\1/p' <<<"$L"); W=$(sed -n 's/.* W=\([0-9]*\) .*/\1/p' <<<"$L")
  WANT=$(subset_classes "$K" "$TAG" | sed 's/,/, /g')
  grep -q 'Job end.*rc=0'                     "$f" || bad "$f  no 'Job end rc=0'"
  grep -q 'PROTOCOL RESOLVED.*loss=ce'        "$f" || bad "$f  loss is not ce"
  grep -q "PROTOCOL RESOLVED.*n_classes=$K "  "$f" || bad "$f  n_classes != $K"
  grep -q "PROTOCOL RESOLVED.*classes=\[$WANT\]" "$f" || bad "$f  class list != [$WANT]"
  grep -q "\[MULTICLASS\].*n_obs=$W "         "$f" || bad "$f  n_obs != $W"
  grep -q 'VERIFIED.*OK'                      "$f" || bad "$f  parameter check not OK"
  grep -qi 'DETERMINISM.*ENABLED'             "$f" || bad "$f  determinism not enabled"
  grep -q 'CONT=[1-9]'                        "$f" && bad "$f  continuation (not bit-reproducible)"
done
echo "Checked $NLOG log(s) for K=$K ($MODE)."
echo "Parameter counts seen (expect one line per model per width):"
grep -h 'VERIFIED' $LOGS | sed 's/^.*\] //' | sort | uniq -c

count() { ls "../$1"/result_*.json 2>/dev/null | wc -l; }
need()  { local n; n=$(count "$1"); echo "  $1: $n result(s) (expect $2)"; [ "$n" -ge "$2" ] || bad "$1 has $n, expected $2"; }
if [ "$MODE" = "smoke" ]; then
  RTAG=$( [ "$K" = "10" ] && echo all || echo A )
  need "$(repro_dir $K $RTAG)" 1
  for TAG in $(subset_tags "$K"); do for W in $WIDTHS_NEW; do need "$(results_dir $K $TAG $W)" 1; done; done
else
  for TAG in $(subset_tags "$K"); do for W in $WIDTHS_NEW; do need "$(results_dir $K $TAG $W)" 20; done; done
  [ "$K" = "10" ] && need "$(results_dir 10 all 3)" $(( $(wc -w <<<"$TOPUP_MODELS") * $(wc -w <<<"$TOPUP_SEEDS") ))
fi
if [ $FAIL -ne 0 ]; then echo "GATE FAILED -- do not read accuracies. Paste this output."; rm -f ".smoke_ok_k$K"; exit 1; fi

if [ "$MODE" = "smoke" ]; then
  echo "Reproduction test (width 3 must match the old result EXACTLY):"
  python3 - "$(ref_result $K $RTAG)" "../$(repro_dir $K $RTAG)/result_M2_mnist_seed42_alpha0.0.json" <<'PY' || { echo "GATE FAILED: reproduction mismatch. Do NOT submit the grid. Paste this output."; rm -f ".smoke_ok_k$K"; exit 1; }
import json, sys
old, new = (json.load(open(p)) for p in sys.argv[1:3])
ok = True
for k in ("test_acc", "best_epoch", "stop_epoch"):
    same = old.get(k) == new.get(k)
    ok &= same
    print(f"  {k:<11} old={old.get(k)}  new={new.get(k)}  {'MATCH' if same else 'DIFFERENT'}")
pk = next((k for k in ("predictions", "test_predictions", "preds") if k in old and k in new), None)
if pk:
    same = old[pk] == new[pk]; ok &= same
    print(f"  {pk:<11} {len(new[pk])} per-image predictions  {'MATCH' if same else 'DIFFERENT'}")
else:
    print("  (no per-image predictions key in both files; matched on scalars only)")
sys.exit(0 if ok else 1)
PY
  echo "ALL CHECKS PASSED"
  touch ".smoke_ok_k$K"
  for TAG in $(subset_tags "$K"); do for W in $WIDTHS_NEW; do
    python3 -c "import json;r=json.load(open('../$(results_dir $K $TAG $W)/result_M2_mnist_seed42_alpha0.0.json'));print('  smoke K=$K $TAG w=$W: test_acc',r['test_acc'],'best',r.get('best_epoch'),'stop',r.get('stop_epoch'))"
  done; done
  echo "Sanity only (not a result): accuracy must be well above chance (1/$K). Then ./submit_v13_grid.sh $K"
else
  echo "ALL CHECKS PASSED"
  echo "When BOTH K=5 and K=10 grids pass: cd .. && python3 scripts/collect_v13.py && python3 scripts/plot_v13.py"
fi
