#!/bin/bash
# INTEGRITY GATE -- run BEFORE reading any accuracy number.
#   ./check_v135.sh smoke [K ...]     ./check_v135.sh grid [K ...]     (default: all K)
# Every job log: rc=0, loss=ce, n_classes=K, class list, n_obs=W, params
# VERIFIED OK, determinism on, no continuation. Every M5 log additionally:
# the freeze banner and the exit self-check "front-end after run: UNCHANGED OK".
# Then result counts per cell. Smoke also runs the M3 REPRODUCTION test against
# v12 (exact match) and checks M5 resolved the same min-epochs as M3.
cd "$(dirname "$0")"
source ./config135.sh
MODE="${1:?usage: ./check_v135.sh smoke|grid [K ...]}"; shift
KLIST="${*:-$KS}"
FAIL=0; bad() { echo "  FAIL: $1"; FAIL=1; }
minep() { grep -oiE 'min_?epochs?[A-Za-z_]*[[:space:]]*[=:][[:space:]]*[0-9]+' "$1" | grep -oE '[0-9]+$' | sort -u | tr '\n' ' '; }
PREFIX=$( [ "$MODE" = "smoke" ] && echo v135s || echo v135 )
ALL_LOGS=""
for K in $KLIST; do
  LOGS=$(ls ${PREFIX}_k${K}[A-Za-z]*.o* 2>/dev/null)
  [ -z "$LOGS" ] && { bad "no $MODE logs for K=$K yet (check qstat)"; continue; }
  ALL_LOGS="$ALL_LOGS $LOGS"
  NLOG=0
  for f in $LOGS; do
    NLOG=$((NLOG+1))
    L=$(grep -m1 '^\[V135\] K=' "$f") || { bad "$f  no [V135] header"; continue; }
    TAG=$(sed -n 's/.* TAG=\([A-Za-z]*\) .*/\1/p' <<<"$L"); W=$(sed -n 's/.* W=\([0-9]*\) .*/\1/p' <<<"$L")
    M=$(sed -n 's/.* MODEL=\(M[0-9]\) .*/\1/p' <<<"$L")
    WANT=$(subset_classes "$K" "$TAG" | sed 's/,/, /g')
    grep -q 'Job end.*rc=0'                     "$f" || bad "$f  no 'Job end rc=0'"
    grep -q 'PROTOCOL RESOLVED.*loss=ce'        "$f" || bad "$f  loss is not ce"
    grep -q "PROTOCOL RESOLVED.*n_classes=$K "  "$f" || bad "$f  n_classes != $K"
    grep -q "PROTOCOL RESOLVED.*classes=\[$WANT\]" "$f" || bad "$f  class list != [$WANT]"
    grep -q "\[MULTICLASS\].*n_obs=$W "         "$f" || bad "$f  n_obs != $W"
    grep -q 'VERIFIED.*OK'                      "$f" || bad "$f  parameter check not OK"
    grep -qi 'DETERMINISM.*ENABLED'             "$f" || bad "$f  determinism not enabled"
    grep -q 'CONT=[1-9]'                        "$f" && bad "$f  continuation (not bit-reproducible)"
    if [ "$M" = "M5" ]; then
      grep -q '\[V135\] M5 = M3 with front-end frozen'        "$f" || bad "$f  M5 freeze banner missing"
      grep -q '\[V135\] M5 front-end after run: UNCHANGED OK' "$f" || bad "$f  M5 exit self-check missing"
      grep -q '\[V135\] warmup: model=M5 min_epochs=30' "$f" || bad "$f  M5 warmup is not 30 epochs (must match M3)"
      grep -q 'CHANGED -- FAIL'                                "$f" && bad "$f  M5 FRONT-END CHANGED DURING TRAINING"
    elif ! { [ "$MODE" = "smoke" ] && [ "$M" = "$REPRO_MODEL" ]; }; then
      bad "$f  unexpected model $M"
    fi
  done
  echo "Checked $NLOG log(s) for K=$K ($MODE)."
done

count() { ls "../$1"/result_$2_*.json 2>/dev/null | wc -l; }
need()  { local n; n=$(count "$1" "$3"); echo "  $1: $n $3 result(s) (expect $2)"; [ "$n" -ge "$2" ] || bad "$1 has $n, expected $2"; }
EXP=$( [ "$MODE" = "smoke" ] && echo 1 || echo $(wc -w <<<"$SEEDS") )
for K in $KLIST; do for TAG in $(subset_tags "$K"); do for W in $(widths_for "$K"); do
  need "$(results_dir $K $TAG $W)" "$EXP" M5
done; done; done
[ -n "$ALL_LOGS" ] && { echo "Parameter counts seen (one line per K x width):"; grep -h 'VERIFIED' $ALL_LOGS | sed 's/^.*\] //' | sort | uniq -c; }

DO_REPRO=0
[ "$MODE" = "smoke" ] && grep -qw "$REPRO_K" <<<"$KLIST" && DO_REPRO=1
if [ $DO_REPRO = 1 ]; then
  need "$(repro_dir)" 1 M3
  RL=$(ls v135s_k5Ar.o* 2>/dev/null | tail -1); [ -n "$RL" ] && { grep -q '\[V135\] warmup: model=M3 min_epochs=30' "$RL" || bad "$RL  M3 warmup is not 30 epochs"; }
  RLOG=$(ls v135s_k5Ar.o* 2>/dev/null | tail -1); SLOG=$(ls v135s_k5Aw3.o* 2>/dev/null | tail -1)
  if [ -n "$RLOG" ] && [ -n "$SLOG" ]; then
    echo "Protocol lines (M3 repro vs M5, K=5 A w3) -- must agree on everything except the model:"
    grep -h 'PROTOCOL RESOLVED' "$RLOG" | head -1 | sed 's/^/  M3: /'
    grep -h 'PROTOCOL RESOLVED' "$SLOG" | head -1 | sed 's/^/  M5: /'
    A=$(minep "$RLOG"); B=$(minep "$SLOG")
    if [ -n "$A$B" ]; then
      echo "  min-epochs  M3: [$A]  M5: [$B]"
      [ "$A" = "$B" ] || bad "M5 min-epochs differ from M3 (pre-registered: M5 trains under M3's protocol)"
    else
      echo "  (no min-epochs value printed in the logs: paste the two lines above the first time)"
    fi
  fi
fi

if [ $FAIL -ne 0 ]; then
  echo "GATE FAILED -- do not read accuracies. Paste this output."
  for K in $KLIST; do rm -f ".smoke_ok_k$K"; done; [ $DO_REPRO = 1 ] && rm -f .repro_ok
  exit 1
fi

if [ $DO_REPRO = 1 ]; then
  echo "Reproduction test (patched repo, M3 K=5 A w3 seed 42 must equal v12 EXACTLY):"
  python3 - "$(ref_result)" "../$(repro_dir)/result_M3_mnist_seed42_alpha0.0.json" <<'PY' || { echo "GATE FAILED: reproduction mismatch. Do NOT submit the grid. Paste this output."; rm -f .repro_ok .smoke_ok_k5; exit 1; }
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
  touch .repro_ok
fi

echo "ALL CHECKS PASSED"
if [ "$MODE" = "smoke" ]; then
  for K in $KLIST; do
    touch ".smoke_ok_k$K"
    for TAG in $(subset_tags "$K"); do for W in $(widths_for "$K"); do
      python3 -c "import json;r=json.load(open('../$(results_dir $K $TAG $W)/result_M5_mnist_seed42_alpha0.0.json'));print('  smoke M5 K=$K $TAG w=$W: test_acc',r['test_acc'],'best',r.get('best_epoch'),'stop',r.get('stop_epoch'))"
    done; done
  done
  echo "Sanity only (not a result): accuracy must be well above chance (1/K). Then ./submit_v135_grid.sh $*"
else
  echo "When every K passes: cd .. && python3 scripts/collect_v135.py && python3 scripts/plot_v135.py"
fi
