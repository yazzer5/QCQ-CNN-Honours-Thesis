#!/usr/bin/env python3
"""Apply the pre-fixed decision rule to the two loss-smoke runs.

The rule was written into run_loss_smoke.pbs BEFORE submission and is
implemented here verbatim. It looks ONLY at convergence behaviour -- how
long each arm sits near chance, and whether it is still climbing at the
cap. It deliberately does NOT compare M2 - M1, because there is no M1 run
here and because choosing a loss on the size of the primary contrast would
be choosing the answer.

    python scripts/compare_loss_smoke.py
"""
import json, sys
from pathlib import Path

CHANCE = 0.10          # 10 classes
NEAR_CHANCE_PP = 0.02  # "within 2 pp of chance"
PLATEAU_EPOCH = 20     # "past epoch 20"


def load(loss):
    d = Path(f"results_loss_smoke_{loss}")
    files = sorted(d.glob("result_M2_mnist_seed42_alpha*.json"))
    if not files:
        raise SystemExit(f"No result in {d}/ -- has the {loss} job finished?")
    return json.loads(files[0].read_text())


def describe(loss, r):
    h = r["history"]
    stamped = r.get("loss")
    if stamped != loss:
        raise SystemExit(
            f"[INTEGRITY] {loss} result JSON is stamped loss={stamped!r}. "
            f"The run did not use the loss its directory claims. Stop.")

    near = [e["epoch"] for e in h if abs(e["val_acc"] - CHANCE) <= NEAR_CHANCE_PP]
    last_near = max(near) if near else 0
    escape = next((e["epoch"] for e in h
                   if e["val_acc"] > CHANCE + NEAR_CHANCE_PP), None)
    tail = h[-10:]
    climbing = len(tail) >= 2 and tail[-1]["val_acc"] - tail[0]["val_acc"] > 0.01

    print(f"\n--- loss = {loss} ---")
    print(f"  epochs run              : {len(h)}  (stopped at {r.get('stop_epoch')})")
    print(f"  best epoch              : {r.get('best_epoch')}")
    print(f"  first epoch above chance: {escape if escape else 'NEVER'}")
    print(f"  last epoch near chance  : {last_near}")
    print(f"  best val acc            : {r['best_val_acc']:.4f}")
    print(f"  test acc                : {r['test_acc']:.4f}")
    print(f"  still climbing at cap   : {'YES' if climbing else 'no'}"
          f"   (val acc moved {tail[-1]['val_acc']-tail[0]['val_acc']:+.4f} "
          f"over the last {len(tail)} epochs)")
    plateaued = last_near > PLATEAU_EPOCH
    print(f"  PLATEAUED past epoch {PLATEAU_EPOCH}  : {'YES' if plateaued else 'no'}")
    return plateaued, climbing


def main():
    bce = load("bce")
    ce = load("ce")

    print("=" * 68)
    print("LOSS-FUNCTION SMOKE — decision rule fixed before submission")
    print("=" * 68)
    print("This compares CONVERGENCE ONLY. It is not a result.")

    p_bce, c_bce = describe("bce", bce)
    p_ce, c_ce = describe("ce", ce)

    print("\n" + "=" * 68)
    if not p_bce and not p_ce:
        print("VERDICT: neither arm plateaus. KEEP BCE.")
        print("  The concern is retired with evidence from the real task.")
        print("  Run the v10 grid exactly as pre-registered, and record this")
        print("  smoke in the pre-registration as the reason the loss was")
        print("  kept rather than assumed.")
    elif p_bce and not p_ce:
        print("VERDICT: BCE plateaus, CE does not. SWITCH THE GRID TO CE.")
        print("  Amend PREREGISTRATION_MNIST10.md with a dated pre-grid note")
        print("  citing this decision rule. The amendment is legitimate")
        print("  because it precedes every grid job and was decided on")
        print("  convergence, not on M2 - M1.")
    elif p_ce and not p_bce:
        print("VERDICT: CE plateaus and BCE does not. UNEXPECTED.")
        print("  This inverts the synthetic-data prediction. Do not proceed")
        print("  on it -- find out why first.")
    else:
        print("VERDICT: BOTH arms plateau. The loss is NOT the problem.")
        print("  Do not run the grid. Something else is wrong at 10 classes")
        print("  and it will affect every one of the 20 jobs.")
    print("=" * 68)
    print("\nReminder: neither number above is a thesis result. This smoke")
    print("selects a protocol; it does not measure an architecture.")


if __name__ == "__main__":
    main()
