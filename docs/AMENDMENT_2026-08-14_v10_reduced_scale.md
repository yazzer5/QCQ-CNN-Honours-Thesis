# Amendment — v10 10-class MNIST grid ran REDUCED-SCALE, not the locked full protocol

**Date discovered:** 2026-08-14, on receipt of the raw per-run result JSONs
and PBS job logs (previously only a fallback-constants summary was in the
repo — see `results/results_v10_mnist10/summary_v10.json`'s original `_note`).

**Status:** the numbers do not change — this corrects the *label* on results
already written up in `MNIST_LOG.md` §4b, not the results themselves. Every
per-seed test accuracy in the raw JSONs matches the fallback constants that
were already committed. What changes is that those numbers must now be
described as a reduced-scale/feasibility run rather than the final locked
grid.

---

## 1. What the job logs actually say

Every one of the 20 v10 job logs (`results/job_logs/v10_mnist10/v10_m10.o*`)
prints, verbatim:

```
[PROTOCOL] REDUCED-SCALE / FEASIBILITY RUN -- not the locked full protocol.
N_TRAIN=5000 N_VAL=1250 N_TEST=2000 MAX_EPOCHS=200 PATIENCE=20
MIN_EPOCHS_QUANTUM_FRONTEND=30. Label any results from this run accordingly
-- do not mix with full-scale (5000/1250/1000, 100 max epochs) results.
```

Every raw result JSON's `protocol` block confirms `"reduced_scale": true`.
**0 of 20 runs carry a `FULL PROTOCOL` tag; 20 of 20 carry `REDUCED-SCALE`.**
This is the same integrity field (`protocol.reduced_scale`) that the
project's standing rule — check before reading any accuracy number — exists
to catch, and it was missed when the v10 grid was first written up, because
that write-up worked from a fallback-constants summary rather than the raw
job logs.

## 2. Why this was missed the first time

The commit that added v10 to this repo (`b08aae2` / `59c9ab0`) worked from a
prior upload that did not include the raw `result_*.json` files or job logs
— only figures and constants embedded in `code/v10_mnist10/scripts/
plot_v10_results.py`. Those constants were correctly transcribed, but the
protocol metadata that would have surfaced this flag was not available at
the time, and the write-up presented the grid as the completed, final v10
run. `summary_v10.json`'s own `_note` field already flagged that the raw
JSONs were missing; this amendment is what happens now that they've arrived.

## 3. What is and isn't affected

- **Accuracy numbers, MDE, and the seed-instability finding in
  `MNIST_LOG.md` §4b are unaffected** — they were computed correctly from
  the (correctly transcribed) fallback constants, and now have raw-JSON
  backing.
- **`N_TEST=2000` vs the binary grids' `N_TEST=1000`, and `MAX_EPOCHS=200`
  vs `100`** are protocol differences from the locked binary-grid protocol,
  not from some other "true" 10-class protocol — there is no other v10 run
  to compare against. Whether 200/2000 or 100/1000 should be the locked
  10-class protocol has not yet been decided.
- **The core finding — M2−M1 null at MDE ≈16 pp, QNN-head seed instability
  12–15 pp vs 5–6 pp for classical heads — should be treated as a feasibility
  result pending a locked-protocol confirmation run**, not as the final word
  on 10-class M2 vs M1. This is consistent with the project's existing
  position that the 10-class question is underpowered and open (§9's
  six-step plan already lists "run 10-class MNIST properly" as the top
  remaining priority; that item now explicitly includes locking the
  protocol, not only fixing the readout).

## 4. Action taken

- `MNIST_LOG.md` §4b relabelled to describe the run as reduced-scale/
  feasibility, with this amendment linked at the top of the section.
- `README.md` and `results/README.md` status lines corrected to say
  "feasibility run" rather than "grid" for v10.
- Raw `result_*.json` files and job logs added to
  `results/results_v10_mnist10/` and `results/job_logs/v10_mnist10/`,
  replacing reliance on the fallback-constants summary.
- No new full-scale v10 run has been submitted. That remains future work.
