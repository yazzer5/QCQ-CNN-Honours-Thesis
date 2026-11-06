#!/usr/bin/env python3
"""
v15: extract the IBM device calibration used by the front-end noise model.

Needs Qiskit only (no torch). Run in an env with qiskit-ibm-runtime (or an older
qiskit that ships FakeBrisbane):

    python3 scripts/extract_ibm_calibration.py --out calib/ibm_brisbane_calib.json

The calibration that v15 is PRE-REGISTERED on is the one shipped in the package
(calib/ibm_brisbane_calib.json, FakeBrisbane snapshot). On Gadi, setup_v15.sh runs
this script again with --check against that file: a matching sha256 proves the
same snapshot; a mismatch is reported (it means a different Qiskit snapshot) and
the PACKAGED file is still the one used.

What is recorded (ibm_brisbane, Eagle r3, 127 qubits):
  per qubit : T1, T2, sx error + duration, readout_error (the symmetric value
              NoiseModel.from_backend uses for BackendV2), and the asymmetric
              P(1|0), P(0|1) for reference
  per edge  : ecr error + duration
  median    : the device-median of each quantity (what the model uses). Faulty
              entries (gate error >= 0.5, i.e. IBM's "1.0 = broken" marker) are
              excluded from the medians and counted.
"""
import argparse
import hashlib
import json
import statistics
import sys


def load_backend():
    errs = []
    for mod in ("qiskit_ibm_runtime.fake_provider", "qiskit.providers.fake_provider"):
        try:
            m = __import__(mod, fromlist=["FakeBrisbane"])
            return m.FakeBrisbane(), mod
        except Exception as e:  # noqa: BLE001
            errs.append(f"{mod}: {e}")
    sys.exit("FakeBrisbane not importable:\n  " + "\n  ".join(errs))


def versions():
    out = {}
    for name in ("qiskit", "qiskit_aer", "qiskit_ibm_runtime"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001
            out[name] = None
    return out


def canonical_sha(d):
    body = {k: v for k, v in d.items() if k not in ("sha256", "versions", "note")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def extract():
    b, mod = load_backend()
    p = b.properties()
    nq = b.num_qubits
    qubits, edges = {}, {}
    for q in range(nq):
        qp = {x.name: x.value for x in p.qubits[q]}
        t1, t2 = p.t1(q), p.t2(q)
        qubits[str(q)] = dict(
            t1=t1, t2=t2,
            sx_error=p.gate_error("sx", q), sx_length=p.gate_length("sx", q),
            readout_error=float(qp["readout_error"]),
            p_meas1_prep0=float(qp["prob_meas1_prep0"]),
            p_meas0_prep1=float(qp["prob_meas0_prep1"]),
        )
    for g in p.gates:
        if g.gate == "ecr":
            a, c = g.qubits
            edges[f"{a}-{c}"] = dict(ecr_error=p.gate_error("ecr", [a, c]),
                                     ecr_length=p.gate_length("ecr", [a, c]))

    def med(vals):
        return float(statistics.median(vals))

    good_q = [v for v in qubits.values() if v["sx_error"] < 0.5]
    good_e = [v for v in edges.values() if v["ecr_error"] < 0.5]
    median = dict(
        t1=med([v["t1"] for v in good_q]),
        t2=med([v["t2"] for v in good_q]),
        sx_error=med([v["sx_error"] for v in good_q]),
        sx_length=med([v["sx_length"] for v in good_q]),
        ecr_error=med([v["ecr_error"] for v in good_e]),
        ecr_length=med([v["ecr_length"] for v in good_e]),
        readout_error=med([v["readout_error"] for v in good_q]),
        p_meas1_prep0=med([v["p_meas1_prep0"] for v in good_q]),
        p_meas0_prep1=med([v["p_meas0_prep1"] for v in good_q]),
    )
    d = dict(
        backend="ibm_brisbane (FakeBrisbane snapshot)",
        source_module=mod,
        last_update_date=str(p.last_update_date),
        num_qubits=nq,
        excluded=dict(qubits=nq - len(good_q), edges=len(edges) - len(good_e)),
        median=median, qubits=qubits, edges=edges,
    )
    d["sha256"] = canonical_sha(d)
    d["versions"] = versions()
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="write the calibration JSON here")
    ap.add_argument("--check", help="compare against this packaged calibration JSON")
    a = ap.parse_args()
    d = extract()
    m = d["median"]
    print(f"[V15-CALIB] backend={d['backend']}  snapshot={d['last_update_date']}  "
          f"from={d['source_module']}  versions={d['versions']}")
    print(f"[V15-CALIB] median T1={m['t1']*1e6:.1f}us T2={m['t2']*1e6:.1f}us  "
          f"sx err={m['sx_error']:.3e} len={m['sx_length']*1e9:.0f}ns  "
          f"ecr err={m['ecr_error']:.3e} len={m['ecr_length']*1e9:.0f}ns  "
          f"readout={m['readout_error']:.4f}  "
          f"excluded={d['excluded']}")
    print(f"[V15-CALIB] sha256={d['sha256']}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(d, f, indent=1, sort_keys=True)
        print(f"[V15-CALIB] wrote {a.out}")
    if a.check:
        ref = json.load(open(a.check))
        same = ref["sha256"] == d["sha256"]
        print(f"[V15-CALIB] packaged sha256={ref['sha256']}  "
              f"{'MATCH (same snapshot)' if same else 'DIFFERENT snapshot -- the packaged file is still the one used'}")


if __name__ == "__main__":
    main()
