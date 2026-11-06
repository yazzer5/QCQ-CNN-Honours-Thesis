#!/usr/bin/env python3
"""
v15: independent Qiskit Aer reference for the torch noise model. Qiskit only, no torch.

    python3 scripts/make_aer_reference_v15.py --calib calib/ibm_brisbane_calib.json \
        --out calib/aer_reference_v15.npz

scripts/verify_v15.py (torch) then compares quanv_noise_v15.py against this file.
The package ships a copy made with qiskit 2.6.0 / qiskit-aer 0.17.2; setup_v15.sh
regenerates it on Gadi when qiskit-aer is importable and verify reads whichever
is present (the Gadi one, if made).

What it records
  A. NATIVE CHANNELS from NoiseModel.from_backend(FakeBrisbane) -- IBM's own
     noise model -- for sx on two calibrated qubits and ecr on the edge between
     them, as Pauli-transfer matrices, plus their readout errors. The torch model,
     built from the same qubits' calibration, must reproduce these exactly.
  B. CIRCUIT 4 with the SAME channels built from Qiskit primitives
     (depolarizing_error + thermal_relaxation_error, depolarizing parameter from
     Aer's own device model where available), density-matrix simulation,
     alpha in {0.5, 1, 2, 4, 16}, device-median calibration: <Z_a>, <Z_b> for
     64 random patches. Must match the torch model to ~1e-10.
  C. A random 4-qubit RX/RY/RZ/CNOT circuit (the M1 gate set) under the same
     channels: <Z_0..3>. Must match to ~1e-10.
  D. FULL DEVICE SIMULATION (informational): the Circuit-4 block transpiled to
     ibm_brisbane's native gates on the calibrated edge and simulated with
     AerSimulator(noise_model=NoiseModel.from_backend(FakeBrisbane)). The gap to
     the torch model on the same qubits measures the logical-gate approximation
     (unscheduled, CX->ECR single-qubit extras not charged). Reported, not gated.
"""
import argparse
import json
import math

import numpy as np
import qiskit_aer  # noqa: F401  (import before anything else: some builds need it first)
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import CRZGate
try:
    from qiskit.circuit.library import UnitaryGate
except ImportError:  # older qiskit
    from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import DensityMatrix, Operator, Pauli, PTM
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error

ALPHAS = [0.5, 1.0, 2.0, 4.0, 16.0]


def depol_param_qiskit(err, relax_error, n):
    """Aer's own device-model formula when importable, else the documented formula."""
    try:
        from qiskit_aer.noise.device.models import _device_depolarizing_error
        e = _device_depolarizing_error(list(range(n)), err, relax_error)
        if e is None:
            return 0.0, "aer"
        # the depolarizing error's identity probability -> p
        R = np.real(PTM(e.to_quantumchannel()).data)      # depolarizing PTM: diag entries 1 - p
        return float(1 - R[1, 1]), "aer"
    except Exception:  # noqa: BLE001
        from qiskit.quantum_info import average_gate_fidelity
        dim = 2 ** n
        err = min(err, dim / (dim + 1))
        f = average_gate_fidelity(relax_error.to_quantumchannel()) if relax_error is not None else 1.0
        p = dim * (err - (1 - f)) / (dim * f - 1)
        return float(min(max(p, 0.0), 4 ** n / (4 ** n - 1))), "formula"


def ptm_mine_order(R, n):
    """Qiskit PTM index has qubit 0 least significant; ours has logical qubit 0 most."""
    if n == 1:
        return R
    T = R.reshape([4] * (2 * n))
    perm = list(range(n))[::-1] + [n + i for i in range(n)][::-1]
    return T.transpose(perm).reshape(4 ** n, 4 ** n)


def channels(q_a, q_b, edge, alpha):
    """Qiskit QuantumErrors for one SX on each logical qubit and one ECR on (a, b)."""
    out = {}
    srcs = []
    sx = []
    for qp in (q_a, q_b):
        rel = thermal_relaxation_error(qp["t1"] / alpha, min(qp["t2"], 2 * qp["t1"]) / alpha, qp["sx_length"])
        p, src = depol_param_qiskit(alpha * qp["sx_error"], rel, 1)
        srcs.append(src)
        sx.append(depolarizing_error(p, 1).compose(rel))
    rel_a = thermal_relaxation_error(q_a["t1"] / alpha, min(q_a["t2"], 2 * q_a["t1"]) / alpha, edge["ecr_length"])
    rel_b = thermal_relaxation_error(q_b["t1"] / alpha, min(q_b["t2"], 2 * q_b["t1"]) / alpha, edge["ecr_length"])
    rel2 = rel_a.expand(rel_b)                       # rel_a on error-qubit 0 (= logical a)
    p2, src = depol_param_qiskit(alpha * edge["ecr_error"], rel2, 2)
    srcs.append(src)
    out["sx"] = sx
    out["ecr"] = depolarizing_error(p2, 2).compose(rel2)
    out["ro"] = [(min(alpha * qp["readout_error"], .5),) * 2 for qp in (q_a, q_b)]
    return out, set(srcs)


def readout(z, ro):
    pa, pb = ro
    return (1 - pa - pb) * z + (pb - pa)


def zexp(rho, n, w):
    lab = ["I"] * n
    lab[n - 1 - w] = "Z"
    return float(np.real(DensityMatrix(rho).expectation_value(Pauli("".join(lab)))))


def circuit4_custom(X, w, ch):
    """Logical Circuit 4 block with charged native-gate channels, Aer density matrix.
    X: (n, 2) patch inputs -> (n, 2) measured <Z_a>, <Z_b>. One batched Aer call."""
    nm = NoiseModel(basis_gates=["ry", "unitary"])
    ry_err = [ch["sx"][i].compose(ch["sx"][i]) for i in range(2)]
    nm.add_quantum_error(ry_err[0], "ry", [0])
    nm.add_quantum_error(ry_err[1], "ry", [1])
    crz_err = ch["ecr"].compose(ch["ecr"])                 # error-qubit 0 = logical a
    nm.add_quantum_error(crz_err, "crz_ba", [0, 1])
    nm.add_quantum_error(crz_err, "crz_ab", [0, 1])
    # CRZ as labelled unitaries on qargs [0, 1] so the custom channel attaches with
    # error-qubit 0 = logical a in both (density_matrix has no native crz)
    ba = QuantumCircuit(2); ba.crz(w[2], 1, 0)
    U_ba, U_ab = Operator(ba), Operator(CRZGate(w[5]))
    circs = []
    for xa, xb in X:
        qc = QuantumCircuit(2)
        qc.ry(np.pi * xa, 0)
        qc.ry(np.pi * xb, 1)
        qc.ry(w[0], 0); qc.ry(w[1], 1)
        qc.append(UnitaryGate(U_ba, label="crz_ba"), [0, 1])            # ctrl q_b tgt q_a
        qc.ry(w[3], 0); qc.ry(w[4], 1)
        qc.append(UnitaryGate(U_ab, label="crz_ab"), [0, 1])            # ctrl q_a tgt q_b
        qc.save_density_matrix()
        circs.append(qc)
    res = AerSimulator(method="density_matrix", noise_model=nm).run(circs).result()
    return [[readout(zexp(res.data(k)["density_matrix"], 2, i), ch["ro"][i]) for i in range(2)]
            for k in range(len(circs))]


def circuit4_custom_ptm_check(ch):
    """The Qiskit native channels as PTMs in OUR qubit ordering (for B's sanity)."""
    return ptm_mine_order(PTM(ch["ecr"].to_quantumchannel()).data, 2)


def henderson_like_circuit(rng):
    gates = []
    for _ in range(14):
        k = rng.integers(4)
        if k < 3:
            gates.append((["RX", "RY", "RZ"][k], [int(rng.integers(4))], float(rng.uniform(0, 2 * np.pi))))
        else:
            c, t = rng.choice(4, 2, replace=False)
            gates.append(("CNOT", [int(c), int(t)], 0.0))
    return gates


def henderson_custom(X4, gates, ch):
    nm = NoiseModel(basis_gates=["ry", "rx", "rz", "cx"])
    sxe = ch["sx"][0].compose(ch["sx"][0])
    for q in range(4):
        nm.add_quantum_error(sxe, "ry", [q])
        nm.add_quantum_error(sxe, "rx", [q])
    for c in range(4):
        for t in range(4):
            if c != t:
                nm.add_quantum_error(ch["ecr"], "cx", [c, t])
    circs = []
    for x4 in X4:
        qc = QuantumCircuit(4)
        for q in range(4):
            qc.ry(np.pi * x4[q], q)
        for name, wires, ang in gates:
            if name == "RX": qc.rx(ang, wires[0])
            elif name == "RY": qc.ry(ang, wires[0])
            elif name == "RZ": qc.rz(ang, wires[0])
            else: qc.cx(wires[0], wires[1])
        qc.save_density_matrix()
        circs.append(qc)
    res = AerSimulator(method="density_matrix", noise_model=nm).run(circs).result()
    return [[readout(zexp(res.data(k)["density_matrix"], 4, i), ch["ro"][0]) for i in range(4)]
            for k in range(len(circs))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cal = json.load(open(a.calib))
    med = cal["median"]
    rng = np.random.default_rng(20261017)

    # choose the calibrated edge closest to the device median (both directions exist on Eagle? use as listed)
    def score(e, ed):
        qa, qb = map(int, e.split("-"))
        A, B = cal["qubits"][str(qa)], cal["qubits"][str(qb)]
        return (abs(math.log(ed["ecr_error"] / med["ecr_error"])) + abs(math.log(A["sx_error"] / med["sx_error"]))
                + abs(math.log(B["sx_error"] / med["sx_error"])) + abs(math.log(A["t1"] / med["t1"])) + abs(math.log(B["t1"] / med["t1"])))
    edge_key = min((e for e, ed in cal["edges"].items() if ed["ecr_error"] < .5), key=lambda e: score(e, cal["edges"][e]))
    qa, qb = map(int, edge_key.split("-"))
    print(f"[AER] calibrated edge used for A and D: {edge_key}")
    out = dict(qa=qa, qb=qb, calib_sha=cal["sha256"])

    # ---- A. IBM's own channels -------------------------------------------------------
    backend = None
    for mod in ("qiskit_ibm_runtime.fake_provider", "qiskit.providers.fake_provider"):
        try:
            backend = __import__(mod, fromlist=["FakeBrisbane"]).FakeBrisbane()
            break
        except Exception:  # noqa: BLE001
            continue
    if backend is None:
        raise SystemExit("FakeBrisbane not importable")
    nm = NoiseModel.from_backend(backend)
    loc = nm._local_quantum_errors
    out["A_sx_a"] = PTM(loc["sx"][(qa,)].to_quantumchannel()).data
    out["A_sx_b"] = PTM(loc["sx"][(qb,)].to_quantumchannel()).data
    out["A_ecr"] = ptm_mine_order(PTM(loc["ecr"][(qa, qb)].to_quantumchannel()).data, 2)
    ro = nm._local_readout_errors
    out["A_ro"] = np.array([np.asarray(ro[(q,)].probabilities) for q in (qa, qb)])   # [[P(0|0),P(1|0)],[P(0|1),P(1|1)]]
    print(f"[AER] A: from_backend sx/ecr channels on qubits {qa},{qb} recorded")

    # ---- B. Circuit 4, median calibration, several alphas ----------------------------
    X = rng.uniform(0, 1, size=(64, 2))
    Wt = rng.uniform(0, 2 * np.pi, size=6)
    out["B_x"], out["B_w"], out["B_alphas"] = X, Wt, np.array(ALPHAS)
    res, srcs = [], set()
    for al in ALPHAS:
        ch, s = channels(med, med, med, al)
        srcs |= s
        res.append(circuit4_custom(X, Wt, ch))
    out["B_z"] = np.array(res)                                  # (n_alpha, 64, 2)
    print(f"[AER] B: Circuit 4 x {len(ALPHAS)} alphas done (depolarizing parameter source: {sorted(srcs)})")
    out["B_param_source"] = np.array(sorted(srcs))

    # B'. same at alpha=1 on the calibrated edge (per-qubit values)
    ch, _ = channels(cal["qubits"][str(qa)], cal["qubits"][str(qb)], cal["edges"][edge_key], 1.0)
    out["B_edge_z"] = np.array(circuit4_custom(X, Wt, ch))

    # ---- C. random 4-qubit circuit, M1 gate set ----------------------------------------
    gates = henderson_like_circuit(rng)
    X4 = rng.uniform(0, 1, size=(32, 4))
    out["C_x"] = X4
    out["C_gates"] = np.array(json.dumps(gates))
    resC = []
    for al in (1.0, 4.0):
        ch, _ = channels(med, med, med, al)
        resC.append(henderson_custom(X4, gates, ch))
    out["C_alphas"] = np.array([1.0, 4.0])
    out["C_z"] = np.array(resC)
    print("[AER] C: random 4-qubit RX/RY/RZ/CNOT circuit done")

    # ---- D. full device simulation on the calibrated edge (informational) -----------
    # Restrict IBM's noise model to the two active physical qubits (a 127-qubit
    # density matrix is impossible): every native op on (qa, qb) keeps its
    # from_backend channel, remapped to qubits 0, 1.
    pmap = {qa: 0, qb: 1}
    nmD = NoiseModel(basis_gates=["sx", "x", "rz", "ecr", "id"])
    for name, errs in loc.items():
        for qs, err in errs.items():
            if all(q in pmap for q in qs):
                nmD.add_quantum_error(err, name, [pmap[q] for q in qs])
    simD = AerSimulator(method="density_matrix", noise_model=nmD)
    circs = []
    for x in X[:32]:
        qc = QuantumCircuit(2)
        qc.ry(np.pi * x[0], 0); qc.ry(np.pi * x[1], 1)
        qc.ry(Wt[0], 0); qc.ry(Wt[1], 1); qc.crz(Wt[2], 1, 0)
        qc.ry(Wt[3], 0); qc.ry(Wt[4], 1); qc.crz(Wt[5], 0, 1)
        tq = transpile(qc, backend, initial_layout=[qa, qb], optimization_level=1, seed_transpiler=1)
        small = QuantumCircuit(2)
        for ins in tq.data:
            qs = [tq.find_bit(q).index for q in ins.qubits]
            if not qs:
                continue
            if any(q not in pmap for q in qs):
                raise SystemExit(f"D: transpiled op {ins.operation.name} on unexpected qubits {qs} (routing?)")
            small.append(ins.operation, [pmap[q] for q in qs])
        counts = small.count_ops()
        small.save_density_matrix()
        circs.append(small)
    resD = simD.run(circs).result()
    zD = [[readout(zexp(resD.data(k)["density_matrix"], 2, i),
                   (ro[(q,)].probabilities[0][1], ro[(q,)].probabilities[1][0]))
           for i, q in enumerate((qa, qb))] for k in range(len(circs))]
    out["D_z"] = np.array(zD)
    out["D_ops"] = np.array(json.dumps({k: int(v) for k, v in counts.items()}))
    print(f"[AER] D: full transpiled from_backend simulation done; native ops per block: {dict(counts)}")

    import qiskit
    out["versions"] = np.array(json.dumps(dict(qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__)))
    np.savez(a.out, **out)
    print(f"[AER] wrote {a.out}")


if __name__ == "__main__":
    main()
