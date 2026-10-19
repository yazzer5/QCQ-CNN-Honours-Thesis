#!/usr/bin/env python3
"""
v15: the torch noise model (quanv_noise_v15.py) against Qiskit Aer. Torch only.

    python3 scripts/verify_noise_v15.py [calib/aer_reference_v15.npz]

Reads the Aer reference (scripts/make_aer_reference_v15.py) and checks:
  A. our native-gate channels == NoiseModel.from_backend(FakeBrisbane)'s channels
     on the same calibrated qubits (sx on each qubit, ecr on the edge, readout)
  B. Circuit 4 under noise, device median, alpha in {0.5,1,2,4,16}: == Aer
  C. random 4-qubit RX/RY/RZ/CNOT circuit (the M1 gate set): == Aer
  D. (informational) gap to a full transpiled from_backend simulation, shown
     next to the size of the noise effect itself
Also run from verify_v15.py. Exit code 1 on any FAIL.
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import quanv_noise_v15 as N  # noqa: E402

TOL_CH = 1e-9      # channel PTMs
TOL_Z = 1e-6       # expectation values. Aer's QuantumError drops mixture terms with
                   # probability < 1e-8 when composing channels, which leaves ~1e-8
                   # differences (exact match, 1e-15, where no term is dropped).
FAIL = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        FAIL.append(name)


def gate_u(name, wires, ang):
    I2 = np.eye(2, dtype=complex)
    c, s = np.cos(ang / 2), np.sin(ang / 2)
    G = {"RX": np.array([[c, -1j * s], [-1j * s, c]]), "RY": np.array([[c, -s], [s, c]]),
         "RZ": np.diag([np.exp(-1j * ang / 2), np.exp(1j * ang / 2)])}
    if name in G:
        ops = [I2] * 4
        ops[wires[0]] = G[name]
        U = ops[0]
        for o in ops[1:]:
            U = np.kron(U, o)
        return U
    c_, t_ = wires
    U = np.zeros((16, 16), dtype=complex)
    for i in range(16):
        bits = [(i >> (3 - k)) & 1 for k in range(4)]
        if bits[c_]:
            bits[t_] ^= 1
        j = sum(b << (3 - k) for k, b in enumerate(bits))
        U[j, i] = 1
    return U


def run(npz_path, calib_path):
    cal = N.load_calib(calib_path)
    R = np.load(npz_path, allow_pickle=False)
    print(f"  Aer reference: {npz_path}  (made with {str(R['versions'])})")
    check("Aer reference made from the same calibration file", str(R["calib_sha"]) == cal["sha256"],
          f"{str(R['calib_sha'])[:12]} vs {cal['sha256'][:12]}")
    qa, qb = int(R["qa"]), int(R["qb"])

    # A ---------------------------------------------------------------------------
    NPe = N.NoiseParams(cal, 1.0, phys=(qa, qb))
    d_sx_a = np.abs(NPe.q[0]["sx"] - np.real(R["A_sx_a"])).max()
    d_sx_b = np.abs(NPe.q[1]["sx"] - np.real(R["A_sx_b"])).max()
    d_ecr = np.abs(NPe.ecr - np.real(R["A_ecr"])).max()
    check(f"A. sx channel == IBM from_backend, qubit {qa}", d_sx_a < TOL_CH, f"max|d|={d_sx_a:.1e}")
    check(f"A. sx channel == IBM from_backend, qubit {qb}", d_sx_b < TOL_CH, f"max|d|={d_sx_b:.1e}")
    check(f"A. ecr channel == IBM from_backend, edge {qa}-{qb}", d_ecr < TOL_CH, f"max|d|={d_ecr:.1e}")
    ro = R["A_ro"]
    d_ro = max(abs(NPe.q[i]["ro"][0] - ro[i][0][1]) + abs(NPe.q[i]["ro"][1] - ro[i][1][0]) for i in range(2))
    check("A. readout error == IBM from_backend", d_ro < 1e-12, f"max|d|={d_ro:.1e}")

    # B ---------------------------------------------------------------------------
    X, W = torch.tensor(R["B_x"]), torch.tensor(R["B_w"])
    for k, al in enumerate(R["B_alphas"]):
        z = N.circuit4_expvals(X * np.pi, W, N.NoiseParams(cal, float(al))).numpy()
        d = np.abs(z - R["B_z"][k]).max()
        check(f"B. Circuit 4, median calibration, alpha={al:g}: == Aer", d < TOL_Z, f"max|d|={d:.1e}")
    z = N.circuit4_expvals(X * np.pi, W, NPe).numpy()
    d = np.abs(z - R["B_edge_z"]).max()
    check(f"B. Circuit 4, calibrated edge {qa}-{qb}, alpha=1: == Aer", d < TOL_Z, f"max|d|={d:.1e}")

    # C ---------------------------------------------------------------------------
    gates = [(n_, w_, gate_u(n_, w_, a_)) for n_, w_, a_ in json.loads(str(R["C_gates"]))]
    X4 = torch.tensor(R["C_x"])
    for k, al in enumerate(R["C_alphas"]):
        NP = N.NoiseParams(cal, float(al))
        z = N.henderson_expvals(X4 * np.pi, N.henderson_ptm_rows(gates, NP), NP).numpy()
        d = np.abs(z - R["C_z"][k]).max()
        check(f"C. 4-qubit RX/RY/RZ/CNOT circuit ({len(gates)} gates), alpha={al:g}: == Aer", d < TOL_Z, f"max|d|={d:.1e}")

    # D (informational) -------------------------------------------------------------
    Xd = X[: len(R["D_z"])]
    z_model = N.circuit4_expvals(Xd * np.pi, W, NPe).numpy()
    z_clean = N.circuit4_expvals(Xd * np.pi, W, N.NoiseParams(cal, 0.0)).numpy()
    gap = np.abs(z_model - R["D_z"])
    eff = np.abs(R["D_z"] - z_clean)
    dm, dd = (z_model - z_clean).ravel(), (R["D_z"] - z_clean).ravel()
    corr, slope = float(np.corrcoef(dm, dd)[0, 1]), float(dm @ dd / (dm @ dm))
    print(f"  INFO D. full transpiled ibm_brisbane simulation, edge {qa}-{qb}, alpha=1 (native ops {str(R['D_ops'])}):")
    print(f"         noise shift on <Z>: device mean {eff.mean():.4f}, model mean {np.abs(dm).mean():.4f}; "
          f"corr(model shift, device shift) = {corr:.3f}; device/model size = {slope:.2f}")
    print(f"         residual |model - device| mean {gap.mean():.4f} (gate placement of the non-unital T1 decay;"
          f" the logical-gate model is ~{100 * (slope - 1):.0f}% {'optimistic' if slope > 1 else 'pessimistic'})")
    return dict(D_effect=float(eff.mean()), D_gap=float(gap.mean()), D_corr=corr, D_slope=slope)


if __name__ == "__main__":
    npz = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "calib", "aer_reference_v15.npz")
    run(npz, os.path.join(ROOT, "calib", "ibm_brisbane_calib.json"))
    print("\nNOISE MODEL CHECKS " + ("FAILED: " + str(FAIL) if FAIL else "PASSED"))
    sys.exit(1 if FAIL else 0)
