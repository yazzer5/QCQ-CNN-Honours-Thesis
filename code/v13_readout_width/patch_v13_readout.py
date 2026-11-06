#!/usr/bin/env python3
"""
v13 code patch: widen the QNN back-end readout beyond the Z basis.

    python patch_v13_readout.py /path/to/repo        (run by setup_v13.sh)

WHAT IT CHANGES
  qnn_torch.py   n_obs may be 1, 3, 6 or 15 (was 1 or 3).
                   3  = <Z0>,<Z1>,<ZZ>              1 measurement setting (ZZ basis)
                   6  = 3 + <X0>,<X1>,<XX>          2 settings (ZZ, XX)
                   15 = all 15 two-qubit Paulis     9 settings ({X,Y,Z} x {X,Y,Z})
                 The n_obs = 1 and 3 code paths are NOT touched: same ops, same
                 order, so v10/v12 results reproduce bit for bit.
                 Wider readouts are appended AFTER the original 3 columns.
  train_gadi.py  --n_obs choices [1, 3] -> [1, 3, 6, 15]. Nothing else.
                 expected_params() is already generic in n_obs (v10).

SAFETY
  - Every edit is anchored on exact text. If an anchor is missing (code changed
    since v10), NOTHING is written and the script exits 1 with what to paste.
  - Originals are kept as qnn_torch.py.v12bak / train_gadi.py.v12bak, which
    scripts/verify_v13_readout.py uses to prove the n_obs=3 path is unchanged.
  - Refuses to patch twice.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# V13_READOUT_PATCH"

HELPERS = '''

''' + MARK + ''' ---------------------------------------------------------
# Wider readouts for the 2-qubit back-end (v13). Appended after the original
# <Z0>,<Z1>,<ZZ> columns, so column order is nested: w3 c w6 c w15.
#
# Basis convention (same as the diagonal path above): index = b0 + 2*b1, so an
# operator P_q0 (x) P_q1 is the 4x4 matrix kron(P_q1, P_q0).
#
# Why these sets (see PREREGISTRATION_v13.md):
#   RealAmplitudes is a REAL orthogonal U, so any observable with an even number
#   of Y's stays real-symmetric under U and its expectation contains only
#   cos(phase differences) of the encoded state. The Z basis gives 3 such
#   features; Z+X gives all 6 that exist. Only observables with an odd number
#   of Y's give the 6 sin(phase difference) features. Hence 3 -> 6 -> 15.
_V13_EXTRA = {
    6:  ["XI", "IX", "XX"],
    15: ["XI", "IX", "XX",
         "YI", "IY", "YY", "XY", "YX", "XZ", "ZX", "YZ", "ZY"],
}
# Measurement settings needed on hardware (shots scale with this).
V13_MEASUREMENT_SETTINGS = {1: 1, 3: 1, 6: 2, 15: 9}
# Name convention: first letter acts on q0, second on q1. "ZI" = <Z0>.
V13_READOUT_NAMES = {
    1: ["ZZ"],
    3: ["ZI", "IZ", "ZZ"],
    6: ["ZI", "IZ", "ZZ"] + _V13_EXTRA[6],
    15: ["ZI", "IZ", "ZZ"] + _V13_EXTRA[15],
}
_V13_PAULI = {
    "I": [[1, 0], [0, 1]],
    "X": [[0, 1], [1, 0]],
    "Y": [[0, -1j], [1j, 0]],
    "Z": [[1, 0], [0, -1]],
}
_V13_CACHE = {}


def _v13_pauli_mats(n_obs, cdt, dev):
    key = (n_obs, cdt, str(dev))
    if key not in _V13_CACHE:
        mats = []
        for name in _V13_EXTRA[n_obs]:
            p0 = torch.tensor(_V13_PAULI[name[0]], dtype=cdt, device=dev)
            p1 = torch.tensor(_V13_PAULI[name[1]], dtype=cdt, device=dev)
            mats.append(torch.kron(p1, p0))
        _V13_CACHE[key] = torch.stack(mats)            # (n_extra, 4, 4)
    return _V13_CACHE[key]


def _v13_extra_expectations(psi, n_obs):
    """<psi|P|psi> for the extra Paulis; psi is (B, 4) complex. Returns (B, n_extra) real."""
    M = _v13_pauli_mats(n_obs, psi.dtype, psi.device)
    Mpsi = torch.einsum("nij,bj->bni", M, psi)         # (B, n, 4)
    return (psi.conj().unsqueeze(1) * Mpsi).sum(-1).real
'''


def die(msg):
    print("PATCH ABORTED, nothing written.\n" + msg)
    print("\nPaste the output of:\n  grep -n 'n_obs' qnn_torch.py train_gadi.py")
    sys.exit(1)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    qp, tp = root / "qnn_torch.py", root / "train_gadi.py"
    for p in (qp, tp):
        if not p.exists():
            die(f"{p} not found")
    q, t = qp.read_text(), tp.read_text()
    if MARK in q:
        die("qnn_torch.py is already patched (marker found). Start from a clean copy.")

    # --- qnn_torch.py: 1. constructor assertion ------------------------------
    a_re = re.compile(r'assert n_obs in \(1, 3\), f"n_obs must be 1 or 3, got \{n_obs\}"')
    if len(a_re.findall(q)) != 1:
        die("qnn_torch.py: expected exactly one 'assert n_obs in (1, 3) ...' line.")
    q = a_re.sub('assert n_obs in (1, 3, 6, 15), f"n_obs must be 1, 3, 6 or 15, got {n_obs}"', q)

    # --- qnn_torch.py: 2. the n_obs=3 return line ----------------------------
    r_re = re.compile(r"^(?P<ind>[ \t]+)return torch\.stack\(\[p @ z0_obs, p @ z1_obs, p @ zz_obs\], dim=-1\)\.to\(x\.dtype\)[ \t]*$",
                      re.M)
    m = r_re.findall(q)
    if len(m) != 1:
        die(f"qnn_torch.py: expected exactly one n_obs=3 return line, found {len(m)}.")
    ind = r_re.search(q).group("ind")
    new_ret = (f"{ind}base = torch.stack([p @ z0_obs, p @ z1_obs, p @ zz_obs], dim=-1)\n"
               f"{ind}if self.n_obs == 3:\n"
               f"{ind}    return base.to(x.dtype)\n"
               f"{ind}extra = _v13_extra_expectations(psi, self.n_obs).to(base.dtype)  {MARK}\n"
               f"{ind}return torch.cat([base, extra], dim=-1).to(x.dtype)")
    q = r_re.sub(lambda _: new_ret, q)
    if "psi" not in q or "import torch" not in q:
        die("qnn_torch.py: no 'psi' statevector or 'import torch' found.")
    q = q.rstrip("\n") + "\n" + HELPERS

    # --- train_gadi.py: CLI choices ------------------------------------------
    c_re = re.compile(r'(ap\.add_argument\(\s*"--n_obs"[^)]*?choices=)\[1, 3\]', re.S)
    if len(c_re.findall(t)) != 1:
        die('train_gadi.py: expected one --n_obs argument with choices=[1, 3].')
    t = c_re.sub(r"\1[1, 3, 6, 15]", t)
    hard = re.findall(r"n_obs\s+in\s+\(1,\s*3\)|N_OBS\s+in\s+\(1,\s*3\)", t)
    if hard:
        die(f"train_gadi.py has another hard-coded (1, 3) check: {hard}. Needs a look.")

    # --- write ---------------------------------------------------------------
    for p in (qp, tp):
        bak = p.with_name(p.name + ".v12bak")
        if not bak.exists():
            shutil.copy2(p, bak)
    qp.write_text(q)
    tp.write_text(t)
    print(f"Patched {qp.name} and {tp.name} in {root}")
    print(f"Backups: {qp.name}.v12bak, {tp.name}.v12bak")
    print("Next: python scripts/verify_v13_readout.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
