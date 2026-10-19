#!/usr/bin/env python3
"""
v13.5 code patch: add model M5 = M3 with the quantum front-end FROZEN.

    python3 patch_v135_m5.py /path/to/repo        (run by setup_v135.sh)

WHAT M5 IS
  Exactly M3 (trainable Circuit-4 quanvolution front-end + CNN backbone +
  classical head, at whatever --n_obs width is asked for), built by the
  UNCHANGED build_model("M3", seed), after which every front-end parameter is
  set requires_grad=False. Same seed -> same initial weights as M3, so
  M3 - M5 isolates one thing: whether the front-end is trained.
  (M1 is NOT this: M1's frozen front-end is Henderson RandomLayers, and it
  has a QNN head.)

WHAT IT CHANGES (train_gadi.py only; qnn_torch.py / quanv_torch.py untouched)
  1. --model choices gain "M5".
  2. A wrapper block, inserted just above `if __name__ == "__main__":`,
     rebinds build_model and expected_params. For M1-M4 both wrappers call
     the original function with the original arguments: no code path for
     M1-M4 changes (verified bit-exact by scripts/verify_v135_m5.py).
  3. Model-name rules that mention minimum epochs and list "M3" also get
     "M5", so M5 trains under M3's protocol exactly. Each such edit is
     printed. The verify script's dry run then checks that M5 and M3
     resolve the same min-epochs value.
  4. Runtime self-check: every M5 prints, at process exit,
       [V135] M5 front-end after run: UNCHANGED OK
     if its front-end weights are bit-identical to their initial values.
     The integrity gate requires this line in every M5 log.

SAFETY
  - Anchored on exact text. If an anchor is missing, NOTHING is written and the
    script exits 1 with what to paste.
  - Backup: train_gadi.py.v13bak (used by the verify script).
  - Refuses to patch twice, and refuses an un-patched (non-v13) qnn_torch.py.
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# V135_M5_PATCH"

BLOCK = '''

''' + MARK + ''' ---------------------------------------------------------
# v13.5: M5 = M3 with the quantum front-end frozen at its seed initialisation.
# M1-M4 go through the original functions with the original arguments.
import atexit as _v135_atexit
import torch as _v135_torch

_v135_orig_build_model = build_model
_v135_orig_expected_params = expected_params


def build_model(model_name, *args, **kwargs):
    if model_name != "M5":
        return _v135_orig_build_model(model_name, *args, **kwargs)
    mdl = _v135_orig_build_model("M3", *args, **kwargs)
    fe = getattr(mdl, "front_end", None)
    if fe is None:
        raise RuntimeError("[V135] M5: model has no .front_end attribute; cannot freeze it")
    n = 0
    for p in fe.parameters():
        p.requires_grad_(False)
        n += p.numel()
    if n == 0:
        raise RuntimeError("[V135] M5: M3's front-end has no parameters to freeze")
    snap = {k: v.detach().clone().cpu() for k, v in fe.state_dict().items()}
    print(f"[V135] M5 = M3 with front-end frozen: {n} front-end params "
          f"requires_grad=False ({type(fe).__name__})", flush=True)

    def _v135_check(fe=fe, snap=snap):
        cur = fe.state_dict()
        same = all(_v135_torch.equal(cur[k].detach().cpu(), v) for k, v in snap.items())
        print(f"[V135] M5 front-end after run: {'UNCHANGED OK' if same else 'CHANGED -- FAIL'}",
              flush=True)

    _v135_atexit.register(_v135_check)
    return mdl


def expected_params(model_name, *args, **kwargs):
    # M5 is architecturally M3. Whether total_params() counts frozen tensors is
    # checked by scripts/verify_v135_m5.py, which stops if this is wrong.
    return _v135_orig_expected_params("M3" if model_name == "M5" else model_name, *args, **kwargs)
# end ''' + MARK + '''

'''


def die(msg):
    print("PATCH ABORTED, nothing written.\n" + msg)
    print("\nPaste the output of:\n"
          "  grep -n 'def build_model\\|def expected_params\\|__main__\\|\"--model\"\\|min_epoch\\|front_end =' train_gadi.py")
    sys.exit(1)


def main():
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()
    tp, qp = root / "train_gadi.py", root / "qnn_torch.py"
    for p in (tp, qp):
        if not p.exists():
            die(f"{p} not found")
    t = tp.read_text()
    if MARK in t:
        die("train_gadi.py is already patched (marker found). Start from a clean copy.")
    if "V13_READOUT_PATCH" not in qp.read_text():
        die("qnn_torch.py is not the v13-patched file. Build from qcq_cnn_v13, not v12.")

    for pat, what in ((r"^def build_model\(", "def build_model("),
                      (r"^def expected_params\(", "def expected_params("),
                      (r"^if __name__ == [\"']__main__[\"']:", "if __name__ == '__main__':")):
        n = len(re.findall(pat, t, re.M))
        if n != 1:
            die(f"train_gadi.py: expected exactly one top-level '{what}', found {n}.")
    if "front_end" not in t:
        die("train_gadi.py: no 'front_end' attribute anywhere.")

    # 1. --model choices -----------------------------------------------------
    c_re = re.compile(r"""(\.add_argument\(\s*["']--model["'][^)]*?choices\s*=\s*)(\[[^\]]*\]|\([^)]*\)|[A-Za-z_][A-Za-z0-9_.]*)""",
                      re.S)
    hits = c_re.findall(t)
    if len(hits) != 1:
        die(f"train_gadi.py: expected one --model argument with choices=..., found {len(hits)}.")
    old_choices = hits[0][1]
    if '"M5"' in old_choices or "'M5'" in old_choices:
        die("--model choices already contain M5.")
    t = c_re.sub(lambda m: m.group(1) + f'list({m.group(2)}) + ["M5"]', t)
    print(f"  --model choices: {old_choices}  ->  list({old_choices}) + [\"M5\"]")

    # 2. min-epochs rules keyed on model names -------------------------------
    lines = t.split("\n")
    lit_re = re.compile(r"""([\(\[\{])([^\(\)\[\]\{\}]*["']M3["'][^\(\)\[\]\{\}]*)([\)\]\}])""")
    edits = []

    def add_m5(s):
        def rep(m):
            body = m.group(2)
            if "M5" in body:
                return m.group(0)
            q = '"' if '"M3"' in body else "'"
            sep = ", " if body.strip() else ""
            return f"{m.group(1)}{body.rstrip().rstrip(',')}{sep}{q}M5{q}{m.group(3)}"
        return lit_re.sub(rep, s)

    consts = set()
    for i, ln in enumerate(lines):
        if re.search(r"min_?epoch", ln, re.I) and not ln.lstrip().startswith("#"):
            if lit_re.search(ln):
                new = add_m5(ln)
                if new != ln:
                    edits.append((i + 1, ln.strip(), new.strip())); lines[i] = new
            for name in re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", ln):
                consts.add(name)
    for name in sorted(consts):
        for i, ln in enumerate(lines):
            if re.match(rf"^{name}\s*=\s*", ln) and lit_re.search(ln):
                new = add_m5(ln)
                if new != ln:
                    edits.append((i + 1, ln.strip(), new.strip())); lines[i] = new
                    others = [j + 1 for j, l2 in enumerate(lines) if re.search(rf"\b{name}\b", l2) and j != i]
                    print(f"  NOTE: constant {name} now includes M5; it is also used on lines {others}."
                          f" Paste them if unsure.")
    t = "\n".join(lines)
    if edits:
        for ln_no, a, b in edits:
            print(f"  min-epochs rule, line {ln_no}:\n      {a}\n   -> {b}")
    else:
        print("  min-epochs: no model-name rule listing M3 found. The verify dry run checks"
              " that M5 and M3 resolve the same value.")

    # 2b. v12-v13 warmup rule (amendment A1-v13.5, 2026-10-12) ---------------
    w_old = 'needs_warmup = model_name in ("M2", "M3") or isinstance(model.back_end, QNNBackend)'
    if t.count(w_old) != 1:
        die("train_gadi.py: warmup rule 'needs_warmup = model_name in (\"M2\", \"M3\") or ...' not found once.")
    t = t.replace(w_old, w_old.replace('("M2", "M3")', '("M2", "M3", "M5")'))
    wm = re.search(r'^( *)min_epochs = MIN_EPOCHS_QUANTUM_FRONTEND if needs_warmup else 0\n', t, re.M)
    if not wm:
        die("train_gadi.py: 'min_epochs = MIN_EPOCHS_QUANTUM_FRONTEND if needs_warmup else 0' not found.")
    t = (t[:wm.end()] + f'{wm.group(1)}print(f"[V135] warmup: model={{model_name}} min_epochs={{min_epochs}}", flush=True)  {MARK}\n'
         + t[wm.end():])
    print("  warmup rule: needs_warmup now includes M5; per-run '[V135] warmup:' log line added")

    # 3. wrapper block above the __main__ guard ------------------------------
    g = re.search(r"^if __name__ == [\"']__main__[\"']:", t, re.M)
    t = t[:g.start()] + BLOCK.lstrip("\n") + "\n" + t[g.start():]

    bak = tp.with_name(tp.name + ".v13bak")
    if not bak.exists():
        shutil.copy2(tp, bak)
    tp.write_text(t)
    print(f"Patched {tp.name} in {root}  (backup {bak.name})")
    print("Next: python3 scripts/verify_v135_m5.py   (must end in ALL CHECKS PASSED)")


if __name__ == "__main__":
    main()
