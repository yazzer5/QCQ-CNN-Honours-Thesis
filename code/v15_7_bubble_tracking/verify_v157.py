#!/usr/bin/env python3
"""verify_v157.py <repo157> <repo156>
V1 trainer minus the two V157 blocks == v156 trainer (bytes); V2 compiles; V3 private RNG only;
V4 REAL models M3/M4/M5: a simulated training loop (probe forward, 4 epochs of AdamW steps, per-epoch eval)
   gives bit-identical parameters and global RNG state with the tracker on vs off; records at epochs 0-3;
   sigma=0 sanity; save() writes the JSON."""
import ast
import contextlib
import io
import json
import os
import random
import sys
import tempfile

FAIL = []


def check(n, ok, d=""):
    print("%-4s %-66s %s" % ("OK" if ok else "FAIL", n, d), flush=True)
    if not ok:
        FAIL.append(n)


def cut(s, b, e):
    i, j = s.find(b), s.find(e)
    return s if i < 0 or j < 0 else s[:i] + s[j + len(e):]


def main():
    r7, r6 = sys.argv[1], sys.argv[2]
    s7 = open(os.path.join(r7, "train_gadi.py")).read()
    s6 = open(os.path.join(r6, "train_gadi.py")).read()
    st = cut(cut(s7, "\n\n# === V157_TRACK begin", "# === V157_TRACK end ==="),
             "        # === V157_SAVE begin ===\n", "        # === V157_SAVE end ===\n")
    check("V1  v157 trainer minus V157 blocks == v156 trainer (bytes)", st == s6)
    try:
        compile(s7, "train_gadi.py", "exec")
        check("V2  compiles", True)
    except SyntaxError as e:
        check("V2  compiles", False, str(e))
    tree = ast.parse(open(os.path.join(r7, "track_v157.py")).read())
    bad = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
           and n.func.attr in ("rand", "randn", "randint", "randperm", "normal")
           and not any(k.arg == "generator" for k in n.keywords)]
    bad += [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Attribute) and n.attr == "random"]
    check("V3  track_v157 uses private RNGs only", not bad, str(bad))
    check("V3b V157_TRACK_ON marker present in repo", os.path.exists(os.path.join(r7, "V157_TRACK_ON")))

    sys.path.insert(0, r7)
    import numpy as np
    import torch
    with contextlib.redirect_stdout(io.StringIO()):
        import train_gadi as tg
    check("V4a import real train_gadi (v157) with wrapper active", hasattr(tg, "_v157_build_model_orig"))
    if not hasattr(tg, "_v157_build_model_orig"):
        return finish()
    for nm, v in (("N_CLASSES", 5), ("N_OBS", 15), ("HEAD_VARIANT", "linear")):
        if hasattr(tg, nm):
            setattr(tg, nm, v)
    g = torch.Generator().manual_seed(7)
    X = (torch.rand(300, 1, 28, 28, generator=g) > 0.8).float()
    y = torch.arange(5).repeat_interleave(60)
    tg._V155["test"] = torch.utils.data.TensorDataset(X, y)
    Xtr = (torch.rand(256, 1, 28, 28, generator=g) > 0.8).float()
    ytr = torch.randint(0, 5, (256,), generator=g)

    def train(model):
        torch.manual_seed(0)
        np.random.seed(0)
        random.seed(0)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, weight_decay=1e-4)
        model.train()
        model(Xtr[:32]).sum().backward()          # gradient-flow probe
        opt.zero_grad()
        for ep in range(4):
            model.train()
            for b in range(0, 256, 32):
                opt.zero_grad()
                torch.nn.functional.cross_entropy(model(Xtr[b:b + 32]), ytr[b:b + 32]).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                model(X[:64])
        return [p.detach().clone() for p in model.parameters()], torch.get_rng_state(), np.random.get_state()[1].copy(), \
            random.getstate()

    for M in ("M3", "M4", "M5"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                a = tg.build_model(M, 42)
                b = tg._v157_build_model_orig(M, 42)
            pa, ra, na, qa = train(a)
            pb, rb, nb, qb = train(b)
            same = all(torch.equal(u, v) for u, v in zip(pa, pb))
            check("V4b %s params after training identical, tracker on vs off" % M, same)
            check("V4c %s global torch/numpy/python RNG identical on vs off" % M,
                  torch.equal(ra, rb) and (na == nb).all() and qa == qb)
            rec = a._v157.records
            check("V4d %s records at epochs 0,1,2,3; no tracker error" % M,
                  [r["epoch"] for r in rec] == [0, 1, 2, 3] and a._v157.err is None,
                  "%s err=%s" % ([r["epoch"] for r in rec], a._v157.err))
            r0 = rec[-1]
            check("V4e %s sigma=0 sanity (eta=0, trSN=0) and finite radius at z" % M,
                  r0["L2|0"]["eta"] == 0 and r0["L2|0"]["trSN"] == 0 and np.isfinite(r0["L2|0.3"]["radius"]))
            tmp = tempfile.mkdtemp()
            with contextlib.redirect_stdout(io.StringIO()) as buf:
                import track_v157
                track_v157.save(a, tmp, M, 42, 0.0)
            j = json.load(open(os.path.join(tmp, "track_%s_seed42.json" % M)))
            check("V4f %s save(): JSON written, TRACK OK printed" % M,
                  j["meta"]["n_records"] == 4 and "[V157] TRACK OK" in buf.getvalue())
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            check("V4  %s" % M, False, "%s: %s" % (type(e).__name__, e))
    return finish()


def finish():
    print("=" * 80)
    if FAIL:
        print("VERIFY FAILED: %s" % ", ".join(FAIL))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
