#!/usr/bin/env python3
"""verify_v156.py <repo156> <repo155>

V1  patched train_gadi.py minus the V156 block == the v15.5 train_gadi.py, byte for byte
V2  patched file compiles; block sits inside _v155_xeval, before _v155_rng_restore(st)
V3  geom_v156.py: every torch random call uses a private generator; no numpy/python RNG use
V4  metric maths on synthetic data (sigma=0 sanity, Fisher vs theory, Mahalanobis vs brute force,
    scale invariance)
V5  REAL models (train_gadi.build_model M3, M4, M5, K=5 and K=10): hook shapes 784/784/64/2/K,
    full geom run with the real _v155_fixed; A1 draw-0 predictions == _v155_predict; clean ==
    _v155_predict; global torch/numpy/python RNG state unchanged; bit-identical rerun
"""
import ast
import hashlib
import io
import json
import os
import random
import sys
import tempfile
import contextlib

FAIL = []


def check(name, ok, detail=""):
    print("%-4s %-62s %s" % ("OK" if ok else "FAIL", name, detail), flush=True)
    if not ok:
        FAIL.append(name)


def md5(b):
    return hashlib.md5(b).hexdigest()


def main():
    repo156, repo155 = sys.argv[1], sys.argv[2]
    p156 = os.path.join(repo156, "train_gadi.py")
    p155 = os.path.join(repo155, "train_gadi.py")
    src = open(p156).read()
    orig = open(p155, "rb").read()

    # V1
    b, e = src.find("        # === V156_GEOM begin"), src.find("        # === V156_GEOM end ===\n")
    check("V1a exactly one V156 block", src.count("V156_GEOM begin") == 1 and b >= 0 and e > b)
    stripped = (src[:b] + src[e + len("        # === V156_GEOM end ===\n"):]).encode()
    check("V1b patched minus block == v15.5 train_gadi.py (bytes)", stripped == orig,
          "md5 %s vs %s" % (md5(stripped), md5(orig)))

    # V2
    try:
        compile(src, p156, "exec")
        check("V2a patched train_gadi.py compiles", True)
    except SyntaxError as ex:
        check("V2a patched train_gadi.py compiles", False, str(ex))
    fdef = src.rfind("\ndef ", 0, b)
    check("V2b block inside _v155_xeval", src[fdef:].startswith("\ndef _v155_xeval("))
    check("V2c block directly before _v155_rng_restore(st)",
          src[e + len("        # === V156_GEOM end ===\n"):].startswith("        _v155_rng_restore(st)\n"))

    # V3
    gsrc = open(os.path.join(repo156, "geom_v156.py")).read()
    tree = ast.parse(gsrc)
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            nm = node.func.attr
            base = ast.dump(node.func.value)
            if nm in ("rand", "randn", "randint", "randperm", "normal", "bernoulli", "multinomial") \
                    and "torch" in base and not any(k.arg == "generator" for k in node.keywords):
                bad.append("%s line %d" % (nm, node.lineno))
            if "random" in base and "torch" not in base:
                bad.append("np/python random line %d" % node.lineno)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
            if "random" in names:
                bad.append("import random")
    check("V3  geom_v156 uses private generators only", not bad, "; ".join(bad))

    sys.path.insert(0, repo156)
    import numpy as np
    import torch
    import geom_v156 as G

    # V4 synthetic maths
    rs = np.random.RandomState(0)          # local RNG object, verify only
    K, n, d = 3, 100000, 2
    means = np.array([[0, 0], [3, 0], [0, 4.0]])
    F = np.concatenate([rs.randn(n, d) + means[k] for k in range(K)])
    y = np.repeat(np.arange(K), n)
    mus, SW, SB = G.scatter(F, y, np.arange(K))
    dm = means - means.mean(0)
    J_theory = np.trace(dm.T @ dm / K)
    check("V4a Fisher J vs theory (SW = I) and vs explicit inverse",
          abs(G.fisher_full(SW, SB) / J_theory - 1) < 0.02 and
          abs(G.fisher_full(SW, SB) - np.trace(np.linalg.inv(SW) @ SB)) < 1e-9,
          "%.3f vs %.3f" % (G.fisher_full(SW, SB), J_theory))
    A = G.whitener(SW, 0.0)
    Sinv = np.linalg.inv(SW)
    brute = np.array([[(f - m) @ Sinv @ (f - m) for m in mus] for f in F[:500]]).argmin(1)
    check("V4b whitened nearest mean == brute Mahalanobis", (G.nearest_mean(F[:500], mus, A) == brute).all())
    d = 20
    Fc = rs.rand(300, d)
    ys = np.repeat(np.arange(3), 100)
    Fn = Fc[None] + 0.1 * rs.randn(5, 300, d)
    m1, _ = G.layer_metrics(Fc, ys, Fc, Fn, ys, np.arange(3), "L0")
    m2, _ = G.layer_metrics(7.3 * Fc, ys, 7.3 * Fc, 7.3 * Fn, ys, np.arange(3), "L0")
    ks = ["rho_W", "rho_B", "eta", "Jtr_ret", "bleed"]
    check("V4c metrics invariant to rescaling a layer", all(abs(m1[k] - m2[k]) < 1e-9 for k in ks))
    m0, e0 = G.layer_metrics(Fc, ys, Fc, Fc[None], ys, np.arange(3), "L2" if d == 2 else "L0")
    check("V4d sigma=0: trSN = 0, eta = 0, Jtr_ret = 1",
          m0["trSN"] == 0 and m0["eta"] == 0 and abs(m0["Jtr_ret"] - 1) < 1e-12)

    # V5 real models
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            import train_gadi as tg
        check("V5a import real train_gadi (v156 copy)", True)
    except Exception as ex:  # noqa: BLE001
        check("V5a import real train_gadi (v156 copy)", False, "%s: %s" % (type(ex).__name__, ex))
        return finish()
    g = torch.Generator().manual_seed(123)
    for Kc in (5, 10):
        X = (torch.rand(60 * Kc, 1, 28, 28, generator=g) > 0.8).float()
        X[:, :, 10:12, :] = 0.5
        yv = torch.arange(Kc).repeat_interleave(60)
        for M in ("M3", "M4", "M5"):
            tag = "K%d %s" % (Kc, M)
            try:
                for nm, v in (("N_CLASSES", Kc), ("N_OBS", 15), ("HEAD_VARIANT", "linear")):
                    if hasattr(tg, nm):
                        setattr(tg, nm, v)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = tg.build_model(M, 42)
                f = G.extract(model, X[:40], 32)
                shp = [f[k].shape[1] for k in G.LAYERS]
                check("V5b %s hook dims Lpix/L0/L1/L2/L3" % tag, shp == [784, 784, 64, 2, Kc], str(shp))
                ds = torch.utils.data.TensorDataset(X, yv)
                p0, lab = tg._v155_predict(model, ds)
                res = dict(labels=lab, predictions={"g0": p0})
                for s in (0.1, 0.2, 0.3, 0.5):
                    Xn = tg._v155_fixed(X, "gauss", s, tg.V155_TEST_GAUSS_BASE)
                    res["predictions"]["g%g" % s] = tg._v155_predict(model, torch.utils.data.TensorDataset(Xn, yv))[0]
                st = (torch.get_rng_state(), np.random.get_state(), random.getstate())
                tmp = tempfile.mkdtemp()
                fx = (lambda X_, lv_: tg._v155_fixed(X_, "gauss", lv_, tg.V155_TEST_GAUSS_BASE))
                with contextlib.redirect_stdout(io.StringIO()) as buf:
                    meta = G.run(model, X, res, 0.0, M, 42, tmp, fx, bs=32)
                st2 = (torch.get_rng_state(), np.random.get_state(), random.getstate())
                same = torch.equal(st[0], st2[0]) and st[1][0] == st2[1][0] and (st[1][1] == st2[1][1]).all() \
                    and st[2] == st2[2]
                check("V5c %s global RNG untouched" % tag, same)
                check("V5d %s clean == _v155_predict, A1 draw0 == _v155_predict" % tag,
                      meta["clean_mismatch"] == 0 and all(v == 0 for v in meta["a1_mismatch"].values()),
                      json.dumps(meta["a1_mismatch"]))
                check("V5e %s prints GEOM OK" % tag, "[V156] GEOM OK" in buf.getvalue())
                z = np.load(os.path.join(tmp, "geom_%s_seed42.npz" % M))
                met = json.loads(str(z["metrics"]))
                l0 = met["L3|0"]
                pclean = np.asarray(p0)[z["sub"]]
                check("V5f %s sigma=0 sanity (trSN=0, eta=0, L3 bleed = clean error)" % tag,
                      all(met["%s|0" % L]["trSN"] == 0 and met["%s|0" % L]["eta"] == 0 for L in G.LAYERS)
                      and abs(l0["bleed"] - float((pclean != z["y_sub"]).mean())) < 1e-12)
                pn = z["pred_noisy"][3]
                check("V5g %s L3 bleed at 0.3 == argmax error of saved preds" % tag,
                      abs(met["L3|0.3"]["bleed"] - float((pn != z["y_sub"][None]).mean())) < 1e-12)
                logits = np.asarray(f["L2"]) @ z["W_eff"].T + z["b_eff"]
                check("V5h %s logits == W_eff z + b_eff (linear head)" % tag,
                      np.abs(logits - f["L3"]).max() < 1e-4, "max err %.2e" % np.abs(logits - f["L3"]).max())
                os.makedirs(os.path.join(tmp, "b"))
                with contextlib.redirect_stdout(io.StringIO()):
                    G.run(model, X, res, 0.0, M, 42, os.path.join(tmp, "b"), fx, 32)
                z2 = np.load(os.path.join(tmp, "b", "geom_%s_seed42.npz" % M))
                check("V5i %s rerun bit-identical" % tag, str(z["metrics"]) == str(z2["metrics"]) and
                      np.array_equal(z["z_noisy"], z2["z_noisy"]))
            except Exception as ex:  # noqa: BLE001
                import traceback
                traceback.print_exc()
                check("V5  %s real-model run" % tag, False, "%s: %s" % (type(ex).__name__, ex))
    return finish()


def finish():
    print("=" * 78)
    if FAIL:
        print("VERIFY FAILED: %d check(s): %s" % (len(FAIL), ", ".join(FAIL)))
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
