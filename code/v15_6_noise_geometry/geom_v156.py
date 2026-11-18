"""v15.6 geometry pass (PREREGISTRATION_v15_6.md, A1, A2).

Called from inside train_gadi._v155_xeval, after the v15.5 cross-eval and before the RNG restore,
on the model that passed selfcheck=MATCH. Measures, at five layers:

  Lpix  noisy/clean image (784)          L0  front-end output (4x14x14 = 784)
  L1    conv backbone output (64)        L2  z = back_end.fc2 output (2)
  L3    logits (K)

for test sigma in SIGMAS, with R noise draws on a fixed 50-per-class subset:

  rho_W = tr S_N / tr S_W          rho_B = tr S_N / tr S_B
  eta   = mean_i ||mean_r f(x_i+n_r) - f(x_i)|| / sqrt(2 tr S_B)
  J     = tr(S_W^-1 S_B) (L2, full)  and  J_tr = tr S_B / tr S_W (all layers)
          retention = J(noisy subset points) / J(clean subset points)       [A2]
  bleed = fraction of noisy points nearer (Mahalanobis, shrunk S_W) to another class's clean mean;
          at L3, the argmax error.

Never touches the global torch / numpy / python RNGs: only private torch.Generator objects.
Python-3.6 compatible.
"""
import hashlib
import json
import time

import numpy as np
import torch

VERSION = "v156-geom-1"
SUBSET_SEED = 20261104
PER_CLASS = 50
R = 20
SIGMAS = (0.0, 0.1, 0.2, 0.3, 0.5)
FRESH_BASE = 15600000          # A1: draw r>=1 seed = FRESH_BASE + 1000*r + round(1000*sigma)
LAM_HI = 0.1                   # shrinkage for Mahalanobis at Lpix, L0, L1
LAYERS = ("Lpix", "L0", "L1", "L2", "L3")
METRICS = ("trSW", "trSB", "trSN", "rho_W", "rho_B", "eta", "Jtr_clean", "Jtr_noisy", "Jtr_ret",
           "J_clean", "J_noisy", "J_ret", "bleed")


def _code(s):
    return int(round(1000 * float(s)))


def sha(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def select_subset(y, per_class=PER_CLASS, seed=SUBSET_SEED):
    """Fixed per-class subset; depends only on the label vector and the seed."""
    y = np.asarray(y)
    g = torch.Generator().manual_seed(int(seed))
    idx = []
    for c in np.unique(y):
        ii = np.where(y == c)[0]
        if len(ii) < per_class:
            raise ValueError("class %s has %d < %d test images" % (c, len(ii), per_class))
        p = torch.randperm(len(ii), generator=g).numpy()[:per_class]
        idx.extend(ii[p].tolist())
    return np.array(sorted(idx), dtype=np.int64)


def fresh_draws(Xs, sigma, r_lo=1, r_hi=R):
    """Draws r = r_lo..r_hi-1 on the subset images Xs (CPU), each from its own private generator."""
    out = []
    for r in range(r_lo, r_hi):
        g = torch.Generator().manual_seed(FRESH_BASE + 1000 * r + _code(sigma))
        out.append((Xs + float(sigma) * torch.randn(Xs.shape, generator=g, dtype=Xs.dtype)).clamp(0.0, 1.0))
    return out


def extract(model, X, bs):
    """Features at all five layers for inputs X (CPU tensor, N x 1 x 28 x 28), batched exactly like
    _v155_predict (contiguous batches of size bs, eval mode, no_grad)."""
    dev = next(model.parameters()).device
    cap = {}

    def hook(name):
        def f(_m, _i, o):
            cap[name] = o
        return f

    fc2 = getattr(model.back_end, "fc2", None)
    if fc2 is None:
        raise AttributeError("model.back_end has no fc2")
    hooks = [model.front_end.register_forward_hook(hook("L0")),
             model.backbone.register_forward_hook(hook("L1")),
             fc2.register_forward_hook(hook("L2"))]
    out = dict((k, []) for k in LAYERS)
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            for s in range(0, X.shape[0], bs):
                xb = X[s:s + bs]
                cap.clear()
                lg = model(xb.to(dev))
                n = xb.shape[0]
                out["Lpix"].append(xb.reshape(n, -1).double().cpu().numpy())
                for k in ("L0", "L1", "L2"):
                    out[k].append(cap[k].reshape(n, -1).double().cpu().numpy())
                out["L3"].append(lg.reshape(n, -1).double().cpu().numpy())
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return dict((k, np.concatenate(v, 0)) for k, v in out.items())


def scatter(F, y, classes):
    """Class means, mean of per-class covariances (ddof=1), between-class scatter (class-balanced)."""
    mus = np.stack([F[y == c].mean(0) for c in classes])
    d = F.shape[1]
    SW = np.zeros((d, d))
    for k, c in enumerate(classes):
        D = F[y == c] - mus[k]
        SW += D.T @ D / max(D.shape[0] - 1, 1)
    SW /= len(classes)
    dm = mus - mus.mean(0)
    SB = dm.T @ dm / len(classes)
    return mus, SW, SB


def fisher_full(SW, SB):
    try:
        return float(np.trace(np.linalg.solve(SW, SB)))
    except np.linalg.LinAlgError:
        return float("nan")


def whitener(SW, lam):
    d = SW.shape[0]
    S = (1.0 - lam) * SW + lam * (np.trace(SW) / d) * np.eye(d)
    L = np.linalg.cholesky(S)
    return np.linalg.inv(L)            # A with A S A^T = I


def nearest_mean(F, mus, A):
    Fw = F @ A.T
    Mw = mus @ A.T
    d2 = (Fw ** 2).sum(1)[:, None] - 2 * Fw @ Mw.T + (Mw ** 2).sum(1)[None, :]
    return d2.argmin(1)


def layer_metrics(Fc_full, y_full, Fc_sub, Fn, y_sub, classes, layer):
    """Fc_full: clean full test (N x d). Fc_sub: clean subset (n x d). Fn: noisy subset (R x n x d)."""
    mus, SW, SB = scatter(Fc_full, y_full, classes)
    trSW, trSB = float(np.trace(SW)), float(np.trace(SB))
    Rn = Fn.shape[0]
    fbar = Fn.mean(0)
    trSN = float(((Fn - fbar[None]) ** 2).sum(axis=(0, 2)).mean() / (Rn - 1)) if Rn > 1 else 0.0
    D = np.sqrt(2.0 * trSB)
    eta_i = np.sqrt(((fbar - Fc_sub) ** 2).sum(1)) / D
    Fpool = Fn.reshape(-1, Fn.shape[2])
    ypool = np.tile(y_sub, Rn)
    _, SWc, SBc = scatter(Fc_sub, y_sub, classes)
    _, SWn, SBn = scatter(Fpool, ypool, classes)
    m = dict(trSW=trSW, trSB=trSB, trSN=trSN, rho_W=trSN / trSW, rho_B=trSN / trSB,
             eta=float(eta_i.mean()),
             Jtr_clean=float(np.trace(SBc) / np.trace(SWc)), Jtr_noisy=float(np.trace(SBn) / np.trace(SWn)))
    m["Jtr_ret"] = m["Jtr_noisy"] / m["Jtr_clean"]
    if layer == "L2":
        m["J_clean"], m["J_noisy"] = fisher_full(SWc, SBc), fisher_full(SWn, SBn)
        m["J_ret"] = m["J_noisy"] / m["J_clean"]
        m["J0_full"] = fisher_full(SW, SB)
    else:
        m["J_clean"] = m["J_noisy"] = m["J_ret"] = m["J0_full"] = float("nan")
    if layer == "L3":
        pred = Fpool.argmax(1)
    else:
        pred = nearest_mean(Fpool, mus, whitener(SW, 0.0 if layer == "L2" else LAM_HI))
    m["bleed"] = float((pred != ypool).mean())
    return m, eta_i


def run(model, Xte, res, sigma_train, M, S, out_dir, fixed_fn, bs=32, sigmas=SIGMAS):
    """Entry point from train_gadi. res is the v15.5 xeval dict (labels, predictions)."""
    import os
    t0 = time.time()
    y_full = np.asarray([int(v) for v in res["labels"]], dtype=np.int64)
    classes = np.unique(y_full)
    K = len(classes)
    sub = select_subset(y_full)
    y_sub = y_full[sub]
    Xte = Xte.detach().cpu()
    Xs = Xte[torch.from_numpy(sub)]

    clean = extract(model, Xte, bs)
    pred_clean = clean["L3"].argmax(1)
    saved0 = res.get("predictions", {}).get("g0")
    clean_mismatch = int((pred_clean != np.asarray(saved0)).sum()) if saved0 is not None else -1

    metrics = {}
    eta_img = np.zeros((len(sigmas), len(LAYERS), len(sub)), dtype=np.float32)
    zn = np.zeros((len(sigmas), R, len(sub), 2), dtype=np.float32)
    pn = np.zeros((len(sigmas), R, len(sub)), dtype=np.int16)
    a1, nh = {}, {}
    for si, s in enumerate(sigmas):
        s = float(s)
        if s == 0.0:
            Fn = dict((k, clean[k][sub][None]) for k in LAYERS)
            nh["%g" % s] = sha(Xs.numpy())
            a1["%g" % s] = clean_mismatch
        else:
            full0 = fixed_fn(Xte, s)                       # A1: draw 0 = v15.5's own noisy test set
            f0 = extract(model, full0, bs)
            draws = fresh_draws(Xs, s)
            nh["%g" % s] = sha(full0[torch.from_numpy(sub)].numpy(), *[d.numpy() for d in draws])
            fr = extract(model, torch.cat(draws, 0), bs)
            n = len(sub)
            Fn = dict((k, np.concatenate([f0[k][sub][None], fr[k].reshape(R - 1, n, -1)], 0)) for k in LAYERS)
            sv = res.get("predictions", {}).get("g%g" % s)
            a1["%g" % s] = int((f0["L3"].argmax(1)[sub] != np.asarray(sv)[sub]).sum()) if sv is not None else -1
        Rn = Fn["L2"].shape[0]
        zn[si, :Rn] = Fn["L2"]
        pn[si, :Rn] = Fn["L3"].argmax(2)
        for li, L in enumerate(LAYERS):
            m, e = layer_metrics(clean[L], y_full, clean[L][sub], Fn[L], y_sub, classes, L)
            metrics["%s|%g" % (L, s)] = m
            eta_img[si, li] = e

    be = model.back_end
    W3, b3 = be.fc3.weight.detach().double().cpu().numpy(), be.fc3.bias.detach().double().cpu().numpy()
    Wm, bm = be.fc_mid.weight.detach().double().cpu().numpy(), be.fc_mid.bias.detach().double().cpu().numpy()
    fe_state = dict(("fe__" + k.replace(".", "__"), v.detach().cpu().numpy())
                    for k, v in model.front_end.state_dict().items())
    meta = dict(version=VERSION, model=M, seed=int(S), sigma_train=float(sigma_train), K=int(K),
                classes=[int(c) for c in classes], n_test=int(len(y_full)), n_sub=int(len(sub)), R=R,
                sigmas=[float(s) for s in sigmas], subset_hash=sha(sub, y_sub), test_hash=sha(Xte.numpy(), y_full),
                noise_hash=nh, a1_mismatch=a1, clean_mismatch=clean_mismatch, bs=int(bs),
                seconds=round(time.time() - t0, 1))
    out = os.path.join(out_dir, "geom_%s_seed%s.npz" % (M, S))
    np.savez_compressed(out, meta=json.dumps(meta), metrics=json.dumps(metrics), eta_img=eta_img,
                        z_clean=clean["L2"].astype(np.float32), y_full=y_full, sub=sub, y_sub=y_sub,
                        z_noisy=zn, pred_noisy=pn, W_eff=W3 @ Wm, b_eff=W3 @ bm + b3, **fe_state)
    ok = clean_mismatch == 0 and all(v == 0 for v in a1.values())
    print("[V156] GEOM %s model=%s seed=%s sigma_train=%g K=%d n_sub=%d subset_hash=%s a1_mismatch=%s "
          "clean_mismatch=%d time=%.1fs file=%s" % ("OK" if ok else "CHECKFAIL", M, S, float(sigma_train), K,
                                                   len(sub), meta["subset_hash"], json.dumps(a1),
                                                   clean_mismatch, meta["seconds"], os.path.basename(out)),
          flush=True)
    return meta
