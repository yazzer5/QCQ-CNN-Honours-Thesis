"""v15.7: record the noise bubble during training (PREREGISTRATION_v15_7.md).

install(model, ctx) adds a forward pre-hook. A new training epoch is detected when the model runs a
training-mode forward after having run an eval-mode forward (the trainer's per-epoch validation).
At the start of epoch k (k-1 completed epochs = c) with c in SCHEDULE, it measures the bubble on the
v15.6 test subset (no_grad, eval mode, private RNGs) and restores train mode. Never changes training.
"""
import json
import time

import numpy as np
import torch

import geom_v156 as G

R = 10
SIG = (0.0, 0.3)
EARLY = (0, 1, 2, 3, 5, 7, 10)
LAYERS = ("L0", "L1", "L2", "L3")


def scheduled(c):
    return c in EARLY or (c >= 10 and c % 5 == 0)


class _State(object):
    def __init__(self, ctx):
        self.ctx = ctx
        self.busy = False
        self.last_train = None
        self.starts = 0          # epoch starts seen
        self.records = []
        self.cache = None
        self.err = None
        self.seconds = 0.0


def _prepare(st):
    te = st.ctx["get_test"]()
    if te is None:
        raise RuntimeError("test split not captured yet")
    X, ys = st.ctx["stack"](te)
    y = np.asarray([int(v) for v in (ys.tolist() if hasattr(ys, "tolist") else ys)], dtype=np.int64)
    X = X.detach().cpu()
    sub = G.select_subset(y)
    Xs = X[torch.from_numpy(sub)]
    full0 = st.ctx["fixed"](X, 0.3)
    fresh = torch.cat(G.fresh_draws(Xs, 0.3, 1, R), 0)
    st.cache = dict(X=X, y=y, sub=sub, full0=full0, fresh=fresh, classes=np.unique(y))


def measure(model, st, c):
    t0 = time.time()
    if st.cache is None:
        _prepare(st)
    k = st.cache
    bs = st.ctx["bs"]
    clean = G.extract(model, k["X"], bs)          # extract() sets eval/no_grad and restores mode
    f0 = G.extract(model, k["full0"], bs)
    fr = G.extract(model, k["fresh"], bs)
    n = len(k["sub"])
    rec = dict(epoch=int(c), starts=int(st.starts), time=round(time.time(), 1))
    y_sub = k["y"][k["sub"]]
    for s in SIG:
        for L in LAYERS:
            if s == 0.0:
                Fn = clean[L][k["sub"]][None]
            else:
                Fn = np.concatenate([f0[L][k["sub"]][None], fr[L].reshape(R - 1, n, -1)], 0)
            m, _ = G.layer_metrics(clean[L], k["y"], clean[L][k["sub"]], Fn, y_sub, k["classes"], L)
            m["radius"] = float(np.sqrt(((Fn - clean[L][k["sub"]][None]) ** 2).sum(2).mean() / (2 * m["trSB"])))
            rec["%s|%g" % (L, s)] = m
    st.seconds += time.time() - t0
    st.records.append(rec)


def install(model, ctx):
    st = _State(ctx)
    model._v157 = st

    def pre(mod, _inp):
        if st.busy:
            return None
        if not mod.training:
            st.last_train = False
            return None
        if st.last_train is not True:          # first training forward, or first after an eval period
            st.starts += 1
            c = st.starts - 1
            if scheduled(c) and st.err is None:
                st.busy = True
                try:
                    measure(mod, st, c)
                except Exception as e:  # noqa: BLE001
                    st.err = "%s: %s" % (type(e).__name__, e)
                    print("[V157] TRACK FAILED at epoch %d: %s" % (c, st.err), flush=True)
                finally:
                    st.busy = False
                    mod.train(True)
        st.last_train = True
        return None

    model.register_forward_pre_hook(pre)
    return model


def save(model, out_dir, M, S, sigma_train):
    import os
    st = getattr(model, "_v157", None)
    if st is None:
        print("[V157] TRACK FAILED: model has no tracker", flush=True)
        return
    meta = dict(model=M, seed=int(S), sigma_train=float(sigma_train), R=R, epoch_starts=st.starts,
                n_records=len(st.records), seconds=round(st.seconds, 1), error=st.err,
                subset_hash=G.sha(st.cache["sub"], st.cache["y"][st.cache["sub"]]) if st.cache else None)
    out = os.path.join(out_dir, "track_%s_seed%s.json" % (M, S))
    json.dump(dict(meta=meta, records=st.records), open(out, "w"))
    print("[V157] TRACK %s model=%s seed=%s records=%d epoch_starts=%d last_epoch=%s time=%.1fs file=%s"
          % ("OK" if st.err is None else "CHECKFAIL", M, S, len(st.records), st.starts,
             st.records[-1]["epoch"] if st.records else None, st.seconds, os.path.basename(out)), flush=True)
