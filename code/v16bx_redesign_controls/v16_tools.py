"""
v16_tools.py -- model surgery and z-capture used by the V16A block in train_gadi.py.
Kept separate from train_gadi.py so v16b can reuse it. Nothing here touches training.

  install_m2c(model, seed)  M2 -> M2c: replace the QNN module (qnn_torch.QNNBackendTorch, 6 params,
                            15 outputs) by ZZTrig12 (12 closed-form features, 0 params) and the
                            readout Linear(15 -> K) that consumes it by a fresh Linear(12 -> K).
                            Front-end, backbone and fc2 are untouched (same init as M2).
                            The new Linear is initialised under torch.manual_seed(seed) INSIDE a
                            forked RNG: every global RNG stream is left exactly where M2 leaves it,
                            so M2c sees the same batch order as M2 at the same seed.
  z_layer(model)            the unique nn.Linear with out_features == 2 (fc2). Its output is z,
                            the input of the QNN head (checked by verify_v16a for M2).
  predict_with_z(model, ds) test-set predictions, labels and z, in dataset order, eval mode.
"""
import hashlib
import random

import numpy as np
import torch
import torch.nn as nn

from closed_form_zz_v16 import N_FEATURES, ZZTrig12


class V16Error(RuntimeError):
    pass


class M2cFeatureHead(ZZTrig12):
    """ZZTrig12 as installed in M2c. Adds ONE shim, nothing else:

    train_gadi.train_model() (v5 diagnostic) reads `model.back_end.qnn.weight` before training
    and after every epoch, only to print `qnn_weight_drift`. M2c has no circuit, so the module
    in the QNN's slot carries a constant 6-element `weight` BUFFER: not a parameter (parameter
    count unchanged, never optimised), non-persistent (not in state_dict / checkpoints), and
    unused by forward(). The diagnostic then prints drift 0 for M2c. (v16a defect D1, 2026-10-21.)
    """

    def __init__(self, n_circuit_params=6):
        super().__init__()
        self.register_buffer("weight", torch.zeros(n_circuit_params), persistent=False)


def rng_save():
    st = dict(py=random.getstate(), np=np.random.get_state(), t=torch.get_rng_state())
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def rng_restore(st):
    random.setstate(st["py"]); np.random.set_state(st["np"]); torch.set_rng_state(st["t"])
    if "cuda" in st:
        torch.cuda.set_rng_state_all(st["cuda"])


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def _qnn_class():
    import qnn_torch
    cls = getattr(qnn_torch, "QNNBackendTorch", None)
    if cls is None:
        raise V16Error("qnn_torch.QNNBackendTorch not found")
    return cls


def find_qnn(model):
    cls = _qnn_class()
    return [(n, m) for n, m in model.named_modules() if isinstance(m, cls)]


def find_linear(model, in_f=None, out_f=None):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)
            and (in_f is None or m.in_features == in_f) and (out_f is None or m.out_features == out_f)]


def set_submodule(model, dotted, new):
    parent = model
    parts = dotted.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    if not hasattr(parent, parts[-1]):
        raise V16Error(f"cannot set {dotted}")
    setattr(parent, parts[-1], new)


def z_layer(model):
    lins = find_linear(model, out_f=2)
    if len(lins) != 1:
        raise V16Error(f"expected exactly one nn.Linear(-> 2) (fc2), found {[n for n, _ in lins]}")
    return lins[0]


def _probe_input(model, n=3, seed=1234):
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, 28, 28, generator=g).to(dev)


def wiring(model, src, dst, x):
    """Run model(x); return (output of module src, input of module dst, logits)."""
    cap = {}
    h1 = src.register_forward_hook(lambda m, i, o: cap.__setitem__("src", o))
    h2 = dst.register_forward_hook(lambda m, i, o: cap.__setitem__("dst", i[0]))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(x)
    finally:
        h1.remove(); h2.remove(); model.train(was)
    return cap.get("src"), cap.get("dst"), out


def install_m2c(model, seed):
    """In place. Returns a dict describing the surgery. Raises V16Error if the model is not M2-shaped."""
    st = rng_save()
    try:
        qs = find_qnn(model)
        if len(qs) != 1:
            raise V16Error(f"expected exactly one QNNBackendTorch, found {[n for n, _ in qs]}")
        qname, qnn = qs[0]
        n_q = sum(p.numel() for p in qnn.parameters())
        n_obs = int(getattr(qnn, "n_obs", 0))
        if n_obs != 15:
            raise V16Error(f"QNN n_obs = {n_obs}; M2c is defined at w15 only")
        x = _probe_input(model)
        qout = {}
        hq = qnn.register_forward_hook(lambda m, i, o: qout.update(i=i[0], o=o))
        was = model.training
        model.eval()
        with torch.no_grad():
            logits_m2 = model(x)
        model.train(was)
        hq.remove()
        K = logits_m2.shape[1]
        cands = find_linear(model, in_f=n_obs, out_f=K)
        if len(cands) != 1:
            raise V16Error(f"expected exactly one Linear({n_obs} -> {K}) after the QNN, found {[n for n, _ in cands]}")
        fname, fc3 = cands[0]
        src, dst, _ = wiring(model, qnn, fc3, x)
        if src is None or dst is None or src.shape != dst.shape or not torch.equal(src, dst):
            raise V16Error(f"{fname} does not read the QNN output directly (wiring is not identity)")
        zname, zl = z_layer(model)
        zo, qi, _ = wiring(model, zl, qnn, x)
        if not torch.equal(zo, qi):
            raise V16Error(f"QNN input is not the output of {zname} (z wiring is not identity)")
        before = n_params(model)
        dev, dt = fc3.weight.device, fc3.weight.dtype
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            new_fc = nn.Linear(N_FEATURES, K)
        new_fc = new_fc.to(device=dev, dtype=dt)
        feat = M2cFeatureHead(n_q).to(device=dev)
        set_submodule(model, qname, feat)
        set_submodule(model, fname, new_fc)
        after = n_params(model)
        # post-surgery: the closed-form module sees the same z, the new Linear reads its 12 outputs
        trig = dict(model.named_modules())[qname]
        src2, dst2, logits = wiring(model, trig, new_fc, x)
        if not (torch.equal(src2, dst2) and src2.shape[1] == N_FEATURES and logits.shape == logits_m2.shape):
            raise V16Error("post-surgery wiring check failed")
        expect_delta = -n_q - (n_obs + 1) * K + (N_FEATURES + 1) * K
        return dict(qnn=qname, readout=fname, z_layer=zname, K=K, n_obs=n_obs, qnn_params=n_q,
                    params_m2=before, params_m2c=after, delta_ok=(after - before == expect_delta))
    finally:
        rng_restore(st)


def predict_with_z(model, ds, bs=250, want_z=True):
    zname, zl = z_layer(model)
    dev = next(model.parameters()).device
    zs = []
    h = zl.register_forward_hook(lambda m, i, o: zs.append(o.detach().cpu())) if want_z else None
    was = model.training
    model.eval()
    preds, labels = [], []
    try:
        loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
        with torch.no_grad():
            for batch in loader:
                xb, yb = batch[0], batch[1]
                out = model(xb.to(dev))
                preds += out.argmax(1).cpu().tolist()
                labels += [int(v) for v in (yb.tolist() if hasattr(yb, "tolist") else yb)]
    finally:
        if h is not None:
            h.remove()
        model.train(was)
    z = torch.cat(zs).numpy() if want_z else None
    return preds, labels, z


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
