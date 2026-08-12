#!/usr/bin/env python
"""
smoke10_mnist.py — 10-class MNIST DIAGNOSTIC across M1-M4.

READ THIS BEFORE READING ANY NUMBER IT PRINTS
=============================================
This is a *diagnostic*, not a thesis experiment. It exists to answer one
question for the supervisor meeting: what happens to the four architectures
when the task has 10 classes instead of 2?

It is NOT comparable to the MNIST 3-vs-5 or Fashion-MNIST 0,6 grids, for three
reasons that must be stated whenever these numbers are shown:

1. THE ISO-PARAMETRIC BOUND IS BROKEN. The locked spec holds all four models
   within 0.93% of 1,600 parameters. Ten classes forces the output layers to
   widen, and they widen by different amounts:

       M1/M2 narrow : fc2(64,2)=130 + qnn=6 + fc3(1,10)=20   ->  156 back-end
       M1/M2 wide   : fc2(64,2)=130 + qnn=6 + fc3(3,10)=40   ->  176 back-end
       M3/M4        : fc2(64,10)=650                          ->  650 back-end

   M3/M4 therefore carry ~500 MORE parameters than M1/M2. Any M3/M4 advantage
   here is confounded by capacity and cannot be attributed to the head being
   classical. This is the opposite confound direction from the binary grids,
   where all four were matched.

2. THE QNN HEAD IS ARCHITECTURALLY BOTTLENECKED. In the locked design the
   back-end squeezes the whole network through fc2(64 -> 2), a 2-qubit circuit,
   and a SINGLE expectation value <ZZ> in [-1,1]. fc3 then maps that one scalar
   to the class logits. For 2 classes a threshold on one scalar is sufficient.
   For 10 classes it is not: fc3(1,10) can only produce logits that are affine
   in one number, so the predicted class is determined by where that single
   scalar falls on the real line. That is a 1-D ordering of 10 classes.

   Two head widths are therefore run:
     --head narrow : the faithful extension (1 observable). Expected to fail.
     --head wide   : 3 observables <Z0>, <Z1>, <ZZ> -> fc3(3,10). Still passes
                     through fc2's 2-D bottleneck, so also expected to be
                     limited, but less severely.

   A LOW M1/M2 SCORE HERE IS EVIDENCE ABOUT THE BOTTLENECK, NOT ABOUT QUANTUM
   LAYERS. Do not report it as "quantum models are worse at multiclass".

3. ONE SEED. The training protocol otherwise MATCHES the binary grids exactly:
   5000/1250/1000 split, batch 32, lr 1e-3, weight decay 1e-4, dropout 0.0,
   max 100 epochs, patience 10, min_epochs 30 warm-up. The only reduction is
   that this runs a single seed (42) rather than five, so no error bars and no
   significance testing. Every result is tagged REDUCED_SCALE.

   An earlier draft of this script used a flat 15 epochs with no early
   stopping. That was withdrawn: M1/M2 with the wide head were still improving
   at epoch 15 (val 0.254 -> 0.489, no plateau), so the fixed budget understated
   them and biased the comparison AGAINST the quantum models. The narrow head
   was genuinely saturated by epoch 2 either way.

The front-ends, the CNN backbone, and the quantum simulation modules
(quanv_torch, qnn_torch) are the SAME verified code used by the real grids.
Only the class-count-dependent output layers differ.

Usage
-----
    python smoke10_mnist.py --head narrow
    python smoke10_mnist.py --head wide
    python smoke10_mnist.py --head both        # default
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quanv_torch import TrainableQuanvFrontEndTorch, FrozenHendersonFrontEndTorch
from qnn_torch import _fixed_gates, _ry_layer, N_BE_PARAMS

N_QUBITS_FE = 4


# ─────────────────────────────────────────────────────────────────────
# Front-end M4 (copied verbatim from train_gadi.py)
# ─────────────────────────────────────────────────────────────────────
class ClassicalResizeFrontEnd(nn.Module):
    def forward(self, x):
        r = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        return r.repeat(1, N_QUBITS_FE, 1, 1)

    def n_trainable_params(self):
        return 0


# ─────────────────────────────────────────────────────────────────────
# Backbone (identical to train_gadi.py ClassicalBackbone, dropout=0)
# ─────────────────────────────────────────────────────────────────────
class ClassicalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.0)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop(x)
        return torch.flatten(x, 1)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ─────────────────────────────────────────────────────────────────────
# Multi-observable QNN. Same circuit as qnn_torch.TorchQNNBackend; the only
# difference is how many observables are read off the final state.
# ─────────────────────────────────────────────────────────────────────
class MultiObsQNN(nn.Module):
    """ZZFeatureMap(2,reps=1) + RealAmplitudes(2,reps=2), n_obs outputs.

    n_obs=1 -> <ZZ> only, byte-equivalent to the verified qnn_torch back-end.
    n_obs=3 -> <Z0>, <Z1>, <ZZ>. Adds NO parameters: these are three different
    measurements of the same 6-parameter state, not a bigger circuit.
    """

    def __init__(self, seed=None, n_obs=1):
        super().__init__()
        g = torch.Generator().manual_seed(seed if seed is not None else 0)
        w = (torch.rand(N_BE_PARAMS, generator=g, dtype=torch.float32) * 2 - 1) * np.pi
        self.weight = nn.Parameter(w)
        self.n_obs = n_obs

    def forward(self, x):
        B = x.shape[0]
        dev = x.device
        cdt = torch.complex128 if x.dtype == torch.float64 else torch.complex64
        H0, H1, CX01 = _fixed_gates(cdt, dev)
        x0, x1 = x[:, 0], x[:, 1]

        psi = torch.zeros(B, 4, dtype=cdt, device=dev)
        psi[:, 0] = 1.0
        psi = psi @ H0.transpose(0, 1)
        psi = psi @ H1.transpose(0, 1)

        ph = torch.stack([torch.zeros_like(x0), 2.0 * x0, 2.0 * x1,
                          2.0 * x0 + 2.0 * x1], dim=-1)
        psi = psi * torch.exp(1j * ph.to(cdt))
        psi = psi @ CX01.transpose(0, 1)

        zz = 2.0 * (np.pi - x0) * (np.pi - x1)
        ph2 = torch.stack([torch.zeros_like(zz), torch.zeros_like(zz), zz, zz], dim=-1)
        psi = psi * torch.exp(1j * ph2.to(cdt))
        psi = psi @ CX01.transpose(0, 1)

        w = self.weight
        psi = psi @ _ry_layer(w[0], w[1], cdt, dev).transpose(0, 1)
        psi = psi @ CX01.transpose(0, 1)
        psi = psi @ _ry_layer(w[2], w[3], cdt, dev).transpose(0, 1)
        psi = psi @ CX01.transpose(0, 1)
        psi = psi @ _ry_layer(w[4], w[5], cdt, dev).transpose(0, 1)

        p = (psi.conj() * psi).real                       # (B,4) basis probs
        # basis order: index bit0=q0, bit1=q1 -> |00>,|01>,|10>,|11>
        zz_o = torch.tensor([1., -1., -1., 1.], dtype=p.dtype, device=dev)
        if self.n_obs == 1:
            return (p @ zz_o).reshape(B, 1).to(x.dtype)
        z0_o = torch.tensor([1., -1., 1., -1.], dtype=p.dtype, device=dev)
        z1_o = torch.tensor([1., 1., -1., -1.], dtype=p.dtype, device=dev)
        return torch.stack([p @ z0_o, p @ z1_o, p @ zz_o], dim=-1).to(x.dtype)


class QNNHead(nn.Module):
    def __init__(self, seed, n_classes, n_obs):
        super().__init__()
        self.fc2 = nn.Linear(64, 2)
        self.qnn = MultiObsQNN(seed, n_obs=n_obs)
        self.fc3 = nn.Linear(n_obs, n_classes)

    def forward(self, x):
        return self.fc3(self.qnn(self.fc2(x)))

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ClassicalHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x):
        return self.fc2(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class QCQCNN(nn.Module):
    def __init__(self, front_end, back_end):
        super().__init__()
        self.front_end = front_end
        self.backbone = ClassicalBackbone()
        self.back_end = back_end

    def forward(self, x):
        return self.back_end(self.backbone(self.front_end(x)))

    def total_params(self):
        fe = self.front_end.n_trainable_params()
        bb = self.backbone.n_params()
        be = self.back_end.n_params()
        return fe + bb + be, (fe, bb, be)


def build_model(name, seed, n_classes, n_obs):
    if name == "M1":
        return QCQCNN(FrozenHendersonFrontEndTorch(seed), QNNHead(seed, n_classes, n_obs))
    if name == "M2":
        return QCQCNN(TrainableQuanvFrontEndTorch(seed=seed), QNNHead(seed, n_classes, n_obs))
    if name == "M3":
        return QCQCNN(TrainableQuanvFrontEndTorch(seed=seed), ClassicalHead(n_classes))
    if name == "M4":
        return QCQCNN(ClassicalResizeFrontEnd(), ClassicalHead(n_classes))
    raise ValueError(name)


# ─────────────────────────────────────────────────────────────────────
# Data — offline, all 10 MNIST classes, balanced
# ─────────────────────────────────────────────────────────────────────
def load_mnist10(data_root, n_train, n_val, n_test, seed):
    from torchvision import datasets
    try:
        tr = datasets.MNIST(data_root, train=True, download=False)
        te = datasets.MNIST(data_root, train=False, download=False)
    except Exception as exc:
        raise SystemExit(
            f"Could not open MNIST offline ({exc}).\n"
            f"Run:  python scripts/preflight_data.py --data_root {data_root}")

    def balanced(images, labels, n_per):
        xs, ys = [], []
        for c in range(10):
            idx = np.where(labels == c)[0][:n_per]
            xs.append(images[idx]); ys.append(np.full(len(idx), c))
        return np.concatenate(xs), np.concatenate(ys)

    pool_x, pool_y = balanced(tr.data.numpy(), tr.targets.numpy(), (n_train + n_val) // 10)
    test_x, test_y = balanced(te.data.numpy(), te.targets.numpy(), n_test // 10)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(pool_x))
    pool_x, pool_y = pool_x[perm], pool_y[perm]

    def to_t(a, b):
        x = torch.tensor(a, dtype=torch.float32).unsqueeze(1) / 255.0
        return torch.utils.data.TensorDataset(x, torch.tensor(b, dtype=torch.long))

    return (to_t(pool_x[:n_train], pool_y[:n_train]),
            to_t(pool_x[n_train:n_train + n_val], pool_y[n_train:n_train + n_val]),
            to_t(test_x, test_y))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    npred = set()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).argmax(1)
        npred.update(pred.unique().tolist())
        correct += (pred == yb).sum().item(); total += yb.numel()
    return correct / total, len(npred)


def run_one(name, seed, args, n_obs, device, data):
    train_ds, val_ds, test_ds = data
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_model(name, seed, 10, n_obs).to(device)
    total, (fe, bb, be) = model.total_params()

    tl = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    vl = torch.utils.data.DataLoader(val_ds, batch_size=256)
    sl = torch.utils.data.DataLoader(test_ds, batch_size=256)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_val, best_state, hist = -1.0, None, []
    # PROTOCOL MATCH: identical early stopping to the binary grids --
    # max_epochs=100, patience=10, and a min_epochs=30 warm-up before the
    # patience counter starts. The counter is gated by min_epochs as well as
    # the stop check; incrementing it from epoch 1 was defect 04 in the MNIST
    # phase and would terminate every run at the degenerate early state.
    bad, stop_ep = 0, args.epochs
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        c = n = 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward(); opt.step()
            c += (out.argmax(1) == yb).sum().item(); n += yb.numel()
        tr_acc = c / n
        va, nclass_v = evaluate(model, vl, device)
        hist.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va,
                     "n_classes_predicted": nclass_v})
        improved = va > best_val
        if improved:
            best_val = va
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"  [{name}] ep {ep:3d}  train={tr_acc:.4f}  val={va:.4f}  "
              f"distinct_preds={nclass_v}/10", flush=True)

        if ep >= args.min_epochs:
            bad = 0 if improved else bad + 1
            if bad >= args.patience:
                stop_ep = ep
                print(f"  [{name}] early stopping at epoch {ep} "
                      f"(best val_acc={best_val:.4f})", flush=True)
                break

    model.load_state_dict(best_state)
    test_acc, nclass_t = evaluate(model, sl, device)
    dt = time.time() - t0
    print(f"  [{name}] TEST={test_acc:.4f}  distinct_preds={nclass_t}/10  "
          f"params={total} (fe={fe} bb={bb} be={be})  {dt:.1f}s\n", flush=True)
    return {"model": name, "seed": seed, "n_obs": n_obs, "test_acc": test_acc,
            "best_val_acc": best_val, "distinct_test_preds": nclass_t,
            "params_total": total, "params_front": fe, "params_backbone": bb,
            "params_backend": be, "history": hist, "seconds": dt,
            "stop_epoch": stop_ep, "protocol": {
                "max_epochs": args.epochs, "patience": args.patience,
                "min_epochs": args.min_epochs, "batch_size": args.batch_size,
                "lr": args.lr, "weight_decay": 1e-4, "dropout": 0.0,
                "n_train": args.n_train, "n_val": args.n_val, "n_test": args.n_test},
            "REDUCED_SCALE": True, "ISO_PARAMETRIC": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="./data")
    ap.add_argument("--head", choices=["narrow", "wide", "both"], default="both")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100,
                    help="max epochs (grid protocol: 100)")
    ap.add_argument("--patience", type=int, default=10,
                    help="early-stopping patience (grid protocol: 10)")
    ap.add_argument("--min_epochs", type=int, default=30,
                    help="warm-up before patience counting (grid protocol: 30)")
    ap.add_argument("--n_train", type=int, default=5000)
    ap.add_argument("--n_val", type=int, default=1250)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="results_smoke10")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 74)
    print("10-CLASS MNIST DIAGNOSTIC  —  REDUCED SCALE, NOT ISO-PARAMETRIC")
    print("Not comparable to the binary grids. See module docstring before")
    print("quoting any number. M1/M2 are bottleneck-limited by construction.")
    print("=" * 74)
    print(f"device={device} seed={args.seed} epochs={args.epochs} "
          f"train={args.n_train} val={args.n_val} test={args.n_test}")
    print(f"chance level = 10.00%\n")

    data = load_mnist10(args.data_root, args.n_train, args.n_val, args.n_test, args.seed)
    os.makedirs(args.out, exist_ok=True)

    heads = ["narrow", "wide"] if args.head == "both" else [args.head]
    results = []
    for h in heads:
        n_obs = 1 if h == "narrow" else 3
        print("-" * 74)
        print(f"QNN head = {h}  ({n_obs} observable{'s' if n_obs > 1 else ''} "
              f"-> fc3({n_obs},10)).  M3/M4 are unaffected by this setting.")
        print("-" * 74)
        for name in ["M1", "M2", "M3", "M4"]:
            if h == "wide" and name in ("M3", "M4"):
                continue          # identical to the narrow pass; don't re-run
            r = run_one(name, args.seed, args, n_obs, device, data)
            r["head"] = h
            results.append(r)
            with open(f"{args.out}/smoke10_{name}_{h}_seed{args.seed}.json", "w") as f:
                json.dump(r, f, indent=2)

    print("=" * 74)
    print(f"{'model':6} {'head':8} {'test acc':>9} {'distinct':>9} {'params':>8}")
    print("-" * 74)
    for r in results:
        print(f"{r['model']:6} {r['head']:8} {r['test_acc']*100:8.2f}% "
              f"{r['distinct_test_preds']:6}/10 {r['params_total']:8}")
    print("=" * 74)
    print("chance = 10.00%.  'distinct' = how many of the 10 classes the model")
    print("ever predicts. A low count means the head cannot separate the")
    print("classes, which is a capacity/bottleneck fact, not a quantum result.")


if __name__ == "__main__":
    main()
