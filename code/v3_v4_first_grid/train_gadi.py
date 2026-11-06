#!/usr/bin/env python
"""
train_gadi.py — M1-M4 QCQ-CNN comparison training script for NCI Gadi

Thesis: "End-to-End Trainable Quantum-Classical-Quantum CNNs for Image
         Classification" — Yasser Arafat, UNSW ELEC4951-4953

════════════════════════════════════════════════════════════════════════
TRIPLE-CHECKED AGAINST architecture_report.html (source of truth)
════════════════════════════════════════════════════════════════════════

CIRCUIT 4 (Hur et al. 2022, arXiv:2108.00661, Fig. 2(d)) — front-end for M2 & M3
  RE-VERIFIED 2026-07-18 directly against the source figure (a prior session
  had the two middle single-qubit gates wrong as RZ; corrected to RY below).
  Applied identically (SAME 6 shared weights) to pair (q0,q1) and pair (q2,q3).
  Per pair (q_a, q_b):
      RY(theta0)  on q_a
      RY(theta1)  on q_b
      CRZ(theta2)  ctrl=q_b, tgt=q_a      <-- CRZ, NOT CNOT, NOT CZ
      RY(theta3)  on q_a                  <-- RY, NOT RZ (corrected 2026-07-18)
      RY(theta4)  on q_b                  <-- RY, NOT RZ (corrected 2026-07-18)
      CRZ(theta5)  ctrl=q_a, tgt=q_b      <-- CRZ, NOT CNOT, NOT CZ
  PennyLane convention wires=[ctrl, tgt]:
      CRZ #1 -> qml.CRZ(theta2, wires=[q_b, q_a])
      CRZ #2 -> qml.CRZ(theta5, wires=[q_a, q_b])
  6 trainable parameters total (theta0..theta5), shared across both pairs.
  Encoding (pixel -> RY(pixel * pi)) is FIXED, not trainable, identical for
  M1/M2/M3/M4 wherever a quantum front-end is used.

HENDERSON FRONT-END (M1 only) — frozen, 0 trainable params
  qml.RandomLayers with a fixed (non-trainable) random parameter tensor,
  same RY(pixel*pi) encoding, 4 qubits, applied per 2x2 patch.

M4 FRONT-END — classical, 0 trainable params
  Bilinear resize 28x28 -> 14x14, replicated to 4 channels (4,14,14),
  matching the (4,14,14) tensor shape the shared CNN backbone expects.
  No quantum operation at all for M4.

CNN BACKBONE (shared, identical for all 4 models) — 1,464 params
  Conv2d(4,8,kernel=3,no padding) -> ReLU -> MaxPool2d(2)
  Conv2d(8,16,kernel=3,no padding) -> ReLU -> MaxPool2d(2) -> Dropout2d
  Spatial: 4x14x14 -> 8x12x12 -> 8x6x6 -> 16x4x4 -> 16x2x2 -> flatten(64)
  Params: Conv1 = 4*8*9+8 = 296 ; Conv2 = 8*16*9+16 = 1,168 ; total = 1,464

QNN BACKEND (M1, M2 only) — 140 params
  fc2: Linear(64,2)                       -> 130 params  (64*2+2)
  ZZFeatureMap(2)  (data encoding, NOT trainable)
  RealAmplitudes(num_qubits=2, reps=2)    ->   6 params  (reps=2, NOT reps=1
                                                — reps=1 was an identified bug)
  fc3: Linear(1,2)                        ->   4 params  (1*2+2)
  Total = 130 + 6 + 4 = 140

CLASSICAL HEAD (M3, M4 only) — 130 params
  fc2: Linear(64,2) -> 130 params, straight to class logits.

VERIFIED PARAMETER TOTALS (assert-checked at runtime, see build_model()):
  M1 = 0   + 1,464 + 140 = 1,604   (replication anchor, exempt from iso-match)
  M2 = 6   + 1,464 + 140 = 1,610   (P_M2 -- iso-parametric reference) [STAR]
  M3 = 6   + 1,464 + 130 = 1,600   (delta -10, 0.62%)
  M4 = 0   + 1,464 + 130 = 1,594   (delta -16, 0.994% -- tolerance edge, disclosed)

TRAINING PROTOCOL (Long et al. faithful, deviations flagged)
  Optimiser        : Adam, lr = 0.001
  Loss              : BCE (2-class, one-hot)
  Max epochs        : 100
  Patience          : 10 (validation accuracy), post min_epochs warmup
  min_epochs        : 30 for M2/M3 ONLY (trainable quantum front-end plateau) [DEVIATION]
  Batch size        : 1 for M1/M2/M3 (parameter-shift rule requires it -- MANDATORY)
                        32 for M4 (classical only)
  Split             : 5,000 / 1,250 / 1,000 (train/val/test)
  Datasets          : mnist (3 vs 5), fashion_mnist (T-shirt/top vs Trouser),
                        mri (tumour vs no-tumour)
  Seeds             : {42,43,44,45,46} -- 5 seeds  [DEVIATION from Long's 10]

USAGE
  python train_gadi.py --model M2 --dataset mnist --seed 42 \
                        --noise_alpha 0.0 --output_dir ./results --data_root ./data
"""

import os
import sys
import json
import time
import math
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.qnn import TorchLayer

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
N_C4_PARAMS   = 6      # Circuit 4 shared params (theta0..theta5)
N_QUBITS_FE   = 4      # front-end qubits (one per 2x2-patch pixel)
N_QUBITS_BE   = 2      # QNN backend qubits

SEEDS      = [42, 43, 44, 45, 46]
MAX_EPOCHS = 100
PATIENCE   = 10
MIN_EPOCHS_QUANTUM_FRONTEND = 30   # M2/M3 only
LR         = 1e-3
N_TRAIN, N_VAL, N_TEST = 5_000, 1_250, 1_000

EXPECTED_PARAMS = {"M1": 1604, "M2": 1610, "M3": 1600, "M4": 1594}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# CIRCUIT 4 BLOCK — verified gate-by-gate against architecture_report.html
# ═══════════════════════════════════════════════════════════════════════
def circuit4_block(weights, wire_a, wire_b):
    """Hur et al. (2022) Circuit 4. 6 params. CRZ entanglers (NOT CNOT/CZ).
    Re-verified 2026-07-18 directly against Hur et al. Fig. 2(d): the two
    un-controlled single-qubit gates in the middle column are RY, NOT RZ
    (a prior session had this wrong as RZ -- corrected here)."""
    qml.RY(weights[0], wires=wire_a)
    qml.RY(weights[1], wires=wire_b)
    qml.CRZ(weights[2], wires=[wire_b, wire_a])   # ctrl=wire_b, tgt=wire_a
    qml.RY(weights[3], wires=wire_a)              # FIXED: was RZ
    qml.RY(weights[4], wires=wire_b)              # FIXED: was RZ
    qml.CRZ(weights[5], wires=[wire_a, wire_b])   # ctrl=wire_a, tgt=wire_b


def build_trainable_quanv_qnode(dev):
    @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
    def qnode(inputs, weights):
        # Fixed angle encoding: RY(pixel * pi) on each of the 4 qubits
        for i in range(N_QUBITS_FE):
            qml.RY(inputs[i] * np.pi, wires=i)
        # Circuit 4, SAME 6 weights applied to both pairs
        circuit4_block(weights, 0, 1)
        circuit4_block(weights, 2, 3)
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_FE)]
    return qnode


def build_frozen_henderson_qnode(dev, rand_params):
    @qml.qnode(dev, interface="torch", diff_method=None)
    def qnode(inputs):
        for i in range(N_QUBITS_FE):
            qml.RY(inputs[i] * np.pi, wires=i)
        qml.RandomLayers(rand_params, wires=list(range(N_QUBITS_FE)))
        return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS_FE)]
    return qnode


# ═══════════════════════════════════════════════════════════════════════
# FRONT-ENDS
# ═══════════════════════════════════════════════════════════════════════
class TrainableQuanvFrontEnd(nn.Module):
    """M2 / M3 front-end: trainable Circuit 4, 6 shared params, 0 leakage
    across patches (weights reused identically for every 2x2 patch)."""

    def __init__(self):
        super().__init__()
        dev = qml.device("lightning.qubit", wires=N_QUBITS_FE)
        qnode = build_trainable_quanv_qnode(dev)
        self.qlayer = TorchLayer(qnode, {"weights": (N_C4_PARAMS,)})

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 4, 14, 14)
        B, _, H, W = x.shape
        oh, ow = H // 2, W // 2
        out = torch.zeros(B, N_QUBITS_FE, oh, ow, dtype=x.dtype, device=x.device)
        for i in range(oh):
            for j in range(ow):
                patch = x[:, 0, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(B, 4)
                res = torch.stack([self.qlayer(patch[b]) for b in range(B)], dim=0)
                out[:, :, i, j] = res
        return out

    def n_trainable_params(self):
        return N_C4_PARAMS


class FrozenHendersonFrontEnd(nn.Module):
    """M1 front-end: frozen random circuit (Henderson-style quanvolution).
    0 trainable params — rand_params is a registered buffer, not a Parameter."""

    def __init__(self, seed):
        super().__init__()
        dev = qml.device("default.qubit", wires=N_QUBITS_FE)
        g = np.random.RandomState(seed)
        rand_params = qnp.array(
            g.uniform(high=2 * np.pi, size=(1, N_QUBITS_FE)), requires_grad=False
        )
        self.qnode = build_frozen_henderson_qnode(dev, rand_params)

    def forward(self, x):
        B, _, H, W = x.shape
        oh, ow = H // 2, W // 2
        out = torch.zeros(B, N_QUBITS_FE, oh, ow, dtype=x.dtype, device=x.device)
        for i in range(oh):
            for j in range(ow):
                patch = x[:, 0, 2 * i:2 * i + 2, 2 * j:2 * j + 2].reshape(B, 4)
                for b in range(B):
                    vals = self.qnode(patch[b].detach().cpu().numpy())
                    out[b, :, i, j] = torch.tensor(vals, dtype=x.dtype, device=x.device)
        return out

    def n_trainable_params(self):
        return 0


class ClassicalResizeFrontEnd(nn.Module):
    """M4 front-end: classical bilinear resize 28x28 -> 14x14, replicated to
    4 channels. 0 trainable params, no quantum operation."""

    def forward(self, x):
        # x: (B, 1, 28, 28) -> (B, 1, 14, 14) -> (B, 4, 14, 14)
        resized = F.interpolate(x, size=(14, 14), mode="bilinear", align_corners=False)
        return resized.repeat(1, N_QUBITS_FE, 1, 1)

    def n_trainable_params(self):
        return 0


# ═══════════════════════════════════════════════════════════════════════
# SHARED CNN BACKBONE — identical across all 4 models, 1,464 params
# ═══════════════════════════════════════════════════════════════════════
class ClassicalBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3)     # no padding
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)    # no padding
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (B,8,6,6)
        x = self.pool(F.relu(self.conv2(x)))   # (B,16,2,2)
        x = self.drop(x)
        return x.flatten(1)                    # (B,64)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# QNN BACKEND (M1, M2) — ZZFeatureMap(2) + RealAmplitudes(2, reps=2)
# ═══════════════════════════════════════════════════════════════════════
def build_qnn_backend():
    feature_map = ZZFeatureMap(feature_dimension=N_QUBITS_BE, reps=1)
    ansatz = RealAmplitudes(num_qubits=N_QUBITS_BE, reps=2)   # reps=2, NOT reps=1
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
    )
    assert len(ansatz.parameters) == 6, f"RealAmplitudes(2,reps=2) should have 6 params, got {len(ansatz.parameters)}"
    return TorchConnector(qnn)


class QNNBackend(nn.Module):
    """M1 / M2 back-end. fc2(64->2) -> QNN(2->1) -> fc3(1->2)."""

    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(64, 2)
        self.qnn = build_qnn_backend()
        self.fc3 = nn.Linear(1, 2)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.fc2(x)
        x = self.qnn(x)
        # TorchConnector output shape can vary (B,1) or (B,) depending on
        # version/batch-size edge cases — normalise to (B,1) before fc3.
        x = x.reshape(batch_size, -1)
        x = self.fc3(x)
        return x

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ClassicalHead(nn.Module):
    """M3 / M4 back-end. fc2(64->2) straight to logits."""

    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        return self.fc2(x)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════════════════
# FULL MODELS M1-M4
# ═══════════════════════════════════════════════════════════════════════
class QCQCNN(nn.Module):
    """Generic wrapper: front_end -> backbone -> back_end."""

    def __init__(self, front_end, back_end):
        super().__init__()
        self.front_end = front_end
        self.backbone = ClassicalBackbone()
        self.back_end = back_end

    def forward(self, x):
        x = self.front_end(x)
        x = self.backbone(x)
        x = self.back_end(x)
        return x

    def total_params(self):
        fe = self.front_end.n_trainable_params()
        bb = self.backbone.n_params()
        be = self.back_end.n_params()
        return fe + bb + be, (fe, bb, be)


def build_model(model_name, seed):
    if model_name == "M1":
        m = QCQCNN(FrozenHendersonFrontEnd(seed), QNNBackend())
    elif model_name == "M2":
        m = QCQCNN(TrainableQuanvFrontEnd(), QNNBackend())
    elif model_name == "M3":
        m = QCQCNN(TrainableQuanvFrontEnd(), ClassicalHead())
    elif model_name == "M4":
        m = QCQCNN(ClassicalResizeFrontEnd(), ClassicalHead())
    else:
        raise ValueError(f"Unknown model {model_name}")

    total, (fe, bb, be) = m.total_params()
    expected = EXPECTED_PARAMS[model_name]
    assert total == expected, (
        f"[PARAM COUNT MISMATCH] {model_name}: got {total} "
        f"(front={fe}, backbone={bb}, back_end={be}), expected {expected}. "
        f"ABORTING — do not train with an unverified architecture."
    )
    print(f"[VERIFIED] {model_name} params: front={fe} backbone={bb} "
          f"back_end={be} total={total} (expected {expected}) OK", flush=True)
    return m


# ═══════════════════════════════════════════════════════════════════════
# DATASETS
# ═══════════════════════════════════════════════════════════════════════
def _balanced_subset(images, labels, classes, n_per_class):
    sel_i, sel_l = [], []
    for idx, c in enumerate(classes):
        mask = np.where(labels == c)[0][:n_per_class]
        sel_i.append(images[mask])
        sel_l.append(np.full(len(mask), idx))
    return np.concatenate(sel_i), np.concatenate(sel_l)


def load_dataset(name, data_root, seed):
    """Returns (train_ds, val_ds, test_ds) each yielding (1,28,28) float tensors
    in [0,1] and integer labels in {0,1}."""
    rng = np.random.RandomState(seed)

    if name == "mnist":
        classes = (3, 5)
        train_raw = datasets.MNIST(data_root, train=True, download=True)
        test_raw = datasets.MNIST(data_root, train=False, download=True)
    elif name == "fashion_mnist":
        classes = (0, 1)   # 0=T-shirt/top, 1=Trouser
        train_raw = datasets.FashionMNIST(data_root, train=True, download=True)
        test_raw = datasets.FashionMNIST(data_root, train=False, download=True)
    elif name == "mri":
        # NOTE: MRI brain-tumour dataset path is project-specific.
        # Point --data_root at a directory with `mri_images.npy` (N,28,28 uint8)
        # and `mri_labels.npy` (N,) with 0=no-tumour, 1=tumour (subtypes collapsed).
        mri_dir = Path(data_root) / "mri"
        imgs = np.load(mri_dir / "mri_images.npy")
        labs = np.load(mri_dir / "mri_labels.npy")
        n_needed = N_TRAIN + N_VAL + N_TEST
        idx = rng.permutation(len(imgs))[:n_needed]
        imgs, labs = imgs[idx], labs[idx]
        tr_i, va_i, te_i = imgs[:N_TRAIN], imgs[N_TRAIN:N_TRAIN+N_VAL], imgs[N_TRAIN+N_VAL:n_needed]
        tr_l, va_l, te_l = labs[:N_TRAIN], labs[N_TRAIN:N_TRAIN+N_VAL], labs[N_TRAIN+N_VAL:n_needed]
        return (_TensorImgDS(tr_i, tr_l), _TensorImgDS(va_i, va_l), _TensorImgDS(te_i, te_l))
    else:
        raise ValueError(f"Unknown dataset {name}")

    train_imgs = train_raw.data.numpy()
    train_labs = train_raw.targets.numpy()
    test_imgs = test_raw.data.numpy()
    test_labs = test_raw.targets.numpy()

    # Build train+val pool from the train split, held-out test from the test split
    pool_imgs, pool_labs = _balanced_subset(train_imgs, train_labs, classes, (N_TRAIN + N_VAL) // 2)
    te_imgs, te_labs = _balanced_subset(test_imgs, test_labs, classes, N_TEST // 2)

    perm = rng.permutation(len(pool_imgs))
    pool_imgs, pool_labs = pool_imgs[perm], pool_labs[perm]
    tr_imgs, tr_labs = pool_imgs[:N_TRAIN], pool_labs[:N_TRAIN]
    va_imgs, va_labs = pool_imgs[N_TRAIN:N_TRAIN + N_VAL], pool_labs[N_TRAIN:N_TRAIN + N_VAL]

    return (_TensorImgDS(tr_imgs, tr_labs), _TensorImgDS(va_imgs, va_labs), _TensorImgDS(te_imgs, te_labs))


class _TensorImgDS(Dataset):
    def __init__(self, imgs, labs):
        self.imgs = imgs.astype(np.float32) / 255.0 if imgs.max() > 1.0 else imgs.astype(np.float32)
        self.labs = labs.astype(np.int64)

    def __len__(self):
        return len(self.labs)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.imgs[idx]).unsqueeze(0)   # (1,28,28)
        if img.shape[-1] != 28:
            img = F.interpolate(img.unsqueeze(0), size=(28, 28), mode="bilinear", align_corners=False).squeeze(0)
        return img, self.labs[idx]


# ═══════════════════════════════════════════════════════════════════════
# TRAIN / EVAL LOOP
# ═══════════════════════════════════════════════════════════════════════
def run_epoch(model, loader, optimizer, device, train=True):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_onehot = F.one_hot(y, num_classes=2).float()
        if train:
            optimizer.zero_grad()
        logits = model(x)
        probs = torch.sigmoid(logits)
        loss = F.binary_cross_entropy(probs, y_onehot)
        if train:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def save_checkpoint(path, epoch, model, optimizer, best_val_acc, best_state, patience_ctr, history):
    """Atomic-ish checkpoint write: write to a temp file then rename, so a job
    killed mid-write (e.g. hard walltime kill) can't leave a corrupt checkpoint
    that a resumed job would then fail to load."""
    tmp_path = str(path) + ".tmp"
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_acc": best_val_acc,
        "best_state": best_state,
        "patience_ctr": patience_ctr,
        "history": history,
    }, tmp_path)
    os.replace(tmp_path, path)


def load_checkpoint(path, model, optimizer, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def train_model(model, train_ds, val_ds, test_ds, model_name, device,
                 checkpoint_path=None, time_budget_seconds=None):
    """Trains model_name. If checkpoint_path is given and a checkpoint already
    exists there, resumes from it (continuing the SAME early-stopping state --
    epoch counter, best_val_acc, best_state, patience_ctr, history -- rather
    than restarting). If time_budget_seconds is given and is exceeded mid-run,
    saves a checkpoint and returns {"status": "requeue", ...} instead of
    finishing, so main() can exit with a distinct code for the PBS wrapper to
    detect and resubmit a continuation job.

    NOTE: dataloader shuffle order is NOT bit-exactly reproduced across a
    resume (each resumed epoch reshuffles freshly). This is a deliberate
    simplification -- seeds still control weight init and the data split
    deterministically, and exact per-epoch shuffle order does not affect the
    scientific validity of the result, only exact bit-reproducibility of a
    single unbroken run, which the walltime kills already made impossible."""
    batch_size = 1 if model_name in ("M1", "M2", "M3") else 32
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    min_epochs = MIN_EPOCHS_QUANTUM_FRONTEND if model_name in ("M2", "M3") else 0

    start_epoch = 1
    best_val_acc, best_state, patience_ctr = -1.0, None, 0
    history = []

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        print(f"[{model_name}] Resuming from checkpoint: {checkpoint_path}", flush=True)
        ckpt = load_checkpoint(checkpoint_path, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        best_val_acc = ckpt["best_val_acc"]
        best_state = ckpt["best_state"]
        patience_ctr = ckpt["patience_ctr"]
        history = ckpt["history"]
        print(f"[{model_name}] Resumed at epoch {start_epoch} "
              f"(best_val_acc so far={best_val_acc:.4f}, patience_ctr={patience_ctr}, "
              f"{len(history)} epochs of history loaded)", flush=True)

    job_start = time.time()
    finished = False

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                         "val_loss": va_loss, "val_acc": va_acc, "time_sec": dt})
        print(f"[{model_name}] epoch {epoch:3d}  train_acc={tr_acc:.4f} "
              f"val_acc={va_acc:.4f} ({dt:.1f}s)", flush=True)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if checkpoint_path is not None:
            save_checkpoint(checkpoint_path, epoch, model, optimizer,
                             best_val_acc, best_state, patience_ctr, history)

        if epoch >= min_epochs and patience_ctr >= PATIENCE:
            print(f"[{model_name}] early stopping at epoch {epoch} "
                  f"(best val_acc={best_val_acc:.4f})", flush=True)
            finished = True
            break

        if time_budget_seconds is not None and (time.time() - job_start) >= time_budget_seconds:
            elapsed_h = (time.time() - job_start) / 3600
            print(f"[{model_name}] TIME BUDGET REACHED at epoch {epoch} "
                  f"(this job ran {elapsed_h:.2f}h). Checkpoint saved -- "
                  f"requeueing to continue.", flush=True)
            return {"status": "requeue", "epoch": epoch}
    else:
        finished = True   # loop completed all MAX_EPOCHS without early stopping or requeue

    if best_state is not None:
        model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, test_loader, optimizer, device, train=False)
    print(f"[{model_name}] FINAL test_acc={te_acc:.4f}", flush=True)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)   # done -- no resumable state needed anymore

    return {"status": "done", "history": history, "best_val_acc": best_val_acc,
            "test_acc": te_acc, "test_loss": te_loss}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["M1", "M2", "M3", "M4"])
    ap.add_argument("--dataset", required=True, choices=["mnist", "fashion_mnist", "mri"])
    ap.add_argument("--seed", type=int, required=True, choices=SEEDS)
    ap.add_argument("--noise_alpha", type=float, default=0.0)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--checkpoint_dir", default=None,
                     help="Dir for resumable checkpoints, keyed by "
                          "(model,dataset,seed,alpha). If a checkpoint already "
                          "exists for this exact combo, training resumes from "
                          "it instead of restarting from epoch 1. Omit to "
                          "disable checkpointing entirely (old behaviour).")
    ap.add_argument("--time_budget_hours", type=float, default=None,
                     help="Wall-clock budget for THIS process only (not "
                          "cumulative across a chain). If training hasn't "
                          "converged by then, saves a checkpoint and exits "
                          "with code 99 so a PBS wrapper can requeue a "
                          "continuation job. Omit to disable (old behaviour, "
                          "process runs until convergence or MAX_EPOCHS "
                          "regardless of walltime).")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Run: model={args.model} dataset={args.dataset} seed={args.seed} "
          f"noise_alpha={args.noise_alpha}", flush=True)

    if args.noise_alpha != 0.0:
        print("[WARNING] noise_alpha != 0.0 requested but IBM-calibrated noise "
              "injection is not wired into this Phase-1 script — running IDEAL "
              "(noiseless) simulation regardless. Implement Aer noise model "
              "before running the noise sweep phase.", flush=True)

    model = build_model(args.model, args.seed).to(device)
    train_ds, val_ds, test_ds = load_dataset(args.dataset, args.data_root, args.seed)
    print(f"Split sizes: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}", flush=True)

    checkpoint_path = None
    if args.checkpoint_dir is not None:
        ckpt_dir = Path(args.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = ckpt_dir / (
            f"ckpt_{args.model}_{args.dataset}_seed{args.seed}_alpha{args.noise_alpha}.pt"
        )

    time_budget_seconds = (
        args.time_budget_hours * 3600 if args.time_budget_hours is not None else None
    )

    result = train_model(
        model, train_ds, val_ds, test_ds, args.model, device,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        time_budget_seconds=time_budget_seconds,
    )

    if result.get("status") == "requeue":
        print(f"[{args.model}] Not converged within this job's time budget "
              f"(stopped after epoch {result['epoch']}). Checkpoint saved at "
              f"{checkpoint_path}. Exiting 99 for PBS wrapper to requeue.", flush=True)
        sys.exit(99)

    result.update({
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
        "noise_alpha": args.noise_alpha,
        "n_params": EXPECTED_PARAMS[args.model],
    })

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"result_{args.model}_{args.dataset}_seed{args.seed}_alpha{args.noise_alpha}.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved -> {out_file}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
