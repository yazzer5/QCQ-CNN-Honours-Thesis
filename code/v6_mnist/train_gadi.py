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
  Conv2d(8,16,kernel=3,no padding) -> ReLU -> MaxPool2d(2) -> Dropout2d(0.0)
                                              [rate 0.0 as of 2026-07-28, was 0.5 -- see DROPOUT_RATE]
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
  min_epochs        : 30 for M1/M2/M3 -- any model with a trainable quantum
                        front-end OR back-end (widened 2026-07-28, was
                        M2/M3-only; M1's trainable QNN back-end has the same
                        documented slow-start plateau, see Long et al. Sec
                        4.3.2 and MODELS_SOURCE_OF_TRUTH.md Sec 9) [DEVIATION]
  Batch size        : 32 for ALL FOUR models (changed 2026-07-28, see below)
  Split             : 5,000 / 1,250 / 1,000 (train/val/test)
  Datasets          : mnist (3 vs 5), fashion_mnist (T-shirt/top vs Trouser),
                        mri (tumour vs no-tumour)
  Seeds             : {42,43,44,45,46} -- 5 seeds  [DEVIATION from Long's 10]

USAGE
  python train_gadi.py --model M2 --dataset mnist --seed 42 \
                        --noise_alpha 0.0 --output_dir ./results --data_root ./data

GPU ROOT CAUSE + FIX — 2026-07-28  (v5)
  Every prior attempt to get GPU utilisation up failed because the problem
  was never device selection. The front-end circuits are 4 qubits (16
  amplitudes); no state-vector backend can occupy a V100 at that size --
  cuStateVec pays ~5-10us of launch overhead per call for a few dozen FLOPs
  of work, so lightning.gpu is actually SLOWER than lightning.qubit below
  ~18 qubits and reports ~0% utilisation. On top of that, PennyLane's
  TorchLayer never turned the v4 unfold() batch into a single device call:
  the lightning devices don't support parameter broadcasting, so PennyLane
  applied broadcast_expand and re-serialised it into B*196 sequential tape
  executions. The v4 "batching fix" therefore changed nothing.

  v5 fix: the front-end circuits are computed directly as batched torch
  tensor algebra (quanv_torch.py) instead of being handed to a simulator.

    * M2/M3: the RY encoding is a product state and circuit4_block(0,1) /
      circuit4_block(2,3) share no wires, so the 4-qubit state stays an
      exact tensor product -- the front-end is TWO independent 2-qubit
      (dim-4) simulations, i.e. a batched 4x4 complex matmul.
      Verified to 8.9e-16 against a brute-force 4-qubit reference.
    * M1: RandomLayers is not separable, so the 16x16 unitary is extracted
      ONCE via qml.matrix() at construction (identical circuit semantics)
      and applied as a batched matmul thereafter. M1 is frozen, so nothing
      is approximated.
    * Gradients come from torch autograd and are NUMERICALLY IDENTICAL to
      the analytic parameter-shift result (verified ~1e-15). Note CRZ needs
      the FOUR-term shift rule, not the two-term one -- its generator has
      eigenvalues {0,0,-1/2,+1/2}. That also means the old cost model was
      wrong: Circuit 4 needs 16 tape evaluations per backward pass, not 12.

  Consequences: the whole model now lives on cuda, and the batch_size=1
  constraint is gone -- there is no parameter-shift tape machinery left in
  the training path for batched autograd to interact with. Batch size is
  32 for all four models (see MODELS_SOURCE_OF_TRUTH.md Sec 6 for the
  reasoning and the required bs=1 vs bs=32 control run).

  Parameter-shift remains REQUIRED for the later noisy/hardware phase,
  where it is a genuine hardware-faithfulness constraint rather than a
  simulation choice. Run with --backend pennylane to restore the old path
  and assert equivalence (scripts/equivalence_check.py).

  NOTE: M1/M2's QNN BACK-END (build_qnn_backend(), qiskit EstimatorQNN)
  still runs on CPU -- the default qiskit Estimator has no GPU code path.
  Out of scope here; tied to the later IBM-noise phase.
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

from quanv_torch import (
    TrainableQuanvFrontEndTorch,
    FrozenHendersonFrontEndTorch,
)
from qnn_torch import QNNBackendTorch

# PennyLane is imported lazily -- it is only needed to build M1's frozen
# unitary once (qml.matrix) and for the optional --backend pennylane
# reference path. Nothing in the hot training loop touches it.

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
MIN_EPOCHS_QUANTUM_FRONTEND = 30   # M1, M2, M3 -- any model with a trainable
                                    # quantum front-end OR back-end (widened
                                    # 2026-07-28, was M2/M3-only; see Sec 9)
LR         = 1e-3

# WEIGHT DECAY (2026-07-28, v5): Thesis A Sec 5.3 commits to weight decay
# "applied uniformly across all four models, so that capacity-utilisation is
# comparable rather than one model being effectively larger than another".
# It was never implemented -- the optimiser was plain Adam(lr=1e-3). Added
# here to satisfy that commitment.
#
# MEASURED EFFECT: essentially none at this model size. Sweep on M3, MNIST,
# seed 42, n_train=1000 (test set n=200):
#     wd=0.0   train 1.0000  val 0.9720  test 0.9550
#     wd=1e-3  train 1.0000  val 0.9720  test 0.9900
#     wd=1e-2  train 1.0000  val 0.9760  test 0.9650
#     wd=5e-2  train 1.0000  val 0.9760  test 0.9900
#     wd=1e-1  train 1.0000  val 0.9800  test 0.9700
# train_acc stays pinned at 1.0000 at EVERY value including 1e-1, so decay
# is not preventing memorisation at any tested strength. val improves by
# ~2 samples out of 250 across three orders of magnitude. test accuracy is
# non-monotonic -- that is noise at n=200, not signal. Do not present any
# of this as tuning.
#
# Why it matters less than Thesis A Sec 5.3 anticipated: that section's
# concern was that M4 must be PADDED with extra classical capacity to reach
# M2's parameter count, creating an overfitting confound. In this
# implementation M4 is 1,594 vs M2's 1,610 -- it was never meaningfully
# padded, so the confound the mitigation was designed for is largely absent.
# Also note the train/val gap should SHRINK at full scale, not grow: 1,600
# parameters against 5,000 samples is more underparameterised than against
# 1,000.
#
# 1e-4 is kept as a standard, harmless default that honours the protocol
# commitment. It is NOT claimed to improve results. Re-check at full scale
# before reporting anything about regularisation.
#
# Uses AdamW rather than Adam's `weight_decay` argument: Adam's version
# couples the penalty to the adaptive learning rate, so effective decay
# differs per parameter, which would break the "uniform across all models"
# requirement. AdamW decouples them (Loshchilov & Hutter 2019).
#
# NOTE: switching Adam -> AdamW changes the optimiser state_dict format.
# Any checkpoint written before 2026-07-28 cannot be resumed; start fresh.
WEIGHT_DECAY = 1e-4
N_TRAIN, N_VAL, N_TEST = 5_000, 1_250, 1_000

EXPECTED_PARAMS = {"M1": 1604, "M2": 1610, "M3": 1600, "M4": 1594}

# BATCH SIZE (2026-07-28, v5): 32 for ALL FOUR models.
# Previously 1 for M1-M3 / 128 for M4. The bs=1 constraint was tied to
# PennyLane's parameter-shift tape execution, which no longer exists in the
# training path. 32 gives 156 optimiser steps/epoch at n_train=5000 --
# enough for 1,600 params, retains useful gradient noise against the
# barren-plateau risk, and transfers lr=1e-3 from Long et al.'s bs=4
# without retuning. Matched across all four models so batch size is not a
# second uncontrolled variable inside the M3-vs-M4 comparison.
# See MODELS_SOURCE_OF_TRUTH.md Sec 6. Override with --batch_size.
BATCH_SIZE = 32

# DROPOUT RATE (2026-07-28, v5): LOCKED TO 0.0 FOR ALL FOUR MODELS.
# Was 0.5 (Long et al.'s value). Changed because dropout is the sole cause
# of the M1/M2 global decision inversion, established by elimination from
# the code rather than by experiment:
#   1. Train and val are drawn from ONE balanced pool, shuffled by
#      rng.permutation, then sliced -- same distribution, no overlap,
#      identical label mapping (see load_dataset).
#   2. run_epoch computes loss and accuracy from the same y; one-hot BCE
#      and argmax agree on polarity. No flip can originate there.
#   3. The ONLY difference between how train_acc and val_acc are measured
#      is model.train(True) vs model.train(False).
#   4. Dropout2d is the ONLY module in the whole architecture whose
#      behaviour depends on that flag (no BatchNorm/LayerNorm anywhere).
# Therefore a model scoring ~0.96 on train (dropout on) and ~0.04 on val
# (dropout off), on same-distribution data, admits exactly one explanation.
#
# Why it strikes M1/M2 and never M3/M4: the QNN back-end funnels the whole
# network through ONE scalar (64 -> fc2 -> 2 -> QNN -> 1 -> fc3 -> 2), so
# the decision is a threshold on a single number. Dropout shifts that
# number's distribution between modes and, with no redundancy to absorb it,
# the threshold flips for EVERY sample simultaneously. M3/M4's
# Linear(64->2) head is linear and unbottlenecked, so dropout's
# expectation-preservation (E[f(x)] = f(E[x])) holds exactly and they never
# invert. Trajectory-dependence follows too: the flip only occurs when the
# shift happens to cross fc3's learned threshold.
#
# Set to 0.0 for ALL FOUR models, not just M1/M2 -- the iso-parametric
# attribution (Sec 5) requires an identical backbone across models, so
# disabling it selectively would introduce a second uncontrolled variable.
# Dropout has ZERO trainable parameters, so no parameter count changes and
# the comparison is untouched. Documented deviation from Long et al.
# Override with --dropout for the A/B evidence runs.
DROPOUT_RATE = 0.0

# Optional (a,b) class-pair override for fashion_mnist, set by --classes.
CLASS_OVERRIDE = None

# QNN BACK-END IMPLEMENTATION (2026-07-28, v5): "torch" or "qiskit".
# See build_qnn_backend(). "torch" is exact and ~46-672x faster; "qiskit"
# is the reference implementation and will be REQUIRED for the noise phase,
# where Aer's NoiseModel cannot be expressed in the torch path.
QNN_BACKEND = "torch"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════
# QUANTUM FRONT-ENDS
#
# v5: the PennyLane device-selection / qnode / TorchLayer machinery that
# used to live here is gone. See quanv_torch.py and the GPU ROOT CAUSE
# section of this file's docstring. The circuits themselves are unchanged;
# only how they are evaluated changed.
#
# Circuit 4 (Hur et al. 2022 Fig. 2d), per pair (q_a, q_b), 6 shared params:
#     RY(w0) q_a ; RY(w1) q_b ; CRZ(w2) ctrl=q_b tgt=q_a
#     RY(w3) q_a ; RY(w4) q_b ; CRZ(w5) ctrl=q_a tgt=q_b
# The two middle single-qubit gates are RY, NOT RZ (verified 2026-07-18
# against the source figure). Implemented in quanv_torch.py's
# TrainableQuanvFrontEndTorch._block_unitary(), gate for gate.
#
# --backend pennylane restores the original PennyLane path for equivalence
# checking only. It is slow and must not be used for production runs.
# ═══════════════════════════════════════════════════════════════════════
BACKEND = "torch"   # set from --backend in main()


def make_trainable_frontend(seed):
    if BACKEND == "pennylane":
        from pennylane_frontends_legacy import TrainableQuanvFrontEndPennyLane
        return TrainableQuanvFrontEndPennyLane()
    return TrainableQuanvFrontEndTorch(seed=seed)


def make_frozen_frontend(seed):
    if BACKEND == "pennylane":
        from pennylane_frontends_legacy import FrozenHendersonFrontEndPennyLane
        return FrozenHendersonFrontEndPennyLane(seed)
    return FrozenHendersonFrontEndTorch(seed)


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
    def __init__(self, dropout=None):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 8, kernel_size=3)     # no padding
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3)    # no padding
        self.pool = nn.MaxPool2d(2)
        # DROPOUT (2026-07-28, v5): rate made configurable (--dropout) to
        # test the train/eval mismatch that inverted M1/M2's decision. See
        # MODELS_SOURCE_OF_TRUTH.md Sec 9. Dropout has NO trainable
        # parameters, so changing this rate does not affect any parameter
        # count or the iso-parametric comparison.
        self.drop = nn.Dropout2d(DROPOUT_RATE if dropout is None else dropout)

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
def build_qnn_backend(seed=None):
    """QNN back-end for M1/M2. Two implementations, identical mathematics.

    QNN_BACKEND == "torch" (default, 2026-07-28): the circuit simulated as
    batched torch tensor algebra on the model's device -- see qnn_torch.py.
    This is what removed the last CPU bottleneck: M1/M2 were 30-85x slower
    than M3/M4 purely because qiskit's Estimator is a pure-Python/numpy
    reference implementation with no CUDA path at any circuit size, and
    TorchConnector round-trips tensor->numpy->bind->simulate->numpy per
    call. Measured 46x faster at batch 32 and 672x at batch 128 (the gap
    widens because torch does one matmul while qiskit scales per-sample).
    At full scale that is 99.9 s/epoch -> 0.53 s/epoch for the back-end.
    Verified against the real EstimatorQNN to ~1e-6 (float32 precision) on
    forward, weight gradients, AND input gradients --
    scripts/verify_qnn_torch.py.

    QNN_BACKEND == "qiskit": the original TorchConnector(EstimatorQNN(...)).
    Kept as the reference for the equivalence check and for the later
    noise phase, where Aer's noise model genuinely requires it.
    """
    if QNN_BACKEND == "torch":
        return QNNBackendTorch(seed=seed)
    feature_map = ZZFeatureMap(feature_dimension=N_QUBITS_BE, reps=1)
    ansatz = RealAmplitudes(num_qubits=N_QUBITS_BE, reps=2)   # reps=2, NOT reps=1
    qnn = EstimatorQNN(
        circuit=feature_map.compose(ansatz),
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        # GRADIENT FIX (2026-07-27): EstimatorQNN defaults to
        # input_gradients=False, which means TorchConnector's backward
        # returns None for the QNN's input gradient -- PyTorch then never
        # traverses further upstream, so fc2 and the entire shared CNN
        # backbone silently never received a gradient and stayed at random
        # init for the whole run (confirmed: backbone.conv1.weight.grad was
        # None after backward for M1/M2, but populated for M3/M4). Setting
        # input_gradients=True restores the gradient chain into fc2 and the
        # backbone, matching M3/M4's training behaviour.
        input_gradients=True,
    )
    assert len(ansatz.parameters) == 6, f"RealAmplitudes(2,reps=2) should have 6 params, got {len(ansatz.parameters)}"
    return TorchConnector(qnn)


class QNNBackend(nn.Module):
    """M1 / M2 back-end. fc2(64->2) -> QNN(2->1) -> fc3(1->2).

    DEVICE FIX (2026-07-28, v5): qiskit's EstimatorQNN/TorchConnector has no
    GPU code path -- it internally round-trips through numpy to bind circuit
    parameters. v5 initially dropped v4's CPU-pinning workaround (that
    workaround previously covered the front-end only; the front-end is now
    correctly GPU-native, but the back-end still needs it). Root-caused via
    the smoke-scale run 2026-07-28: M1/M2 produced healthy, comparable-
    magnitude gradients on every trainable tensor (back_end.qnn.weight
    ~3.3e-2, same order as backbone.conv1.weight ~2.9e-2) yet accuracy was
    byte-identical across every epoch -- the classic signature of an
    autograd graph computing correct gradients while a downstream device
    round-trip silently fails to apply them to what's actually evaluated.
    Confirmed these runs had Device: cuda.

    Fix: THIS MODULE stays on CPU regardless of what device the rest of the
    model is on (see main()/build_model() -- back_end is explicitly NOT
    moved by the top-level model.to(device) call for M1/M2). forward()
    shuttles its input from the backbone's device to CPU and its output
    back, symmetric with how the front-end handled this in v4 before it
    became GPU-native. fc2/fc3 are tiny (130/4 params) so paying their
    compute on CPU is free; the qnn itself was always CPU-only regardless.
    """

    def __init__(self, seed=None):
        super().__init__()
        self.fc2 = nn.Linear(64, 2)
        self.qnn = build_qnn_backend(seed)
        self.fc3 = nn.Linear(1, 2)

    def forward(self, x):
        orig_device = x.device
        # Only the qiskit path needs the CPU round-trip (EstimatorQNN has no
        # CUDA code path). The torch path runs natively on the model's device.
        if QNN_BACKEND == "qiskit":
            x = x.cpu()
        batch_size = x.shape[0]
        x = self.fc2(x)
        x = self.qnn(x)
        # TorchConnector output shape can vary (B,1) or (B,) depending on
        # version/batch-size edge cases — normalise to (B,1) before fc3.
        x = x.reshape(batch_size, -1)
        x = self.fc3(x)
        return x.to(orig_device)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())


class ClassicalHead(nn.Module):
    """M3 / M4 back-end. fc2(64->2) straight to logits."""

    def __init__(self, seed=None):
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
        m = QCQCNN(make_frozen_frontend(seed), QNNBackend(seed))
    elif model_name == "M2":
        m = QCQCNN(make_trainable_frontend(seed), QNNBackend(seed))
    elif model_name == "M3":
        m = QCQCNN(make_trainable_frontend(seed), ClassicalHead())
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
        # 0=T-shirt/top 1=Trouser 2=Pullover 3=Dress 4=Coat
        # 5=Sandal 6=Shirt 7=Sneaker 8=Bag 9=Ankle boot
        # Thesis A specifies (0,1). NOTE (2026-07-29): that pair is trivially
        # separable -- a single-seed check gave M3 99.80% / M4 98.80%, i.e.
        # as saturated as MNIST, so it cannot differentiate architectures.
        # Override with --classes for a pair that has headroom.
        classes = CLASS_OVERRIDE or (0, 1)
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
def run_epoch(model, loader, optimizer, device, train=True, grad_log=None):
    """grad_log: optional list. If given and train=True, the front-end's
    per-step gradient vector is appended to it. Used to compute per-epoch
    gradient variance for the barren-plateau evidence -- see
    train_model()'s [GRADVAR] output. Thesis A Sec 5.1 commits to logging
    these statistics from the start rather than retrofitting them, so this
    is on by default for any model with a trainable quantum front-end.
    Cost is 6 floats per step; negligible."""
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
            if grad_log is not None:
                fe_w = getattr(model.front_end, "weights", None)
                if fe_w is not None and fe_w.grad is not None:
                    grad_log.append(fe_w.grad.detach().cpu().clone())
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def check_train_eval_shift(model, loader, device, model_name):
    """DIAGNOSTIC (2026-07-28, v5): measures the train-mode vs eval-mode
    discrepancy that globally inverted M1/M2's decision.

    Observed symptom (smoke run 2026-07-28, M2): train_acc climbed to 0.95
    while val_acc FELL to 0.032 -- and 1-0.032 = 0.968, i.e. inverting the
    validation predictions recovers better-than-train accuracy. That is not
    a model failing to learn; it is a model that learned the task and whose
    decision is systematically flipped at evaluation time.

    Mechanism: Dropout2d is expectation-preserving (E[dropout(x)] = x).
    For M3/M4's LINEAR head that is exact -- E[f(x)] = f(E[x]) -- so train
    and eval agree. For M1/M2's NONLINEAR quantum back-end it is not:
    E[QNN(x)] != QNN(E[x]), so the QNN's output distribution shifts between
    modes. fc3 = Linear(1,2) then turns that single scalar into two logits,
    and argmax flips when (w0-w1)*q + (b0-b1) changes sign -- so a global
    mean shift in q flips EVERY sample's prediction simultaneously, which
    is exactly the clean inversion observed.

    This runs the SAME batch through the model in both modes and reports
    the logit shift and how many predictions flip. Informational only.
    """
    x, y = next(iter(loader))
    x = x.to(device)
    with torch.no_grad():
        model.train(True)
        # average over several stochastic passes to estimate E[f(dropout(x))]
        tr_logits = torch.stack([model(x) for _ in range(8)]).mean(0)
        model.eval()
        ev_logits = model(x)
    tr_pred, ev_pred = tr_logits.argmax(1), ev_logits.argmax(1)
    flipped = (tr_pred != ev_pred).sum().item()
    margin_tr = (tr_logits[:, 0] - tr_logits[:, 1])
    margin_ev = (ev_logits[:, 0] - ev_logits[:, 1])
    print(f"[TRAIN/EVAL SHIFT] {model_name}: dropout_rate={DROPOUT_RATE}", flush=True)
    print(f"  mean logit margin (class0-class1)  train-mode={margin_tr.mean().item():+.4f}  "
          f"eval-mode={margin_ev.mean().item():+.4f}", flush=True)
    print(f"  predictions flipping between modes: {flipped}/{len(tr_pred)}", flush=True)
    if margin_tr.mean().item() * margin_ev.mean().item() < 0:
        print("  !! MARGIN SIGN FLIPS between train and eval -- decision is "
              "globally inverted at eval time. This is the M1/M2 failure mode.",
              flush=True)
    model.train(True)


def check_output_diversity(model, loader, device, model_name, n=8):
    """DIAGNOSTIC (2026-07-28, v5): added specifically to investigate M1/M2
    being stuck at byte-identical accuracy every epoch despite healthy,
    moving gradients (see MODELS_SOURCE_OF_TRUTH.md Sec 9, 2026-07-28 entry).
    That symptom is consistent with the model's DECISION not actually
    depending on the input -- i.e. correct weights, correct gradients,
    but a near-constant forward pass. This prints raw logits for n distinct
    samples so that can be checked directly, cheaply, in one run instead of
    guessed at. Purely informational -- does not raise.

    How to read it: if the n rows below are near-identical to each other
    (small spread relative to their magnitude), the model's output does not
    meaningfully depend on the input at this point in training -- point
    the investigation at fc2's output range / the ZZFeatureMap encoding
    next, not at gradients or device placement (both already ruled out for
    M2 on 2026-07-28). If the rows differ substantially from each other but
    argmax still doesn't flip across epochs, the issue is likely the
    decision boundary sitting in a fixed place relative to a narrow output
    range -- look at fc3's bias vs weight magnitude next.
    """
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n].to(device), y[:n]
    with torch.no_grad():
        logits = model(x)
    print(f"[OUTPUT DIVERSITY] {model_name}: logits for {logits.shape[0]} distinct samples "
          f"(label in brackets):", flush=True)
    for i in range(logits.shape[0]):
        print(f"  sample {i}  label={y[i].item()}  logits={logits[i].cpu().numpy()}", flush=True)
    spread = logits.std(dim=0)
    print(f"  per-class std across samples: {spread.cpu().numpy()}  "
          f"({'LOW -- output barely varies with input' if spread.max().item() < 1e-3 else 'nonzero spread'})",
          flush=True)
    model.train()


def assert_gradient_flow(model, loader, device, model_name):
    """Tripwire for the class of bug that left M1 at near-random accuracy:
    a broken gradient chain (e.g. TorchConnector's input_gradients=False)
    silently leaves the shared CNN backbone at random init while only the
    final few parameters train. One forward+backward, then check that every
    trainable tensor actually received a gradient. Cheap, runs once per job,
    and should be permanent -- this failure mode is invisible in the loss
    curve until the run is over."""
    model.train(True)
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    model.zero_grad()
    logits = model(x)
    loss = F.binary_cross_entropy(
        torch.sigmoid(logits), F.one_hot(y, num_classes=2).float())
    loss.backward()

    dead = []
    print(f"[GRADCHECK] {model_name}:", flush=True)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None:
            dead.append((name, "grad is None"))
            print(f"  {name:40s} grad=None   <-- DEAD", flush=True)
        else:
            gn = p.grad.norm().item()
            print(f"  {name:40s} |grad|={gn:.3e}", flush=True)
            if gn == 0.0:
                dead.append((name, "grad norm is exactly 0"))
    model.zero_grad()

    if dead:
        raise RuntimeError(
            f"[GRADCHECK FAILED] {model_name}: {len(dead)} trainable tensor(s) "
            f"received no gradient: {dead}. This is the M1 near-random-accuracy "
            f"failure mode -- training would burn walltime producing a model "
            f"stuck at random init. ABORTING."
        )
    print(f"[GRADCHECK] {model_name}: all trainable tensors receive gradient. OK",
          flush=True)


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
    single unbroken run, which the walltime kills already made impossible.

    BATCH SIZE (2026-07-28, v5): BATCH_SIZE (default 32) for all four
    models. The old bs=1 constraint applied to PennyLane's parameter-shift
    tape execution and no longer exists -- gradients now come from torch
    autograd on an exact simulation and are numerically identical to
    parameter-shift. See MODELS_SOURCE_OF_TRUTH.md Sec 6."""
    batch_size = BATCH_SIZE
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   weight_decay=WEIGHT_DECAY)
    # WARMUP CONDITION FIX (2026-07-28, v5): was gated on model_name in
    # ("M2","M3") -- i.e. keyed to trainable FRONT-end. But the slow-start
    # plateau this warmup exists to protect against is documented by Long
    # et al. (thesis draft Sec 4.3.2) for a frozen-front-end + trainable
    # quantum BACK-end model -- architecturally identical to M1 -- which
    # "starts near 50% (chance level) at epoch 0 and takes roughly 10-15
    # epochs just to catch up." M1 has a trainable QNN back-end and was
    # getting ZERO warmup (min_epochs=0) under the old condition, including
    # in the locked full-scale protocol (patience=10) -- a real risk to the
    # M1 replication anchor if the plateau exceeds patience before min_epochs
    # kicks in, since a min_epochs=0 model starts counting patience at
    # epoch 1. Root-caused via the smoke-scale run 2026-07-28: M2 showed
    # real weight movement and real (if still-collapsing) output variation
    # every epoch, consistent with being mid-plateau, not stuck -- but the
    # smoke config's min_epochs=8/patience=8 stopped it before it could
    # exit. Condition now covers any model with a trainable quantum
    # front-end OR back-end (M1, M2, M3); M4 (fully classical) is
    # unaffected.
    needs_warmup = model_name in ("M2", "M3") or isinstance(model.back_end, QNNBackend)
    min_epochs = MIN_EPOCHS_QUANTUM_FRONTEND if needs_warmup else 0

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

    assert_gradient_flow(model, train_loader, device, model_name)
    check_output_diversity(model, train_loader, device, model_name)
    check_train_eval_shift(model, train_loader, device, model_name)

    job_start = time.time()
    finished = False

    # DIAGNOSTIC (2026-07-28, v5): direct evidence for the M1/M2 device-pin
    # fix. The smoke run showed healthy, comparable-magnitude gradients on
    # back_end.qnn.weight every epoch, yet byte-identical accuracy -- i.e.
    # the weight update wasn't reaching what was actually evaluated. This
    # prints how far that exact tensor moved since the previous epoch, so a
    # rerun answers definitively whether the CPU pin fixed it, without a
    # third Gadi round-trip: if drift is ~0 every epoch, the pin alone
    # didn't fix it and hypothesis 2/3 (input-independent QNN output, or a
    # TorchConnector-side caching bug) need to be checked next. Cheap: one
    # tensor diff per epoch, only for models with a QNNBackend.
    _qnn_weight_ref = None
    if isinstance(model.back_end, QNNBackend):
        _qnn_weight_ref = model.back_end.qnn.weight.detach().clone()

    # BARREN-PLATEAU EVIDENCE (Thesis A Sec 5.1): "Gradient variance of the
    # front-end parameters, sampled across parameter space and across
    # training epochs, provides direct empirical evidence about whether the
    # barren plateau manifests in M2." The prediction on record is that
    # Circuit 4's weight sharing and shallow depth keep the variance on a
    # POLYNOMIALLY decaying trajectory rather than an exponentially decaying
    # one at the 4-qubit scale. Logging it per-epoch from the start (rather
    # than retrofitting) is an explicit Thesis A commitment. Only models
    # with a trainable front-end (M2, M3) have anything to log.
    track_grads = getattr(model.front_end, "weights", None) is not None

    for epoch in range(start_epoch, MAX_EPOCHS + 1):
        t0 = time.time()
        grad_log = [] if track_grads else None
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, device,
                                     train=True, grad_log=grad_log)
        va_loss, va_acc = run_epoch(model, val_loader, optimizer, device, train=False)
        dt = time.time() - t0
        rec = {"epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
                "val_loss": va_loss, "val_acc": va_acc, "time_sec": dt}
        gradvar_msg = ""
        if grad_log:
            G = torch.stack(grad_log)                    # (steps, 6)
            per_param_var = G.var(dim=0, unbiased=True)  # variance across steps
            mean_var = per_param_var.mean().item()
            mean_absgrad = G.abs().mean().item()
            rec["frontend_grad_var"] = per_param_var.tolist()
            rec["frontend_grad_var_mean"] = mean_var
            rec["frontend_grad_absmean"] = mean_absgrad
            rec["frontend_grad_steps"] = len(grad_log)
            gradvar_msg = f"  fe_gradvar={mean_var:.3e}"
        history.append(rec)
        drift_msg = ""
        if _qnn_weight_ref is not None:
            cur = model.back_end.qnn.weight.detach().clone()
            drift = (cur - _qnn_weight_ref).abs().max().item()
            drift_msg = f"  qnn_weight_drift={drift:.3e}"
            _qnn_weight_ref = cur
        # INVERSION DETECTOR (2026-07-28, v5): the QNN back-end can converge
        # to an inverted-polarity solution on some trajectories -- fc3 =
        # Linear(1,2) can represent either polarity of the QNN's scalar
        # output, and the long plateau is where that commitment happens.
        # Observed once (run 174891473: train_acc 0.96 while val_acc fell to
        # 0.032; inverting the val predictions gives 0.968, i.e. better than
        # train). A rerun of the identical config converged normally to 0.99,
        # so this is trajectory-dependent, not systematic -- but it is
        # CATASTROPHIC and SILENT when it happens: early stopping keeps the
        # degenerate epoch-1 state and reports test_acc=0.50, which looks
        # like an architectural finding rather than a failed run. Flag it
        # loudly. Signature: high train accuracy with BELOW-CHANCE val
        # accuracy on a balanced binary task.
        # DECISIVE DIAGNOSTIC (2026-07-28, v5): train_acc is measured in
        # TRAIN mode (dropout on) and val_acc in EVAL mode (dropout off),
        # so comparing them confounds "different data" with "different
        # mode". This evaluates the TRAINING set in EVAL mode, isolating
        # the mode effect on identical data:
        #
        #   tr_acc_eval ~= 1 - tr_acc   -> the model's decision INVERTS
        #                                  between modes on the same data.
        #                                  Dropout is the only train/eval
        #                                  difference, so dropout is the
        #                                  cause, via the nonlinear QNN
        #                                  (E[QNN(x)] != QNN(E[x])).
        #   tr_acc_eval ~= tr_acc       -> mode is NOT the cause; the model
        #                                  is consistent on train data in
        #                                  both modes, and the problem lies
        #                                  in the val SET or split.
        #
        # Only run for QNN-back-end models (M1/M2) and only once inversion
        # is actually suspected, to avoid paying an extra eval pass every
        # epoch on healthy runs.
        if _qnn_weight_ref is not None and tr_acc > 0.70 and va_acc < 0.50:
            _, tr_acc_eval = run_epoch(model, train_loader, optimizer,
                                        device, train=False, grad_log=None)
            verdict = ("MODE FLIP CONFIRMED -- same data, opposite decision "
                       "in eval mode. Dropout is the only train/eval "
                       "difference; the nonlinear QNN back-end is not "
                       "expectation-preserving under it."
                       if tr_acc_eval < 0.50 else
                       "mode is NOT the cause -- model is consistent on "
                       "train data in both modes. Investigate the val "
                       "split instead.")
            print(f"  [MODE CHECK] {model_name} epoch {epoch}: "
                  f"train-set accuracy in TRAIN mode={tr_acc:.4f}, "
                  f"in EVAL mode={tr_acc_eval:.4f}  -> {verdict}", flush=True)
        if tr_acc > 0.70 and va_acc < 0.50:
            print(f"  [INVERSION] {model_name} epoch {epoch}: train_acc="
                  f"{tr_acc:.4f} but val_acc={va_acc:.4f} (BELOW CHANCE). "
                  f"Inverting val predictions would give {1-va_acc:.4f}. "
                  f"This run has likely converged to an inverted-polarity "
                  f"solution -- DO NOT report its result as a valid "
                  f"architectural outcome. See MODELS_SOURCE_OF_TRUTH.md "
                  f"Sec 9 (2026-07-28).", flush=True)
        print(f"[{model_name}] epoch {epoch:3d}  train_acc={tr_acc:.4f} "
              f"val_acc={va_acc:.4f} ({dt:.1f}s){drift_msg}{gradvar_msg}", flush=True)
        if epoch in (3, 15, 25) and isinstance(model.back_end, QNNBackend):
            check_output_diversity(model, train_loader, device,
                                    f"{model_name} (after epoch {epoch})")
            check_train_eval_shift(model, train_loader, device,
                                    f"{model_name} (after epoch {epoch})")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        elif epoch >= min_epochs:
            # EARLY-STOPPING FIX (2026-07-28, v5): this increment was
            # previously unconditional, so patience_ctr accumulated during
            # the warmup window and was already saturated the moment
            # min_epochs was satisfied -- meaning the run stopped on the
            # very first epoch past warmup, every time, regardless of what
            # training was doing. Caught on the smoke-scale M2 run
            # 2026-07-28: the quantum back-end's documented plateau broke
            # at epoch 13 (train_acc 0.506 -> 0.512 -> 0.606 -> 0.750,
            # exactly the "10-15 epochs" Long et al. report for this
            # architecture in thesis draft Sec 4.3.2), but min_epochs=15 /
            # patience=10 killed the run at epoch 15 with patience_ctr=14
            # already banked from the plateau -- discarding the model right
            # as it started learning, and returning the epoch-1 degenerate
            # all-one-class state as "best" (hence FINAL test_acc=0.5000).
            # This contradicted the documented intent in
            # MODELS_SOURCE_OF_TRUTH.md Sec 6 ("warmup BEFORE patience
            # counting") -- only the check was gated, not the counting.
            # At full scale the same bug would have stopped M2/M3 at
            # exactly epoch 30 (min_epochs) on every seed.
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
    ap.add_argument("--n_train", type=int, default=None,
                     help="Override N_TRAIN (default 5000, the locked "
                          "full protocol value). Use for reduced-scale "
                          "feasibility runs -- e.g. local runs where the "
                          "full protocol doesn't fit the available time. "
                          "N_VAL/N_TEST must also be given if this is.")
    ap.add_argument("--n_val", type=int, default=None,
                     help="Override N_VAL (default 1250).")
    ap.add_argument("--n_test", type=int, default=None,
                     help="Override N_TEST (default 1000).")
    ap.add_argument("--max_epochs", type=int, default=None,
                     help="Override MAX_EPOCHS (default 100).")
    ap.add_argument("--min_epochs_quantum", type=int, default=None,
                     help="Override MIN_EPOCHS_QUANTUM_FRONTEND, the warmup "
                          "before patience counting starts for any model "
                          "with a trainable quantum front-end or back-end "
                          "(M1, M2, M3; widened 2026-07-28 -- was M2/M3-only) "
                          "(default 30).")
    ap.add_argument("--patience", type=int, default=None,
                     help="Override PATIENCE (default 10).")
    ap.add_argument("--batch_size", type=int, default=None,
                     help="Override BATCH_SIZE (default 32, all four models). "
                          "Used by the bs=1 vs bs=32 control run. Keep it "
                          "matched across models for any comparison run -- "
                          "an unmatched batch size is a second uncontrolled "
                          "variable inside the M3-vs-M4 comparison.")
    ap.add_argument("--weight_decay", type=float, default=None,
                     help="Override WEIGHT_DECAY (default 1e-4, applied "
                          "uniformly to all four models via AdamW per "
                          "Thesis A Sec 5.3). Set 0.0 to reproduce the "
                          "unregularised pre-2026-07-28 behaviour.")
    ap.add_argument("--classes", type=str, default=None,
                     help="Override the fashion_mnist class pair, e.g. "
                          "--classes 0,6 (T-shirt/top vs Shirt). Thesis A "
                          "specifies 0,1 but that pair is saturated.")
    ap.add_argument("--dropout", type=float, default=None,
                     help="Override DROPOUT_RATE (default 0.0 as of "
                          "2026-07-28; was 0.5). Dropout has no trainable "
                          "parameters, so this does NOT affect parameter "
                          "counts or the iso-parametric comparison. Use "
                          "--dropout 0.5 to reproduce the M1/M2 global "
                          "decision inversion (see MODELS_SOURCE_OF_TRUTH.md "
                          "Sec 9).")
    ap.add_argument("--qnn_backend", choices=["torch", "qiskit"], default="torch",
                     help="QNN back-end implementation for M1/M2. 'torch' "
                          "(default) simulates the circuit as batched tensor "
                          "algebra on the GPU -- exact, and ~46-672x faster "
                          "than qiskit depending on batch size. 'qiskit' is "
                          "the original TorchConnector(EstimatorQNN) "
                          "reference, and will be required for the noise "
                          "phase.")
    ap.add_argument("--backend", choices=["torch", "pennylane"], default="torch",
                     help="Quantum front-end evaluation backend. 'torch' "
                          "(default) computes the circuits as exact batched "
                          "tensor algebra on the GPU. 'pennylane' restores "
                          "the original TorchLayer/parameter-shift path and "
                          "is for equivalence checking ONLY -- it is orders "
                          "of magnitude slower and requires batch_size=1.")
    args = ap.parse_args()

    global BACKEND
    BACKEND = args.backend
    if BACKEND == "pennylane":
        print("[BACKEND] PennyLane reference path selected. This is for "
              "equivalence checking only -- it is orders of magnitude slower "
              "than the torch path and expects --batch_size 1. Do NOT use "
              "for production runs.", flush=True)
    else:
        print("[BACKEND] torch-native quantum front-end (exact batched "
              "tensor algebra, autograd gradients == parameter-shift).",
              flush=True)

    # Apply any scale/schedule overrides. Left untouched (module defaults =
    # locked full protocol: 5000/1250/1000, 100 max epochs, patience 10,
    # min_epochs 30) unless explicitly passed -- so run_chain_link.pbs and
    # every existing Gadi invocation are completely unaffected by this.
    global N_TRAIN, N_VAL, N_TEST, MAX_EPOCHS, PATIENCE, MIN_EPOCHS_QUANTUM_FRONTEND, BATCH_SIZE, DROPOUT_RATE, QNN_BACKEND, WEIGHT_DECAY, CLASS_OVERRIDE
    if args.n_train is not None:
        N_TRAIN = args.n_train
    if args.n_val is not None:
        N_VAL = args.n_val
    if args.n_test is not None:
        N_TEST = args.n_test
    if args.max_epochs is not None:
        MAX_EPOCHS = args.max_epochs
    if args.patience is not None:
        PATIENCE = args.patience
    if args.min_epochs_quantum is not None:
        MIN_EPOCHS_QUANTUM_FRONTEND = args.min_epochs_quantum
    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
    if args.classes is not None:
        CLASS_OVERRIDE = tuple(int(v) for v in args.classes.split(','))
    if args.dropout is not None:
        DROPOUT_RATE = args.dropout
    if args.weight_decay is not None:
        WEIGHT_DECAY = args.weight_decay
    QNN_BACKEND = args.qnn_backend

    is_reduced_scale = any(v is not None for v in (
        args.n_train, args.n_val, args.n_test, args.max_epochs,
        args.min_epochs_quantum, args.patience
    ))
    if is_reduced_scale:
        print(f"[PROTOCOL] REDUCED-SCALE / FEASIBILITY RUN -- not the locked "
              f"full protocol. N_TRAIN={N_TRAIN} N_VAL={N_VAL} N_TEST={N_TEST} "
              f"MAX_EPOCHS={MAX_EPOCHS} PATIENCE={PATIENCE} "
              f"MIN_EPOCHS_QUANTUM_FRONTEND={MIN_EPOCHS_QUANTUM_FRONTEND}. "
              f"Label any results from this run accordingly -- do not mix "
              f"with full-scale (5000/1250/1000, 100 max epochs) results.",
              flush=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"Run: model={args.model} dataset={args.dataset} seed={args.seed} "
          f"noise_alpha={args.noise_alpha} batch_size={BATCH_SIZE} lr={LR} "
          f"weight_decay={WEIGHT_DECAY} dropout={DROPOUT_RATE}",
          flush=True)

    if args.noise_alpha != 0.0:
        print("[WARNING] noise_alpha != 0.0 requested but IBM-calibrated noise "
              "injection is not wired into this Phase-1 script — running IDEAL "
              "(noiseless) simulation regardless. Implement Aer noise model "
              "before running the noise sweep phase.", flush=True)

    model = build_model(args.model, args.seed).to(device)
    print(f"[DEVICE] whole model on {device} "
          f"(front-end included -- v5 torch-native quantum simulation)", flush=True)
    if isinstance(model.back_end, QNNBackend) and QNN_BACKEND == "qiskit":
        # DEVICE FIX (2026-07-28, v5): EstimatorQNN/TorchConnector has no
        # GPU code path. Root-caused via the smoke-scale run -- M1/M2 on
        # cuda produced healthy gradients (comparable magnitude to the
        # backbone) but byte-identical accuracy every epoch, because the
        # weight updates weren't reaching what the Estimator actually
        # evaluates. See QNNBackend's docstring for the full trace. Pinning
        # this submodule to CPU explicitly, after the top-level .to(device)
        # call above, undoes that move for back_end only -- front_end and
        # backbone stay on device.
        model.back_end.to("cpu")
        print("[DEVICE] back_end (QNNBackend/TorchConnector) pinned to CPU "
              "-- EstimatorQNN has no GPU path. front_end and backbone stay "
              "on device.", flush=True)

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
        "protocol": {
            "reduced_scale": is_reduced_scale,
            "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "min_epochs_quantum_frontend": MIN_EPOCHS_QUANTUM_FRONTEND,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "backend": BACKEND,
            "dropout": DROPOUT_RATE,
            "classes": list(CLASS_OVERRIDE) if CLASS_OVERRIDE else None,
            "qnn_backend": QNN_BACKEND,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
        },
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
