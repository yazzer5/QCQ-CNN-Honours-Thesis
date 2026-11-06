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
  Seeds             : 42-46 (v6-v10), 42-51 from v12 -- matches Long's 10

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

# QISKIT IS IMPORTED LAZILY (v8.1, 2026-08-11). It used to be imported here
# at module scope, which meant ANY problem in the qiskit install killed every
# run -- including runs that never touch qiskit. That is exactly what
# happened: Gadi job 175978087 (M2/fashion 2,6, --qnn_backend torch) died 18 s
# in, before main() printed a single line, because qiskit's own __init__.py
# raises when it detects both a >=1.0 and a <1.0 distribution record in the
# environment. The production path had no need of qiskit at all.
#
# The three names below are used ONLY by build_qnn_backend() when
# QNN_BACKEND == "qiskit". That path is the reference implementation for
# scripts/verify_qnn_torch.py and is REQUIRED for the later noise phase
# (Aer's NoiseModel cannot be expressed in the torch path). It is NOT used by
# any alpha=0 production run: --qnn_backend torch is the default and is what
# the PBS scripts pass.
#
# Deferring the import means a broken or mis-resolved qiskit degrades the
# reference path only, instead of taking down the whole pipeline. See
# _import_qiskit() below.


def _import_qiskit():
    """Imports the qiskit names needed by the reference QNN back-end.

    Raises a RuntimeError that names the actual cause and the workaround,
    rather than letting qiskit's own packaging error surface with no
    context about which part of this pipeline needed it."""
    try:
        from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
        from qiskit_machine_learning.neural_networks import EstimatorQNN
        from qiskit_machine_learning.connectors import TorchConnector
    except Exception as exc:
        raise RuntimeError(
            f"--qnn_backend qiskit was requested, but the qiskit stack failed "
            f"to import: {exc}\n"
            f"The pinned stack is qiskit==0.45.3 + qiskit-machine-learning=="
            f"0.7.2 (which requires qiskit<1.0) + setuptools==69.5.1.\n"
            f"Check for a stale >=1.0 distribution record with:\n"
            f"  pip list | grep -i qiskit\n"
            f"  ls -d $CONDA_PREFIX/lib/python3.11/site-packages/qiskit*/\n"
            f"NOTE: --qnn_backend torch (the default) does NOT need qiskit "
            f"and is unaffected by this."
        ) from exc
    return ZZFeatureMap, RealAmplitudes, EstimatorQNN, TorchConnector

# ═══════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════
N_C4_PARAMS   = 6      # Circuit 4 shared params (theta0..theta5)
N_QUBITS_FE   = 4      # front-end qubits (one per 2x2-patch pixel)
N_QUBITS_BE   = 2      # QNN backend qubits
N_CLASSES     = 2      # set from --classes in main(); 2 for binary, 10 for full MNIST
N_OBS         = 1      # QNN readout observables; 1 for binary, 3 for multiclass
HEAD_VARIANT  = "linear"  # "linear" | "tanh" -- classical mid-map activation
SAVE_PREDICTIONS = False  # set from --save_predictions

# LOSS FUNCTION (v10-loss-smoke, 2026-08-13). Default "bce" reproduces
# v6-v10 exactly: sigmoid on each logit independently, then BCE against the
# one-hot target. "ce" is softmax cross-entropy, i.e. Long et al. eq (21).
#
# WHY THIS IS A SWITCH AND NOT AN EDIT: at 2 classes the two are close
# enough that the binary grids are unaffected either way, so nothing about
# v6-v9 is in question. At 10 classes they are not close. Under BCE each of
# the ten outputs is scored independently, so nine of the ten targets are
# "no" for every image and the loss pushes every logit toward
# sigmoid^-1(0.1) ~= -2.2. M1/M2's logits are fc3(Linear(3,10)) applied to
# QNN expectations bounded in [-1,1]; M3/M4's head is unbounded. That is an
# asymmetric handicap on exactly the two models the primary contrast sits
# between. Softmax normalises across classes instead, so only the ARGMAX
# matters and the bounded range stops being a constraint.
#
# Local synthetic-data evidence (CPU, 1 seed, 15 epochs, not MNIST):
#   BCE: M1 11.6% M2 15.6% M3 46.2% M4 50.0%
#   CE : M1 53.6% M2 66.6% M3 100%  M4 98.6%
# and at 60 epochs BCE gave M2 BELOW M1, reversing the sign of the primary
# contrast. That evidence is suggestive, not decisive -- the 10-class
# diagnostic reached 72.9%/66.5% under BCE on real MNIST, so the collapse
# is clearly not total there. Hence: measure it, do not assume it.
LOSS = "bce"


def classification_loss(logits, y):
    """Training loss. Single definition, used by the epoch loop AND by every
    diagnostic that computes a loss, so the two can never silently diverge."""
    if LOSS == "ce":
        return F.cross_entropy(logits, y)
    y_onehot = F.one_hot(y, num_classes=N_CLASSES).float()
    return F.binary_cross_entropy(torch.sigmoid(logits), y_onehot)

SEEDS      = list(range(42, 52))   # v12: widened to 10 seeds (2026-10-02)
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

# CODE VERSION (v9, 2026-08-12). Stamped into every result JSON and printed
# at the top of every run. v9 is v8 with NO code changes at all -- the tag is
# bumped purely so that the Fashion 2,4 grid is attributable to its own
# release. v8 (Fashion 2,6) and v9 (Fashion 2,4) accuracy numbers are
# therefore directly comparable to each other and to the v7 0,6 grid; the
# only behavioural difference from v7 remains the gap@best measurement
# introduced in v8 (see train_model()).
CODE_VERSION = "v10"

# Binary (v6-v9) budget, retained so 2-class runs stay byte-comparable.
EXPECTED_PARAMS_BINARY = {"M1": 1604, "M2": 1610, "M3": 1600, "M4": 1594}


def expected_params(model_name, n_classes, n_obs):
    """Iso-parametric budget as a function of the class count.

    Binary path is unchanged from v9. For n_classes > 2 the head widens and
    the four models must be re-padded, because the naive widening breaks the
    match badly: a Linear(64, 10) classical head is 650 params against the
    QNN head's 176, a 474-param gap on a ~1600-param budget. v10 restores the
    match by giving the classical head the same SHAPE as the quantum one --
    fc2(64->2), then a 2->n_obs map standing in for the QNN, then
    fc3(n_obs->n_classes) -- so the only thing that differs between M2 and M3
    is whether that middle 2->3 map is a quantum circuit or a linear layer.

      M1 = 1464 backbone + 130 fc2 +  6 qnn + fc3          (frozen front-end)
      M2 = M1 + 6 front-end params
      M3 = 1464 backbone + 130 fc2 +  9 mid + fc3 + 6 front-end
      M4 = M3 - 6 front-end params
    """
    if n_classes == 2 and n_obs == 1:
        return EXPECTED_PARAMS_BINARY[model_name]
    backbone, fc2 = 1464, 130
    fc3 = n_obs * n_classes + n_classes          # Linear(n_obs, n_classes)
    qnn_head = fc2 + 6 + fc3                     # QNN mid-map has 6 params
    mid = 2 * n_obs + n_obs                      # Linear(2, n_obs)
    cls_head = fc2 + mid + fc3
    return {"M1": backbone + qnn_head,
            "M2": backbone + qnn_head + 6,
            "M3": backbone + cls_head + 6,
            "M4": backbone + cls_head}[model_name]

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
        return QNNBackendTorch(seed=seed, n_obs=N_OBS)
    # Lazy import: only the qiskit reference path needs these. See
    # _import_qiskit() for why this is not a module-scope import.
    ZZFeatureMap, RealAmplitudes, EstimatorQNN, TorchConnector = _import_qiskit()
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
        self.fc3 = nn.Linear(N_OBS, N_CLASSES)

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
    """M3 / M4 back-end.

    Binary (N_CLASSES == 2, N_OBS == 1): fc2(64->2) straight to logits,
    byte-identical to v6-v9.

    Multiclass: fc2(64->2) -> fc_mid(2->N_OBS) -> fc3(N_OBS->N_CLASSES).
    fc_mid is the classical stand-in for the QNN's 2->N_OBS map, which is
    what keeps M2-M3 a clean quantum-vs-classical contrast at matched
    parameter count. NOTE the honest caveat: the QNN's map is nonlinear and
    bounded in [-1,1] while fc_mid is linear, so fc_mid composed with fc2
    collapses to a single rank-2 linear map. See --head_variant if you want
    a tanh in the middle instead; that choice is a pre-registration decision
    and is deliberately NOT made for you here."""

    def __init__(self, seed=None):
        super().__init__()
        self.fc2 = nn.Linear(64, 2)
        if N_CLASSES == 2 and N_OBS == 1:
            self.fc_mid = None
            self.fc3 = None
        else:
            self.fc_mid = nn.Linear(2, N_OBS)
            self.fc3 = nn.Linear(N_OBS, N_CLASSES)

    def forward(self, x):
        x = self.fc2(x)
        if self.fc_mid is None:
            return x
        x = self.fc_mid(x)
        if HEAD_VARIANT == "tanh":
            x = torch.tanh(x)
        return self.fc3(x)

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
    expected = expected_params(model_name, N_CLASSES, N_OBS)
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


def _open_offline_first(ds_cls, data_root, train):
    """Open a torchvision dataset WITHOUT touching the network if possible.

    v7 change. Gadi compute nodes have no internet. v6 called download=True and
    relied on torchvision's download path noticing the staged .gz had a matching
    MD5 and skipping the fetch. That does work, but it is a fallback inside the
    download path and its behaviour has moved across torchvision versions.

    Here we try download=False first. If the extracted idx files are present
    (scripts/preflight_data.py puts them there) this succeeds immediately and no
    network call is ever attempted. Only if that fails do we fall back to the v6
    behaviour, which still works on a login node or from staged .gz archives.
    """
    try:
        return ds_cls(data_root, train=train, download=False)
    except Exception as exc:
        print(f"[data] offline open of {ds_cls.__name__}(train={train}) failed "
              f"({type(exc).__name__}: {exc}); retrying with download=True. "
              f"On a compute node this will fail -- run "
              f"`python scripts/preflight_data.py` on the login node first.")
        return ds_cls(data_root, train=train, download=True)


def load_dataset(name, data_root, seed):
    """Returns (train_ds, val_ds, test_ds) each yielding (1,28,28) float tensors
    in [0,1] and integer labels in {0,1}."""
    rng = np.random.RandomState(seed)

    if name == "mnist":
        # v10: honours --classes so the same code path serves both the
        # 3-vs-5 binary replication and the full 10-class task. Pass
        # --classes 0,1,2,3,4,5,6,7,8,9 for 10-class MNIST.
        classes = CLASS_OVERRIDE or (3, 5)
        train_raw = _open_offline_first(datasets.MNIST, data_root, True)
        test_raw = _open_offline_first(datasets.MNIST, data_root, False)
    elif name == "fashion_mnist":
        # 0=T-shirt/top 1=Trouser 2=Pullover 3=Dress 4=Coat
        # 5=Sandal 6=Shirt 7=Sneaker 8=Bag 9=Ankle boot
        # Thesis A specifies (0,1). NOTE (2026-07-29): that pair is trivially
        # separable -- a single-seed check gave M3 99.80% / M4 98.80%, i.e.
        # as saturated as MNIST, so it cannot differentiate architectures.
        # Override with --classes for a pair that has headroom.
        classes = CLASS_OVERRIDE or (0, 1)
        train_raw = _open_offline_first(datasets.FashionMNIST, data_root, True)
        test_raw = _open_offline_first(datasets.FashionMNIST, data_root, False)
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
    # v10: per-class counts divide by len(classes), not a hardcoded 2, so a
    # 10-class run draws N_TRAIN+N_VAL images balanced across ten classes
    # rather than silently building a 5x-oversized set.
    n_cls = len(classes)
    pool_imgs, pool_labs = _balanced_subset(train_imgs, train_labs, classes, (N_TRAIN + N_VAL) // n_cls)
    te_imgs, te_labs = _balanced_subset(test_imgs, test_labs, classes, N_TEST // n_cls)
    assert len(np.unique(pool_labs)) == n_cls, (
        f"[CLASS COVERAGE] expected {n_cls} classes in the training pool, "
        f"got {len(np.unique(pool_labs))}. Check --classes and the data root.")
    print(f"[DATA] {name}: {n_cls} classes, pool={len(pool_labs)} test={len(te_labs)} "
          f"({(N_TRAIN + N_VAL) // n_cls} train+val and {N_TEST // n_cls} test per class)",
          flush=True)

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
        if train:
            optimizer.zero_grad()
        logits = model(x)
        loss = classification_loss(logits, y)
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
    loss = classification_loss(logits, y)
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


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Per-image predictions and labels on the restored best checkpoint.

    Why this exists (v10): collapsing 1000 individual test decisions into a
    single accuracy scalar, then comparing five such scalars with a paired
    t-test at df=4, throws away almost all the information in a grid. The
    per-image outcomes support McNemar's test, which asks how often the two
    models DISAGREE on the same image and in which direction. On 2000 images
    x 5 seeds that detects a 0.8 pp effect at z~2.5, where the accuracy-mean
    t-test needs roughly 25 seeds for the same effect.

    Order is deterministic (shuffle=False) so predictions from two different
    models line up index-for-index. Do not reorder the test loader."""
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        logits = model(x.to(device))
        preds.append(logits.argmax(dim=1).cpu())
        labels.append(y.cpu())
    return torch.cat(preds).tolist(), torch.cat(labels).tolist()


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
    needs_warmup = model_name in ("M2", "M3", "M5") or isinstance(model.back_end, QNNBackend)
    min_epochs = MIN_EPOCHS_QUANTUM_FRONTEND if needs_warmup else 0
    print(f"[V135] warmup: model={model_name} min_epochs={min_epochs}", flush=True)  # V135_M5_PATCH

    start_epoch = 1
    best_val_acc, best_state, patience_ctr = -1.0, None, 0
    best_epoch = None
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
        # v10: the inversion signature (below-chance val accuracy on a
        # balanced binary task) is only well-defined for 2 classes. At 10
        # classes "below chance" is 10%, and a genuine early-training state
        # sits there legitimately, so the detector would fire constantly.
        # Gated off for multiclass rather than deleted.
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
        if (N_CLASSES == 2 and _qnn_weight_ref is not None
                and tr_acc > 0.70 and va_acc < 0.50):
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
            best_epoch = epoch
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

    # ═══════════════════════════════════════════════════════════════════
    # GENERALISATION GAP AT THE BEST CHECKPOINT (v8, 2026-08-10)
    #
    # v7 wrote no gap field at all; the gap was reconstructed afterwards
    # from `history` as (final-epoch train_acc - test_acc). That compares
    # TWO DIFFERENT MODELS: the line above restores the best-validation
    # checkpoint (epoch ~12-17 for M1/M2) before testing, while the final
    # history entry comes from epoch ~39-61. M1/M2 keep overfitting past
    # their peak, so their final-epoch train accuracy is inflated relative
    # to the model actually being evaluated -- which is what manufactured
    # the "gap is monotone in quantum content, p < 1e-4" claim that was
    # later withdrawn (FMNIST_LOG.md Sec 5.1/6.1).
    #
    # Fixed here by measuring the train set with the SAME weights that
    # produce test_acc, immediately after the restore and before anything
    # else touches the model. Two further points that the post-hoc
    # reconstruction could not fix even in principle:
    #
    #   * history["train_acc"] is a RUNNING average over an epoch during
    #     which the weights were still being updated. It is not the
    #     accuracy of any single set of weights.
    #   * history["train_acc"] is measured in TRAIN mode, test_acc in EVAL
    #     mode. At DROPOUT_RATE=0.0 those coincide, but the moment arm (b)
    #     runs at dropout 0.5 they do not, and the QNN back-end is exactly
    #     the architecture where that difference is not benign (Sec 9).
    #
    # `train_acc_at_best` below is eval-mode, single-checkpoint, and
    # therefore directly comparable to `test_acc`. Cost is one extra
    # forward pass over 5,000 images, once per run.
    #
    # `train_acc_final_trainmode` is retained ONLY so the v7 Fashion 0,6
    # grid can be recomputed on the old definition for continuity; it must
    # not be used for any new claim.
    # ═══════════════════════════════════════════════════════════════════
    _, tr_acc_at_best = run_epoch(model, train_loader, optimizer, device, train=False)
    te_loss, te_acc = run_epoch(model, test_loader, optimizer, device, train=False)
    gap_at_best = tr_acc_at_best - te_acc

    # Derive the best epoch from history rather than trusting the loop
    # variable: on a checkpoint resume the loop variable is None because
    # save_checkpoint() does not carry it, whereas history does. Ties go to
    # the earliest epoch, matching the strict `>` used to accept a new best.
    best_epoch_from_history = None
    if history:
        best_epoch_from_history = max(
            history, key=lambda r: (r["val_acc"], -r["epoch"])
        )["epoch"]
    if best_epoch is not None and best_epoch_from_history is not None \
            and best_epoch != best_epoch_from_history:
        print(f"  [WARN] best_epoch bookkeeping disagrees: loop={best_epoch} "
              f"history={best_epoch_from_history}. Using history.", flush=True)
    best_epoch = best_epoch_from_history if best_epoch_from_history is not None else best_epoch

    print(f"[{model_name}] FINAL test_acc={te_acc:.4f}", flush=True)
    print(f"[{model_name}] GAP@BEST train_acc={tr_acc_at_best:.4f} "
          f"test_acc={te_acc:.4f} gap={gap_at_best*100:+.2f} pp "
          f"(both measured on the epoch-{best_epoch} checkpoint, eval mode)",
          flush=True)

    if checkpoint_path is not None and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)   # done -- no resumable state needed anymore

    preds = labels = None
    if SAVE_PREDICTIONS:
        preds, labels = collect_predictions(model, test_loader, device)
        assert len(preds) == len(labels), "prediction/label length mismatch"
        recomputed = sum(int(a == b) for a, b in zip(preds, labels)) / len(labels)
        # Tripwire: if this disagrees with te_acc the saved predictions come
        # from a different model state than the reported accuracy, which is
        # exactly the class of defect that produced the retracted gap metric.
        assert abs(recomputed - te_acc) < 1e-6, (
            f"[PREDICTION MISMATCH] recomputed acc {recomputed:.6f} != "
            f"reported test_acc {te_acc:.6f}. Do not use these predictions.")
        print(f"[PREDICTIONS] saved {len(preds)} per-image test predictions, "
              f"recomputed acc={recomputed:.4f} matches reported", flush=True)

    return {"status": "done", "history": history, "best_val_acc": best_val_acc,
            "test_acc": te_acc, "test_loss": te_loss,
            "test_predictions": preds, "test_labels": labels,
            "best_epoch": best_epoch,
            "stop_epoch": history[-1]["epoch"] if history else None,
            "train_acc_at_best": tr_acc_at_best,
            "gap_at_best": gap_at_best,
            "gap_metric": "train_acc_at_best(eval mode) - test_acc, same checkpoint",
            "train_acc_final_trainmode": history[-1]["train_acc"] if history else None}


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(["M1", "M2", "M3", "M4"]) + ["M5"])
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
    ap.add_argument("--n_obs", type=int, default=None, choices=[1, 3, 6, 15],
                    help="QNN readout observables. Default: 1 for binary, 3 for "
                         "multiclass. 3 costs zero extra circuit parameters -- "
                         "<Z0>, <Z1>, <ZZ> are read off the same statevector.")
    ap.add_argument("--head_variant", choices=["linear", "tanh"], default="linear",
                    help="Classical mid-map activation in M3/M4's multiclass head. "
                         "'linear' keeps the head a pure linear map (matching v6-v9 "
                         "behaviour); 'tanh' bounds it to [-1,1] like the QNN's "
                         "output. PRE-REGISTER this choice before running a grid.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Enable torch deterministic algorithms + cuDNN determinism. "
                         "Costs ~10-20%% throughput, removes the ~0.8 pp fixed-seed "
                         "run-to-run variance that currently inflates SD(diff) and "
                         "weakens every paired test.")
    ap.add_argument("--save_predictions", action="store_true",
                    help="Save per-image test predictions and labels to the result "
                         "JSON. Enables McNemar's test, which uses paired per-image "
                         "outcomes instead of 5 averaged scalars and is far more "
                         "sensitive to the ~1 pp effects this project is chasing.")
    ap.add_argument("--loss", choices=["bce", "ce"], default="bce",
                    help="Training loss. 'bce' (DEFAULT, unchanged from v6-v10) "
                         "scores each of the N_CLASSES logits independently with "
                         "sigmoid + binary cross-entropy against the one-hot "
                         "target. 'ce' is softmax cross-entropy (Long et al. "
                         "eq 21). Identical models, identical parameter counts, "
                         "identical data -- ONLY the scoring changes. Defaults to "
                         "'bce' so that omitting this flag reproduces v10 exactly.")
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
    global N_CLASSES, N_OBS, HEAD_VARIANT, SAVE_PREDICTIONS, LOSS
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

    # ── MULTICLASS RESOLUTION (v10) ──────────────────────────────────────
    # N_CLASSES is derived from --classes so it can never disagree with the
    # data actually loaded. N_OBS defaults to 3 whenever N_CLASSES > 2: the
    # 10-class diagnostic showed the QNN back-end is measurement-limited,
    # not parameter-limited, and a single scalar cannot carry 10 classes.
    if CLASS_OVERRIDE is not None:
        N_CLASSES = len(CLASS_OVERRIDE)
    N_OBS = args.n_obs if args.n_obs is not None else (3 if N_CLASSES > 2 else 1)
    HEAD_VARIANT = args.head_variant
    SAVE_PREDICTIONS = args.save_predictions
    LOSS = args.loss
    if N_CLASSES > 2 and N_OBS == 1:
        print("[WARNING] N_CLASSES > 2 with a single observable. The 10-class "
              "diagnostic showed this is the measurement bottleneck, not a "
              "neutral choice. Proceeding only because it was requested "
              "explicitly.", flush=True)
    print(f"[MULTICLASS] n_classes={N_CLASSES} n_obs={N_OBS} "
          f"head_variant={HEAD_VARIANT}", flush=True)
    print(f"[LOSS] {LOSS} "
          + ("(softmax cross-entropy, Long et al. eq 21 -- DEVIATION from "
             "v6-v10, which used bce)" if LOSS == "ce" else
             "(sigmoid + binary cross-entropy on one-hot -- unchanged from "
             "v6-v10)"), flush=True)

    # ── DETERMINISM (v10) ────────────────────────────────────────────────
    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        print("[DETERMINISM] torch deterministic algorithms ENABLED. Two runs "
              "with the same seed should now agree exactly; if they do not, "
              "something else is injecting nondeterminism and must be found "
              "before the grid is trusted.", flush=True)
    else:
        print("[DETERMINISM] OFF -- expect ~0.8 pp run-to-run variation at "
              "fixed seed. Disclose this in any result derived from this run.",
              flush=True)

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

    # ── PROTOCOL READBACK ────────────────────────────────────────────────
    # The v10 PBS script echoed a HARDCODED "[PROTOCOL v10] 5000/1250/2000,
    # 200 epochs, ..." line, and the handover named that line as the
    # integrity check to rely on once REDUCED-SCALE started firing. A
    # hardcoded echo prints the same text whether or not those are the
    # values Python actually resolved, so it is an assertion, not a
    # readback, and it cannot fail. That is the exact defect shape the
    # integrity gate exists to catch. This prints the resolved variables
    # themselves, so a mismatch between intent and behaviour is visible.
    print(f"[PROTOCOL RESOLVED] n_train={N_TRAIN} n_val={N_VAL} "
          f"n_test={N_TEST} max_epochs={MAX_EPOCHS} patience={PATIENCE} "
          f"min_epochs_quantum={MIN_EPOCHS_QUANTUM_FRONTEND} "
          f"batch_size={BATCH_SIZE} lr={LR} weight_decay={WEIGHT_DECAY} "
          f"dropout={DROPOUT_RATE} n_classes={N_CLASSES} n_obs={N_OBS} "
          f"head_variant={HEAD_VARIANT} loss={LOSS} "
          f"qnn_backend={QNN_BACKEND} "
          f"deterministic={bool(args.deterministic)} "
          f"save_predictions={bool(args.save_predictions)} "
          f"classes={list(CLASS_OVERRIDE) if CLASS_OVERRIDE else 'default'}",
          flush=True)

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    print(f"[CODE VERSION] {CODE_VERSION}", flush=True)
    print(f"Run: model={args.model} dataset={args.dataset} seed={args.seed} "
          f"noise_alpha={args.noise_alpha} batch_size={BATCH_SIZE} lr={LR} "
          f"weight_decay={WEIGHT_DECAY} dropout={DROPOUT_RATE}",
          flush=True)
    # INTEGRITY (v8): print the class pair the TRAINER actually received,
    # not just the one the PBS script thinks it passed. The v7 defect where
    # `qsub -v CLASSES=0,6` silently became CLASSES=0 was invisible from the
    # submit side; it is only detectable here.
    print(f"[CLASS PAIR] trainer received classes="
          f"{list(CLASS_OVERRIDE) if CLASS_OVERRIDE else 'default (dataset built-in)'}",
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
        "n_params": expected_params(args.model, N_CLASSES, N_OBS),
        "n_classes": N_CLASSES,
        "n_obs": N_OBS,
        "head_variant": HEAD_VARIANT,
        "loss": LOSS,
        "code_version": CODE_VERSION,
        "protocol": {
            "code_version": CODE_VERSION,
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


# V135_M5_PATCH ---------------------------------------------------------
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
# end # V135_M5_PATCH


# V14_SPLIT_AUDIT -------------------------------------------------------
# v14: audit what load_dataset() returned. Calls the original with the original
# arguments and returns its result untouched. RNG states are saved and restored.
import hashlib as _v14_hashlib
import json as _v14_json
import os as _v14_os
import random as _v14_random
import numpy as _v14_np
import torch as _v14_torch

_v14_orig_load_dataset = load_dataset


def _v14_as_dataset(o):
    d = getattr(o, "dataset", None)            # DataLoader -> its dataset
    if d is not None and hasattr(d, "__getitem__") and hasattr(d, "__len__"):
        return d
    if hasattr(o, "__getitem__") and hasattr(o, "__len__") and not isinstance(o, (str, bytes, dict)):
        return o
    return None


def _v14_items(ds):
    hs, ys = [], []
    for i in range(len(ds)):
        it = ds[i]
        x, y = it[0], it[1]
        xb = _v14_torch.as_tensor(x).detach().cpu().contiguous().to(_v14_torch.float32).numpy().tobytes()
        hs.append(_v14_hashlib.blake2b(xb, digest_size=10).hexdigest())
        yt = _v14_torch.as_tensor(y)
        ys.append(int(yt.argmax()) if yt.numel() > 1 else int(yt))
    return hs, ys


def _v14_set_hash(hs):
    return _v14_hashlib.blake2b("".join(sorted(hs)).encode(), digest_size=8).hexdigest()


def _v14_audit(out):
    if _v14_os.environ.get("V14_NO_AUDIT") == "1":
        print("[V14] SPLIT audit disabled (V14_NO_AUDIT=1)", flush=True)
        return
    st = (_v14_random.getstate(), _v14_np.random.get_state(), _v14_torch.get_rng_state(),
          _v14_torch.cuda.get_rng_state_all() if _v14_torch.cuda.is_available() else None)
    try:
        parts = list(out) if isinstance(out, (tuple, list)) else [out]
        dss = [d for d in (_v14_as_dataset(p) for p in parts) if d is not None][:3]
        if len(dss) != 3:
            print(f"[V14] SPLIT-AUDIT FAILED: expected 3 splits from load_dataset, found {len(dss)} "
                  f"(return type {type(out).__name__})", flush=True)
            return
        dump = {}
        sets = {}
        for name, ds in zip(("train", "val", "test"), dss):
            hs, ys = _v14_items(ds)
            labs = sorted(set(ys))
            pc = [ys.count(c) for c in labs]
            sets[name] = set(hs)
            print(f"[V14] SPLIT {name:<5} n={len(ys)} labels={','.join(map(str, labs))} "
                  f"per_class={','.join(map(str, pc))} min={min(pc) if pc else 0} max={max(pc) if pc else 0} "
                  f"hash={_v14_set_hash(hs)}", flush=True)
            dump[name] = {"hashes": hs, "labels": ys}
        print(f"[V14] SPLIT overlap train&val={len(sets['train'] & sets['val'])} "
              f"train&test={len(sets['train'] & sets['test'])} val&test={len(sets['val'] & sets['test'])}",
              flush=True)
        p = _v14_os.environ.get("V14_SPLIT_DUMP")
        if p:
            with open(p, "w") as fh:
                _v14_json.dump(dump, fh)
    finally:
        _v14_random.setstate(st[0]); _v14_np.random.set_state(st[1]); _v14_torch.set_rng_state(st[2])
        if st[3] is not None:
            _v14_torch.cuda.set_rng_state_all(st[3])
    if _v14_os.environ.get("V14_SPLIT_ONLY") == "1":
        print("[V14] V14_SPLIT_ONLY=1: exiting after the split audit", flush=True)
        import sys as _s
        _s.stdout.flush(); _s.stderr.flush()
        _v14_os._exit(0)


def load_dataset(*args, **kwargs):
    out = _v14_orig_load_dataset(*args, **kwargs)
    _v14_audit(out)
    return out
# end # V14_SPLIT_AUDIT


# ═══════════════════════════════════════════════════════════════════════════════
# V15_FE_NOISE  (inserted by patch_v15_noise.py -- v15 IBM-calibrated front-end noise)
# Active only when the environment variable V15_ALPHA is set (run15.pbs sets it).
# Wraps build_model / load_dataset by name; changes no model, data or training code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v15_atexit
import glob as _v15_glob
import json as _v15_json
import os as _v15_os
import random as _v15_random
import sys as _v15_sys
import time as _v15_time

_V15 = dict(active=_v15_os.environ.get("V15_ALPHA") is not None, t0=_v15_time.time(),
            model=None, noise=None, test=None, name=None, seed=None, registered=False)


def _v15_rng_save():
    import numpy as _np
    import torch as _t
    st = dict(py=_v15_random.getstate(), np=_np.random.get_state(), t=_t.get_rng_state())
    if _t.cuda.is_available():
        st["cuda"] = _t.cuda.get_rng_state_all()
    return st


def _v15_rng_restore(st):
    import numpy as _np
    import torch as _t
    _v15_random.setstate(st["py"]); _np.random.set_state(st["np"]); _t.set_rng_state(st["t"])
    if "cuda" in st:
        _t.cuda.set_rng_state_all(st["cuda"])


def _v15_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_v15_sys.argv[1:])[0]


def _v15_install(model, name, seed):
    import torch as _t
    import quanv_noise_v15 as _qn
    from quanv_torch import TrainableQuanvFrontEndTorch as _C4, FrozenHendersonFrontEndTorch as _HE
    alpha = float(_v15_os.environ["V15_ALPHA"])
    calib = _qn.load_calib(_v15_os.environ["V15_CALIB"])
    fes = [m for m in model.modules() if isinstance(m, (_C4, _HE))]
    if name == "M4":
        if fes:
            _v15_sys.exit(f"[V15] FATAL: M4 has {len(fes)} quantum front-end(s); expected none")
        print(f"[V15] NOISE installed model=M4 kind=none alpha={alpha:g} calib={calib['sha256'][:12]} "
              f"(no quantum front-end: noise has nothing to act on)", flush=True)
        return None
    if len(fes) != 1:
        _v15_sys.exit(f"[V15] FATAL: model {name} has {len(fes)} quantum front-ends; expected exactly 1")
    fe = fes[0]
    kind = "henderson" if isinstance(fe, _HE) else "circuit4"
    st = _v15_rng_save()
    try:
        nz = _qn.FrontEndNoise(fe, kind, calib, seed=seed)
        if kind == "henderson":
            ok = nz.decomp_err < 1e-5
            print(f"[V15] NOISE selfcheck M1 RandomLayers decomposition == stored unitary: "
                  f"max|d|={nz.decomp_err:.2e} {'PASS' if ok else 'FAIL'}", flush=True)
            if not ok:
                _v15_sys.exit("[V15] FATAL: RandomLayers decomposition does not reproduce the front-end unitary")
        dev = next(iter(list(fe.parameters()) + list(fe.buffers()))).device
        g = _t.Generator().manual_seed(1234)
        x = _t.rand(4, 1, 28, 28, generator=g).to(dev)
        with _t.no_grad():
            d = (nz.ptm_path_at_zero(x) - nz._orig(x)).abs().max().item()
        ok = d < 1e-5
        print(f"[V15] NOISE selfcheck PTM path at alpha=0 vs exact forward: max|d|={d:.2e} "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            _v15_sys.exit("[V15] FATAL: noise path at alpha=0 does not reproduce the exact front-end")
        nz.set_alpha(alpha)
    finally:
        _v15_rng_restore(st)
    summ = nz.NP.summary() if nz.NP is not None else "noiseless (original forward)"
    print(f"[V15] NOISE installed model={name} kind={kind} alpha={alpha:g} calib={calib['sha256'][:12]} "
          f"snapshot={calib['last_update_date'][:10]} {summ}", flush=True)
    return nz


_v15_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    model = _v15_build_model_orig(*args, **kwargs)
    if _V15["active"]:
        name = args[0] if args else kwargs.get("model_name", kwargs.get("name"))
        seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
        if seed is None:
            seed = _v15_args().seed
        nz = _v15_install(model, name, seed)
        _V15.update(model=model, name=name, seed=seed, noise=nz)
        _V15.setdefault("built", []).append((model, nz))
        if not _V15["registered"]:
            _v15_atexit.register(_v15_xeval)
            _V15["registered"] = True
    return model


_v15_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v15_load_dataset_orig(*args, **kwargs)
    if _V15["active"] and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V15["test"] = res[2]
    return res


def _v15_predict(model, ds, bs=250):
    import torch as _t
    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    preds, labels = [], []
    loader = _t.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    with _t.no_grad():
        for batch in loader:
            x, y = batch[0], batch[1]
            out = model(x.to(dev))
            preds += out.argmax(1).cpu().tolist()
            labels += [int(v) for v in (y.tolist() if hasattr(y, "tolist") else y)]
    model.train(was_training)
    return preds, labels


def _v15_saved_preds(j, n):
    for k in ("predictions", "test_predictions", "preds", "y_pred", "test_preds"):
        v = j.get(k)
        if isinstance(v, list) and len(v) == n:
            if v and isinstance(v[0], list):
                return k, [max(range(len(r)), key=r.__getitem__) for r in v]
            return k, [int(p) for p in v]
    return None, None


def _v15_xeval():
    import torch as _t
    try:
        a = _v15_args()
        model, nz, ds = _V15["model"], _V15["noise"], _V15["test"]
        M, S = a.model or _V15["name"], a.seed if a.seed is not None else _V15["seed"]
        alpha_tr = float(_v15_os.environ["V15_ALPHA"])
        evals = [float(x) for x in _v15_os.environ.get("V15_EVAL_ALPHAS", "").split()]
        if model is None or ds is None or a.output_dir is None:
            print("[V15] XEVAL skipped: model, test split or --output_dir not captured", flush=True)
            return
        cands = [f for f in _v15_glob.glob(_v15_os.path.join(a.output_dir, f"result_{M}_*seed{S}_*.json"))
                 if _v15_os.path.getmtime(f) >= _V15["t0"] - 1]
        if not cands:
            print(f"[V15] XEVAL skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v15_os.path.getmtime)
        j = _v15_json.load(open(jf))
        st = _v15_rng_save()
        if nz is not None:
            nz.set_alpha(alpha_tr)
        preds, labels = _v15_predict(model, ds)
        key, saved = _v15_saved_preds(j, len(preds))
        classes = [int(c) for c in (a.classes or "").replace(" ", "").split(",") if c != ""]

        def same(p):
            if saved is None:
                return False
            if p == saved:
                return True
            return len(classes) > max(p) and [classes[i] for i in p] == saved

        source = "memory"
        match = same(preds)
        if not match:                       # train_gadi built more than one model: try each, latest first
            for k_, (m_, nz_) in enumerate(reversed(_V15.get("built", [])[:-1])):
                if nz_ is not None:
                    nz_.set_alpha(alpha_tr)
                p_, l_ = _v15_predict(m_, ds)
                if same(p_):
                    model, nz, preds, labels = m_, nz_, p_, l_
                    source, match = "memory(model %d)" % k_, True
                    break
        if not match and a.checkpoint_dir and _v15_os.path.isdir(a.checkpoint_dir):
            for ck in sorted(_v15_glob.glob(_v15_os.path.join(a.checkpoint_dir, "*.pt*"))):
                try:
                    obj = _t.load(ck, map_location=next(model.parameters()).device)
                except Exception:  # noqa: BLE001
                    continue
                sd = None
                if isinstance(obj, dict):
                    for k in ("model_state_dict", "state_dict", "model", "best_state", "best_model_state"):
                        if isinstance(obj.get(k), dict):
                            sd = obj[k]; break
                    if sd is None and obj and all(hasattr(v, "shape") for v in obj.values()):
                        sd = obj
                if sd is None:
                    continue
                try:
                    model.load_state_dict(sd, strict=True)
                except Exception:  # noqa: BLE001
                    continue
                preds, labels = _v15_predict(model, ds)
                if same(preds):
                    source, match = "ckpt:" + _v15_os.path.basename(ck), True
                    break
        acc_tr = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        jacc = float(j.get("test_acc", float("nan")))
        acc_ok = abs(acc_tr - jacc) < 1e-6 or abs(100 * acc_tr - jacc) < 1e-4   # result JSON may hold float32
        res = dict(model=M, seed=S, alpha_train=alpha_tr, eval_alphas=evals,
                   calib_sha=_v15_json.load(open(_v15_os.environ["V15_CALIB"]))["sha256"],
                   result_json=_v15_os.path.basename(jf), predictions_key=key, source=source,
                   selfcheck="MATCH" if (match and acc_ok) else "MISMATCH", n_test=len(labels),
                   labels=labels, acc={}, predictions={})
        for al in evals:
            if nz is not None:
                nz.set_alpha(al)
            p, _ = _v15_predict(model, ds) if (nz is not None or al == evals[0]) else (res["predictions"][f"{evals[0]:g}"], None)
            res["predictions"][f"{al:g}"] = p
            res["acc"][f"{al:g}"] = sum(int(q == y) for q, y in zip(p, labels)) / len(labels)
        if nz is not None:
            nz.set_alpha(alpha_tr)
        _v15_rng_restore(st)
        out = _v15_os.path.join(a.output_dir, f"v15_xeval_{M}_seed{S}.json")
        with open(out, "w") as f:
            _v15_json.dump(res, f)
        accs = " ".join(f"{k}={v:.4f}" for k, v in res["acc"].items())
        print(f"[V15] XEVAL model={M} seed={S} alpha_train={alpha_tr:g} source={source} "
              f"selfcheck={res['selfcheck']} (acc at alpha_train {acc_tr:.4f} vs result JSON {jacc}) "
              f"acc_by_eval_alpha: {accs}", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V15] XEVAL FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V15_FE_NOISE ═══════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# V16A_HEAD_SAVEZ  (inserted by patch_v16a.py -- v16a M2c head + --save_z)
# Active only with --v16_head / --save_z / --v16_sidecar (removed from sys.argv here).
# Wraps build_model / expected_params / load_dataset by name; edits no existing code.
# ═══════════════════════════════════════════════════════════════════════════════
import atexit as _v16_atexit
import glob as _v16_glob
import json as _v16_json
import os as _v16_os
import sys as _v16_sys
import time as _v16_time

_V16 = dict(head=None, save_z=False, sidecar=False, in_build=False, model=None, name=None, seed=None,
            test=None, info=None, registered=False, t0=_v16_time.time(), argv=list(_v16_sys.argv))


def _v16_strip_argv():
    out, i, av = [], 0, _v16_sys.argv
    while i < len(av):
        t = av[i]
        if t == "--save_z":
            _V16["save_z"] = True
        elif t == "--v16_sidecar":
            _V16["sidecar"] = True
        elif t == "--v16_head":
            _V16["head"] = av[i + 1] if i + 1 < len(av) else ""
            i += 1
        elif t.startswith("--v16_head="):
            _V16["head"] = t.split("=", 1)[1]
        else:
            out.append(t)
        i += 1
    _v16_sys.argv[:] = out
    if _V16["head"] not in (None, "m2c"):
        _v16_sys.exit(f"[V16] FATAL: --v16_head {_V16['head']!r} (only 'm2c' exists)")


_v16_strip_argv()


def _v16_active():
    return bool(_V16["head"] or _V16["save_z"] or _V16["sidecar"])


def _v16_label():
    return "M2c" if _V16["head"] == "m2c" else _V16["name"]


def _v16_args():
    import argparse as _ap
    p = _ap.ArgumentParser(add_help=False)
    for a in ("--model", "--output_dir", "--checkpoint_dir", "--classes"):
        p.add_argument(a, default=None)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_known_args(_V16["argv"][1:])[0]


_v16_build_model_orig = build_model  # noqa: F821


def build_model(*args, **kwargs):  # noqa: F811
    _V16["in_build"] = True
    try:
        model = _v16_build_model_orig(*args, **kwargs)
    finally:
        _V16["in_build"] = False
    if not _v16_active():
        return model
    name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
    seed = kwargs.get("seed", args[1] if len(args) > 1 else None)
    _V16.update(model=model, name=name, seed=seed)
    if _V16["head"] == "m2c":
        if name != "M2":
            _v16_sys.exit(f"[V16] FATAL: --v16_head m2c needs --model M2 (got {name})")
        import v16_tools as _vt
        try:
            info = _vt.install_m2c(model, seed if seed is not None else 0)
        except _vt.V16Error as e:
            _v16_sys.exit(f"[V16] FATAL: M2c surgery refused: {e}")
        table = {(5, 15): 1665, (10, 15): 1730}.get((info["K"], info["n_obs"]))
        ok = info["delta_ok"] and (table is None or info["params_m2c"] == table)
        info["table"] = table
        _V16["info"] = info
        print(f"[V16] HEAD m2c installed model=M2 seed={seed} qnn={info['qnn']} readout={info['readout']} "
              f"z={info['z_layer']} wiring=identity params_M2={info['params_m2']} params_M2c={info['params_m2c']} "
              f"expected_M2c={table} {'VERIFIED' if ok else 'MISMATCH'}", flush=True)
        if not ok:
            _v16_sys.exit("[V16] FATAL: M2c parameter count is not the pre-registered one")
    if not _V16["registered"]:
        _v16_atexit.register(_v16_finish)
        _V16["registered"] = True
    return model


if "expected_params" in globals():
    _v16_ep_orig = expected_params  # noqa: F821

    def expected_params(*args, **kwargs):  # noqa: F811
        n = _v16_ep_orig(*args, **kwargs)
        if _V16["head"] == "m2c" and not _V16["in_build"]:
            name = args[0] if args else kwargs.get("model_name", kwargs.get("name", kwargs.get("model")))
            if name == "M2":
                ints = [v for v in list(args[1:]) + list(kwargs.values())
                        if isinstance(v, int) and not isinstance(v, bool)]
                K, W = (ints[0], ints[1]) if len(ints) >= 2 else (N_CLASSES, N_OBS)  # noqa: F821
                n = n - 6 - (W + 1) * K + 13 * K
        return n


_v16_load_dataset_orig = load_dataset  # noqa: F821


def load_dataset(*args, **kwargs):  # noqa: F811
    res = _v16_load_dataset_orig(*args, **kwargs)
    if _v16_active() and isinstance(res, (tuple, list)) and len(res) >= 3:
        _V16["test"] = res[2]
    return res


def _v16_saved_preds(j, n):
    for k in ("predictions", "test_predictions", "preds", "y_pred", "test_preds"):
        v = j.get(k)
        if isinstance(v, list) and len(v) == n:
            if v and isinstance(v[0], list):
                return k, [max(range(len(r)), key=r.__getitem__) for r in v]
            return k, [int(p) for p in v]
    return None, None


def _v16_finish():
    import numpy as _np
    import torch as _t
    import v16_tools as _vt
    try:
        a = _v16_args()
        model, ds = _V16["model"], _V16["test"]
        M, S = _v16_label(), (a.seed if a.seed is not None else _V16["seed"])
        if model is None or ds is None or a.output_dir is None:
            print("[V16] RUN skipped: model, test split or --output_dir not captured", flush=True)
            return
        arch = a.model or _V16["name"]
        cands = [f for f in _v16_glob.glob(_v16_os.path.join(a.output_dir, f"result_{arch}_*seed{S}_*.json"))
                 if _v16_os.path.getmtime(f) >= _V16["t0"] - 1]
        if not cands:
            print(f"[V16] RUN skipped: no result JSON written by this run in {a.output_dir}", flush=True)
            return
        jf = max(cands, key=_v16_os.path.getmtime)
        j = _v16_json.load(open(jf))
        st = _vt.rng_save()
        classes = [int(c) for c in (a.classes or "").replace(" ", "").split(",") if c != ""]
        preds, labels, z = _vt.predict_with_z(model, ds)
        key, saved = _v16_saved_preds(j, len(preds))

        def enc(p):
            if saved is None:
                return None
            if p == saved:
                return "index"
            if classes and len(classes) > max(p) and [classes[i] for i in p] == saved:
                return "class"
            return None

        source, encoding = "memory", enc(preds)
        if encoding is None and a.checkpoint_dir and _v16_os.path.isdir(a.checkpoint_dir):
            for ck in sorted(_v16_glob.glob(_v16_os.path.join(a.checkpoint_dir, "*.pt*"))):
                try:
                    obj = _t.load(ck, map_location=next(model.parameters()).device)
                except Exception:  # noqa: BLE001
                    continue
                sd = None
                if isinstance(obj, dict):
                    for k in ("model_state_dict", "state_dict", "model", "best_state", "best_model_state"):
                        if isinstance(obj.get(k), dict):
                            sd = obj[k]; break
                    if sd is None and obj and all(hasattr(v, "shape") for v in obj.values()):
                        sd = obj
                if sd is None:
                    continue
                try:
                    model.load_state_dict(sd, strict=True)
                except Exception:  # noqa: BLE001
                    continue
                preds, labels, z = _vt.predict_with_z(model, ds)
                encoding = enc(preds)
                if encoding is not None:
                    source = "ckpt:" + _v16_os.path.basename(ck)
                    break
        _vt.rng_restore(st)
        acc = sum(int(p == y) for p, y in zip(preds, labels)) / len(labels)
        jacc = float(j.get("test_acc", float("nan")))
        acc_ok = abs(acc - jacc) < 1e-6 or abs(100 * acc - jacc) < 1e-4
        match = encoding is not None and acc_ok
        from closed_form_zz_v16 import FEATURE_NAMES as _FN
        info = _V16["info"] or {}
        res = dict(model=M, arch=arch, head=_V16["head"] or "native", seed=S, result_json=_v16_os.path.basename(jf),
                   predictions_key=key, pred_encoding=encoding, source=source,
                   selfcheck="MATCH" if match else "MISMATCH", n_test=len(labels), acc=acc, test_acc_json=jacc,
                   params=_vt.n_params(model), m2c=info, features=_FN if _V16["head"] == "m2c" else None,
                   labels=labels, classes=classes, z_saved=False)
        if _V16["save_z"]:
            zf = _v16_os.path.join(a.output_dir, f"v16_z_{M}_seed{S}.npz")
            _np.savez(zf, z=z, labels=_np.asarray(labels), preds=_np.asarray(preds))
            res.update(z_saved=True, z_file=_v16_os.path.basename(zf), z_shape=list(z.shape), z_dtype=str(z.dtype),
                       z_sha256=_vt.sha(z), z_min=z.min(0).tolist(), z_max=z.max(0).tolist(),
                       z_mean=z.mean(0).tolist())
            print(f"[V16] SAVEZ model={M} seed={S} n={z.shape[0]} dim={z.shape[1]} file={_v16_os.path.basename(zf)} "
                  f"sha={res['z_sha256'][:12]} selfcheck={res['selfcheck']}", flush=True)
        with open(_v16_os.path.join(a.output_dir, f"v16_run_{M}_seed{S}.json"), "w") as f:
            _v16_json.dump(res, f)
        print(f"[V16] RUN model={M} arch={arch} head={res['head']} seed={S} params={res['params']} source={source} "
              f"encoding={encoding} selfcheck={res['selfcheck']} (acc {acc:.4f} vs result JSON {jacc})", flush=True)
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"[V16] RUN FAILED: {type(e).__name__}: {e}", flush=True)
# ═══ end V16A_HEAD_SAVEZ ════════════════════════════════════════════════════════


if __name__ == "__main__":
    main()
