"""
QCQ-CNN Faithful Replication — Long et al. (2025)
MNIST 3 vs 5 | 5000 train / 1000 test | seed=42
Run via PBS: qsub run_replication.pbs
"""

import os
os.environ["PYTHONPATH"] = ""
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import matplotlib
matplotlib.use("Agg")  # Headless backend — no display needed on compute node

# ================================================================
#  CONFIGURATION — faithful to Long et al. (2025)
# ================================================================
TARGET_LABELS   = [3, 5]
PER_CLASS_TRAIN = 2500        # 5000 total (Long et al.)
PER_CLASS_TEST  = 500         # 1000 total (Long et al.)

Q_FILTER_SIZE   = 2
Q_FILTER_LAYERS = 1
Q_STRIDE        = 2
Q_PADDING       = 0

TF_EPOCHS       = 100
TF_LR           = 0.001
TF_BATCH_SIZE   = 4

QCQ_EPOCHS      = 100
QCQ_LR          = 0.001      # Long et al. value (was 0.0005 in smoke test)
QCQ_SEED        = 42

OUTDIR = "results_replication"
os.makedirs(OUTDIR, exist_ok=True)

TRAIN_CACHE = f"MNIST_op_train_{TARGET_LABELS[0]}v{TARGET_LABELS[1]}_{PER_CLASS_TRAIN}pc.npy"
TEST_CACHE  = f"MNIST_op_test_{TARGET_LABELS[0]}v{TARGET_LABELS[1]}_{PER_CLASS_TEST}pc.npy"

print("=" * 68)
print("  QCQ-CNN FAITHFUL REPLICATION — Long et al. (2025)")
print(f"  MNIST {TARGET_LABELS[0]} vs {TARGET_LABELS[1]}  |  "
      f"{PER_CLASS_TRAIN*2} train / {PER_CLASS_TEST*2} test  |  seed={QCQ_SEED}")
print(f"  TF: {TF_EPOCHS}ep lr={TF_LR} batch={TF_BATCH_SIZE}  |  "
      f"QCQ: {QCQ_EPOCHS}ep lr={QCQ_LR}")
print("=" * 68)

# ================================================================
#  IMPORTS
# ================================================================
import math, time, random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import (Module, Conv2d, Linear, Dropout2d,
                       CrossEntropyLoss, MaxPool2d, Flatten)
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_

import pennylane as qml
from pennylane import numpy as qnp
from pennylane.templates import RandomLayers

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Resizing

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit_machine_learning.neural_networks import EstimatorQNN
from qiskit_machine_learning.connectors import TorchConnector

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

print("All imports OK")

# ================================================================
#  DATA LOADING
# ================================================================
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()

train_mask = np.isin(train_labels, TARGET_LABELS)
test_mask  = np.isin(test_labels,  TARGET_LABELS)
train_images, train_labels = train_images[train_mask], train_labels[train_mask]
test_images,  test_labels  = test_images[test_mask],   test_labels[test_mask]

train_labels = (train_labels == TARGET_LABELS[1]).astype(np.int32)
test_labels  = (test_labels  == TARGET_LABELS[1]).astype(np.int32)

def select_balanced_subset(images, labels, per_class):
    sel_imgs, sel_labs = [], []
    for v in [0, 1]:
        idx = np.where(labels == v)[0][:per_class]
        sel_imgs.append(images[idx])
        sel_labs.append(labels[idx])
    return np.concatenate(sel_imgs), np.concatenate(sel_labs)

def select_balanced_subset_test(images, labels, per_class):
    sel_imgs, sel_labs = [], []
    for v in [0, 1]:
        idx = np.where(labels == v)[0][:per_class]
        sel_imgs.append(images[idx])
        sel_labs.append(labels[idx])
    return np.concatenate(sel_imgs), np.concatenate(sel_labs)

train_images, train_labels = select_balanced_subset(train_images, train_labels, PER_CLASS_TRAIN)
test_images,  test_labels  = select_balanced_subset_test(test_images, test_labels, PER_CLASS_TEST)

train_images = (train_images / 255.0)[..., np.newaxis]
test_images  = (test_images  / 255.0)[..., np.newaxis]

print(f"Train: {train_images.shape} {train_labels.shape}")
print(f"Test : {test_images.shape}  {test_labels.shape}")

# ================================================================
#  QUANTUM FEATURE EXTRACTION (quanvolutional layer)
# ================================================================
class Quanvolutional_Layer:
    def __init__(self, size_filter, layers_filter, stride_filter, padding):
        self.size_filter   = size_filter
        self.layers_filter = layers_filter
        self.stride_filter = stride_filter
        self.padding       = padding
        self.dev           = qml.device("default.qubit", wires=size_filter * size_filter)
        self.rand_params   = qnp.random.uniform(
            high=2 * qnp.pi,
            size=(layers_filter, size_filter * size_filter)
        )

    def creator(self, phi):
        @qml.qnode(self.dev)
        def circuit(phi):
            op_list = []
            for j in range(self.size_filter * self.size_filter):
                qml.RY(qnp.pi * phi[j], wires=j)
            for k in self.rand_params:
                tmp = qnp.expand_dims(k, axis=0)
                RandomLayers(tmp, wires=list(range(self.size_filter * self.size_filter)))
                for j in range(self.size_filter * self.size_filter):
                    op_list.append(qml.expval(qml.PauliZ(j)))
            return op_list
        op_list = circuit(phi)
        op_list = qnp.array(op_list)
        return list(qnp.reshape(op_list, (self.layers_filter, self.size_filter * self.size_filter)))

    def call(self, image):
        calc = math.floor((image.shape[1] + 2 * self.padding - self.size_filter) / self.stride_filter) + 1
        collection = []
        for idx, img in enumerate(image):
            out = qnp.zeros((calc, calc, self.layers_filter * self.size_filter * self.size_filter))
            if (idx + 1) % 100 == 0:
                print(f"  Quantum encoding: {idx+1}/{len(image)}")
            for j in range(0, img.shape[0], self.stride_filter):
                for k in range(0, img.shape[1], self.stride_filter):
                    if (j + self.size_filter > img.shape[0]) or (k + self.size_filter > img.shape[1]):
                        continue
                    phi = [img[l][m][0] for l in range(j, j+self.size_filter) for m in range(k, k+self.size_filter)]
                    q_results = self.creator(phi)
                    ctr = 0
                    for x in q_results:
                        for y in range(len(x)):
                            out[j//self.stride_filter, k//self.stride_filter, ctr] = x[y]
                            ctr += 1
            collection.append(out)
        return collection

if not (os.path.exists(TRAIN_CACHE) and os.path.exists(TEST_CACHE)):
    print("Computing quantum features (this is the slow step ~30-60 min)...")
    np.random.seed(QCQ_SEED)
    Quan2D = Quanvolutional_Layer(Q_FILTER_SIZE, Q_FILTER_LAYERS, Q_STRIDE, Q_PADDING)
    print("  Encoding training set...")
    op_train = np.asarray(Quan2D.call(train_images), dtype=np.float32)
    print("  Encoding test set...")
    op_test  = np.asarray(Quan2D.call(test_images),  dtype=np.float32)
    np.save(TRAIN_CACHE, op_train)
    np.save(TEST_CACHE,  op_test)
    print(f"Cached to {TRAIN_CACHE} / {TEST_CACHE}")
else:
    print(f"Loading cached quantum features: {TRAIN_CACHE}")
    op_train = np.load(TRAIN_CACHE)
    op_test  = np.load(TEST_CACHE)

print(f"op_train: {op_train.shape}  op_test: {op_test.shape}")

# ================================================================
#  TF SEEDS
# ================================================================
np.random.seed(QCQ_SEED)
tf.random.set_seed(QCQ_SEED)

# ================================================================
#  TF MODEL 1: Q+Dense (quantum features → dense)
# ================================================================
def MyModel():
    model = keras.models.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(
        optimizer=keras.optimizers.SGD(learning_rate=TF_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

print("\n--- Training Q+Dense ---")
q_model = MyModel()
q_history = q_model.fit(
    op_train, train_labels,
    validation_data=(op_test, test_labels),
    batch_size=TF_BATCH_SIZE, epochs=TF_EPOCHS, verbose=2,
)

# ================================================================
#  TF MODEL 2: Classical Dense (raw pixels → dense)
# ================================================================
print("\n--- Training Classical Dense ---")
c_model = MyModel()
c_history = c_model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    batch_size=TF_BATCH_SIZE, epochs=TF_EPOCHS, verbose=2,
)

# ================================================================
#  TF MODEL 3: QC-CNN (quantum features → CNN)
# ================================================================
def QCCNNModel():
    model = models.Sequential([
        layers.Conv2D(4, (3, 3), activation='relu', input_shape=(14, 14, 4)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=TF_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n--- Training QC-CNN ---")
qccnn_model = QCCNNModel()
qccnn_history = qccnn_model.fit(
    op_train, train_labels,
    validation_data=(op_test, test_labels),
    batch_size=TF_BATCH_SIZE, epochs=TF_EPOCHS, verbose=2,
)

# ================================================================
#  TF MODEL 4: Classical CNN (raw pixels → CNN)
# ================================================================
def ClassicalCNNModel():
    model = keras.models.Sequential([
        Resizing(14, 14),
        layers.Conv2D(4, (3, 3), activation='relu', input_shape=(14, 14, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(8, (3, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(learning_rate=TF_LR),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

print("\n--- Training Classical CNN ---")
cnn_model = ClassicalCNNModel()
cnn_history = cnn_model.fit(
    train_images, train_labels,
    validation_data=(test_images, test_labels),
    batch_size=TF_BATCH_SIZE, epochs=TF_EPOCHS, verbose=2,
)

# ================================================================
#  PYTORCH QCQ-CNN
# ================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"\nUsing device: {DEVICE}")

def create_qnn():
    feature_map = ZZFeatureMap(2)
    ansatz      = RealAmplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    return EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
    )

class Net(Module):
    def __init__(self, qnn):
        super().__init__()
        self.conv1   = Conv2d(4, 8, 3)
        self.conv2   = Conv2d(8, 16, 3)
        self.dropout = Dropout2d()
        self.fc2     = None
        self.qnn     = TorchConnector(qnn)
        self.fc3     = Linear(1, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = x.view(x.shape[0], -1)
        if self.fc2 is None:
            self.fc2 = Linear(x.shape[1], 2).to(x.device)
        x = self.fc2(x)
        x = self.qnn(x)
        x = self.fc3(x)
        return x

# Build PyTorch data loaders
op_train_py = np.transpose(np.load(TRAIN_CACHE), (0, 3, 1, 2))
op_test_py  = np.transpose(np.load(TEST_CACHE),  (0, 3, 1, 2))
op_train_tensor = torch.tensor(op_train_py)
op_test_tensor  = torch.tensor(op_test_py)
train_data = [(op_train_tensor[i], train_labels[i]) for i in range(len(op_train_tensor))]
test_data  = [(op_test_tensor[i],  test_labels[i])  for i in range(len(op_test_tensor))]
train_loader = DataLoader(train_data, shuffle=True, batch_size=1)
test_loader  = DataLoader(test_data,  shuffle=True, batch_size=1)

def train_qcq_cnn(train_loader, epochs, seed, lr):
    torch.manual_seed(seed); np.random.seed(seed)
    random.seed(seed); torch.cuda.manual_seed_all(seed)

    qnn       = create_qnn()
    model     = Net(qnn).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = CrossEntropyLoss().to(DEVICE)
    loss_list, acc_list = [], []

    model.train()
    start = time.time()
    for epoch in range(epochs):
        correct, total_loss = 0, []
        for data, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(data.to(DEVICE))
            loss   = loss_func(output, target.to(DEVICE).long())
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss.append(loss.item())
            correct += output.argmax(dim=1).eq(target.to(DEVICE)).sum().item()
        e_loss = sum(total_loss) / len(total_loss)
        e_acc  = 100. * correct / len(train_loader)
        loss_list.append(e_loss)
        acc_list.append(e_acc)
        print(f"[{epoch+1:3d}/{epochs}] Loss: {e_loss:.4f} | Accuracy: {e_acc:.2f}%")

    print(f"QCQ-CNN training time: {time.time()-start:.1f}s")
    return model, loss_list, acc_list

print("\n--- Training QCQ-CNN (PyTorch) ---")
model1, loss1, acc1 = train_qcq_cnn(train_loader, QCQ_EPOCHS, QCQ_SEED, QCQ_LR)

# ================================================================
#  EVALUATION
# ================================================================
model1.eval()
qcq_preds, qcq_true = [], []
with torch.no_grad():
    for data, target in test_loader:
        out = model1(data.to(DEVICE))
        qcq_preds.extend(out.argmax(dim=1).cpu().numpy())
        qcq_true.extend(target.numpy())
qcq_test_acc = float(np.mean(np.array(qcq_preds) == np.array(qcq_true)))

# ================================================================
#  SUMMARY PRINT
# ================================================================
SEP = "=" * 68
tf_models = {
    'Q+Dense':         q_history,
    'Classical Dense': c_history,
    'QC-CNN':          qccnn_history,
    'Classical CNN':   cnn_history,
}

print(f"\n{SEP}")
print("  EXPERIMENT SUMMARY  (Faithful Replication)")
print(f"  MNIST {TARGET_LABELS[0]} vs {TARGET_LABELS[1]}  |  "
      f"{PER_CLASS_TRAIN*2} train / {PER_CLASS_TEST*2} test  |  seed={QCQ_SEED}")
print(SEP)
print(f"\n{'Model':<22} {'Test Acc':>10}")
print("-" * 35)
for name, h in tf_models.items():
    print(f"{name:<22} {h.history['val_accuracy'][-1]:>10.4f}")
print(f"{'QCQ-CNN (PyTorch)':<22} {qcq_test_acc:>10.4f}")
print()
print(classification_report(qcq_true, qcq_preds,
      target_names=[f'Digit {TARGET_LABELS[0]}', f'Digit {TARGET_LABELS[1]}']))

# ================================================================
#  SAVE RESULTS
# ================================================================
# 1. Accuracy bar chart
model_names = list(tf_models.keys()) + ['QCQ-CNN']
final_accs  = [tf_models[n].history['val_accuracy'][-1] for n in tf_models] + [qcq_test_acc]
colors      = ['purple', 'steelblue', 'green', 'orange', 'red']

fig, ax = plt.subplots(figsize=(8, 4))
bars = ax.bar(model_names, final_accs, color=colors, edgecolor='black', alpha=0.85)
ax.axhline(0.5, color='gray', linestyle='--', linewidth=1, label='Chance (50%)')
for bar, val in zip(bars, final_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{val:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylim([0, 1.1]); ax.set_ylabel('Accuracy')
ax.set_title(f'Faithful Replication — MNIST 3v5 | {PER_CLASS_TRAIN*2} train | seed={QCQ_SEED}')
ax.legend(); plt.tight_layout()
plt.savefig(f"{OUTDIR}/results_bar_replication.png", dpi=150)
print(f"Saved {OUTDIR}/results_bar_replication.png")

# 2. Learning curves
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
col = {'Q+Dense': 'purple', 'Classical Dense': 'steelblue', 'QC-CNN': 'green', 'Classical CNN': 'orange'}
for name, h in tf_models.items():
    ax1.plot(h.history['val_accuracy'], label=name, color=col[name])
    ax2.plot(h.history['val_loss'],     label=name, color=col[name])
ax1.plot([a/100 for a in acc1], label='QCQ-CNN (train)', color='red', linestyle='--')
ax2.plot(loss1,                  label='QCQ-CNN (train)', color='red', linestyle='--')
ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1, label='Chance')
ax1.set_title('Accuracy per Epoch'); ax1.set_ylabel('Accuracy'); ax1.legend(fontsize=8); ax1.grid(True)
ax2.set_title('Loss per Epoch');     ax2.set_ylabel('Loss');     ax2.legend(fontsize=8); ax2.grid(True)
fig2.suptitle(f"MNIST 3v5 | {PER_CLASS_TRAIN*2} train | lr(QCQ)={QCQ_LR} | seed={QCQ_SEED}")
plt.tight_layout()
plt.savefig(f"{OUTDIR}/results_summary_replication.png", dpi=150)
print(f"Saved {OUTDIR}/results_summary_replication.png")

# 3. QCQ-CNN confusion matrix
cm     = confusion_matrix(qcq_true, qcq_preds)
cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100
fig3, ax4 = plt.subplots(figsize=(5, 4))
sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='Blues',
            xticklabels=[f'Pred {TARGET_LABELS[0]}', f'Pred {TARGET_LABELS[1]}'],
            yticklabels=[f'True {TARGET_LABELS[0]}', f'True {TARGET_LABELS[1]}'], ax=ax4)
ax4.set_title(f'QCQ-CNN Confusion Matrix (%, n={len(qcq_true)})')
plt.tight_layout()
plt.savefig(f"{OUTDIR}/qcqcnn_confusion_replication.png", dpi=150)
print(f"Saved {OUTDIR}/qcqcnn_confusion_replication.png")

# 4. Save numeric results to Excel
rows = []
for name, h in tf_models.items():
    rows.append({'Model': name, 'Test_Acc': h.history['val_accuracy'][-1],
                 'Train_Acc': h.history['accuracy'][-1]})
rows.append({'Model': 'QCQ-CNN', 'Test_Acc': qcq_test_acc, 'Train_Acc': acc1[-1]/100})
pd.DataFrame(rows).to_excel(f"{OUTDIR}/final_accuracies.xlsx", index=False)
print(f"Saved {OUTDIR}/final_accuracies.xlsx")

print(f"\n{SEP}")
print("  DONE — all results saved to ./{OUTDIR}/")
print(SEP)
