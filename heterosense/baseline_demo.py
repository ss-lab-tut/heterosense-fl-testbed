"""HeteroSense baseline demo: Local vs FedAvg benchmark."""
from __future__ import annotations

import os

import numpy as np
from collections import Counter
from heterosense._core._observation_model import ModalityBundle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STATES = ['ABSENT', 'STATIONARY', 'WALKING', 'TRANSITION', 'ABNORMAL']
STATE_TO_IDX = {s: i for i, s in enumerate(STATES)}
N_CLASSES = len(STATES)
N_POINTS_FIXED = 64   # fixed-N for point cloud sampling

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def _cross_entropy(logits: np.ndarray, labels: np.ndarray) -> float:
    probs = _softmax(logits)
    n = len(labels)
    return float(-np.log(probs[np.arange(n), labels] + 1e-9).mean())

def _accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Standard (macro-averaged) accuracy."""
    return float((logits.argmax(axis=-1) == labels).mean())


def _balanced_accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Balanced accuracy: mean per-class recall.

    More appropriate than standard accuracy when class frequencies differ
    (e.g., ABNORMAL events are rare relative to STATIONARY/WALKING).
    Equivalent to macro-averaged recall.
    """
    preds = logits.argmax(axis=-1)
    classes = np.unique(labels)
    recalls = []
    for c in classes:
        mask = labels == c
        if mask.sum() == 0:
            continue
        recalls.append((preds[mask] == c).mean())
    return float(np.mean(recalls)) if recalls else 0.0

def _sample_points(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed-N sampling: subsample or pad with zeros."""
    if len(pts) >= n:
        idx = rng.choice(len(pts), n, replace=False)
        return pts[idx]
    else:
        pad = np.zeros((n - len(pts), 3), dtype=np.float64)
        return np.vstack([pts, pad])

# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_pressure(
    bundles: list[ModalityBundle],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (X, y) for pressure-only model.
    X: (N, 256)  flattened 16x16 pressure map
    y: (N,)      state label
    Skips None pressure samples.
    """
    X, y = [], []
    for b in bundles:
        if b.pressure is None:
            continue
        X.append(b.pressure.flatten())
        y.append(STATE_TO_IDX[b.semantic_state])
    if not X:
        return None
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)

def prepare_pointcloud(
    bundles: list[ModalityBundle],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (X, y) for point-cloud-only model.
    X: (N, N_POINTS_FIXED*3)  sampled & flattened point cloud
    y: (N,)                   state label
    Skips None lidar samples.
    """
    X, y = [], []
    for b in bundles:
        if b.lidar is None:
            continue
        pts = _sample_points(b.lidar, N_POINTS_FIXED, rng)
        X.append(pts.flatten())
        y.append(STATE_TO_IDX[b.semantic_state])
    if not X:
        return None
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)

def prepare_fusion(
    bundles: list[ModalityBundle],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X, y) for late fusion model.
    X: (N, 256 + N_POINTS_FIXED*3)  concat of pressure + point cloud
    Missing modality -> zero vector
    y: (N,) state label
    """
    X, y = [], []
    for b in bundles:
        p = b.pressure.flatten() if b.pressure is not None \
            else np.zeros(256, dtype=np.float64)
        l = _sample_points(b.lidar, N_POINTS_FIXED, rng).flatten() \
            if b.lidar is not None \
            else np.zeros(N_POINTS_FIXED * 3, dtype=np.float64)
        X.append(np.concatenate([p, l]))
        y.append(STATE_TO_IDX[b.semantic_state])
    return np.array(X, dtype=np.float64), np.array(y, dtype=np.int64)

# ---------------------------------------------------------------------------
# Tiny MLP (numpy only)
# ---------------------------------------------------------------------------

class TinyMLP:
    """2-layer MLP: input -> hidden(64) -> N_CLASSES."""

    def __init__(self, input_dim: int, rng: np.random.Generator) -> None:
        scale1 = np.sqrt(2.0 / input_dim)
        scale2 = np.sqrt(2.0 / 64)
        self.W1 = rng.normal(0, scale1, (input_dim, 64))
        self.b1 = np.zeros(64)
        self.W2 = rng.normal(0, scale2, (64, N_CLASSES))
        self.b2 = np.zeros(N_CLASSES)
        self.lr = 0.01

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.h_pre = X @ self.W1 + self.b1
        self.h     = np.maximum(0, self.h_pre)   # ReLU
        self.logits = self.h @ self.W2 + self.b2
        return self.logits

    def backward(self, X: np.ndarray, labels: np.ndarray) -> None:
        n = len(labels)
        probs = _softmax(self.logits)
        probs[np.arange(n), labels] -= 1.0
        probs /= n

        dW2 = self.h.T @ probs
        db2 = probs.sum(axis=0)
        dh  = probs @ self.W2.T
        dh  *= (self.h_pre > 0)
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 30, batch_size: int = 128,
            rng: np.random.Generator = None) -> list[float]:
        losses = []
        n = len(X)
        for ep in range(epochs):
            idx = rng.permutation(n) if rng else np.arange(n)
            ep_loss = 0.0
            for start in range(0, n, batch_size):
                b_idx = idx[start:start + batch_size]
                Xb, yb = X[b_idx], y[b_idx]
                logits = self.forward(Xb)
                ep_loss += _cross_entropy(logits, yb) * len(yb)
                self.backward(Xb, yb)
            losses.append(ep_loss / n)
        return losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X).argmax(axis=-1)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baseline_demo(
    data: dict[str, list[ModalityBundle]],
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    all_bundles = [b for bundles in data.values() for b in bundles]

    # chance level
    labels_all = [STATE_TO_IDX[b.semantic_state] for b in all_bundles]
    most_common = Counter(labels_all).most_common(1)[0][1]
    chance = most_common / len(labels_all)

    print('HeteroSense-FL -- Baseline Demo')
    print('=' * 55)
    print(f'Total samples: {len(all_bundles)}')
    print(f'Chance level:  {chance:.3f} ({STATES[Counter(labels_all).most_common(1)[0][0]]})')
    print()

    # ① pressure-only
    print('① Pressure-only (pipeline check)')
    res = prepare_pressure(all_bundles)
    if res is None:
        print('   SKIP: no pressure data')
    else:
        X, y = res
        n = len(X)
        split = int(n * 0.8)
        idx = rng.permutation(n)
        Xtr, ytr = X[idx[:split]], y[idx[:split]]
        Xva, yva = X[idx[split:]], y[idx[split:]]
        model = TinyMLP(X.shape[1], rng)
        model.fit(Xtr, ytr, epochs=30, batch_size=128, rng=rng)
        acc = _accuracy(model.forward(Xva), yva)
        print(f'   samples={n}, val_acc={acc:.3f}, chance={chance:.3f}',
              '✅' if acc > chance else '❌')
    print()

    # ② point-cloud-only
    print('② Point-cloud-only (structured observation check)')
    res = prepare_pointcloud(all_bundles, rng)
    if res is None:
        print('   SKIP: no lidar data')
    else:
        X, y = res
        n = len(X)
        split = int(n * 0.8)
        idx = rng.permutation(n)
        Xtr, ytr = X[idx[:split]], y[idx[:split]]
        Xva, yva = X[idx[split:]], y[idx[split:]]
        model = TinyMLP(X.shape[1], rng)
        model.fit(Xtr, ytr, epochs=30, batch_size=128, rng=rng)
        acc = _accuracy(model.forward(Xva), yva)
        print(f'   samples={n}, val_acc={acc:.3f}, chance={chance:.3f}',
              '✅' if acc > chance else '❌')
    print()

    # ③ late fusion
    print('③ Late fusion (multimodal baseline)')
    X, y = prepare_fusion(all_bundles, rng)
    n = len(X)
    split = int(n * 0.8)
    idx = rng.permutation(n)
    Xtr, ytr = X[idx[:split]], y[idx[:split]]
    Xva, yva = X[idx[split:]], y[idx[split:]]
    model = TinyMLP(X.shape[1], rng)
    model.fit(Xtr, ytr, epochs=30, batch_size=128, rng=rng)
    acc = _accuracy(model.forward(Xva), yva)
    print(f'   samples={n}, val_acc={acc:.3f}, chance={chance:.3f}',
          '✅' if acc > chance else '❌')
    print()

    print('Phase 2 complete.')

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# FedAvg
# ---------------------------------------------------------------------------

def fedavg(models: list) -> TinyMLP:
    """Average weights across client models."""
    avg = TinyMLP.__new__(TinyMLP)
    avg.lr = models[0].lr
    avg.W1 = np.mean([m.W1 for m in models], axis=0)
    avg.b1 = np.mean([m.b1 for m in models], axis=0)
    avg.W2 = np.mean([m.W2 for m in models], axis=0)
    avg.b2 = np.mean([m.b2 for m in models], axis=0)
    return avg

# ---------------------------------------------------------------------------
# REPRODUCIBILITY NOTE
# ---------------------------------------------------------------------------
# FedAvg vs Local comparisons require sufficient data to overcome variance.
# The "Local > FedAvg" result under modality heterogeneity is structural,
# but becomes statistically reliable only with adequate sample sizes.
#
# Recommended minimum: n_steps >= 10000 (n_steps=20000 used in paper figures)
# With n_steps < 5000 the comparison may reverse for some seeds due to
# high variance in the small-data regime.
# ---------------------------------------------------------------------------

def run_fl_demo(
    data: dict,
    seed: int = 42,
    fl_rounds: int = 5,
    local_epochs: int = 10,
) -> None:
    """FedAvg demo: local vs federated on late fusion."""
    rng = np.random.default_rng(seed)
    client_ids = list(data.keys())

    print("HeteroSense-FL -- FL Demo (FedAvg)")
    print("=" * 55)
    print(f"Clients: {client_ids}")
    print(f"FL rounds: {fl_rounds}, local epochs/round: {local_epochs}")
    print()

    # Prepare per-client data
    client_data = {}
    for cid, bundles in data.items():
        X, y = prepare_fusion(bundles, rng)
        n = len(X)
        split = int(n * 0.8)
        idx = rng.permutation(n)
        client_data[cid] = {
            "Xtr": X[idx[:split]], "ytr": y[idx[:split]],
            "Xva": X[idx[split:]], "yva": y[idx[split:]],
        }

    input_dim = next(iter(client_data.values()))["Xtr"].shape[1]

    # Local baseline (no FL)
    print("Local training (no FL):")
    local_accs = {}
    for cid, d in client_data.items():
        model = TinyMLP(input_dim, rng)
        model.fit(d["Xtr"], d["ytr"],
                  epochs=fl_rounds * local_epochs,
                  batch_size=128, rng=rng)
        acc = _accuracy(model.forward(d["Xva"]), d["yva"])
        local_accs[cid] = acc
        print(f"  Client {cid}: val_acc={acc:.3f}")
    print(f"  Mean: {np.mean(list(local_accs.values())):.3f}")
    print()

    # FedAvg
    print(f"FedAvg ({fl_rounds} rounds):")
    global_model = TinyMLP(input_dim, rng)
    client_models = {}
    for cid in client_ids:
        m = TinyMLP(input_dim, rng)
        m.W1, m.b1, m.W2, m.b2 = (
            global_model.W1.copy(), global_model.b1.copy(),
            global_model.W2.copy(), global_model.b2.copy()
        )
        client_models[cid] = m

    for r in range(fl_rounds):
        for cid, m in client_models.items():
            d = client_data[cid]
            m.fit(d["Xtr"], d["ytr"],
                  epochs=local_epochs, batch_size=128, rng=rng)
        global_model = fedavg(list(client_models.values()))
        for m in client_models.values():
            m.W1, m.b1, m.W2, m.b2 = (
                global_model.W1.copy(), global_model.b1.copy(),
                global_model.W2.copy(), global_model.b2.copy()
            )

    fed_accs = {}
    for cid, d in client_data.items():
        acc = _accuracy(global_model.forward(d["Xva"]), d["yva"])
        fed_accs[cid] = acc
        delta = acc - local_accs[cid]
        sign = "▲" if delta > 0.005 else ("▼" if delta < -0.005 else "~")
        print(f"  Client {cid}: val_acc={acc:.3f}  "
              f"(local={local_accs[cid]:.3f}, delta={delta:+.3f} {sign})")
    print(f"  Mean: {np.mean(list(fed_accs.values())):.3f}")
    print()
    print("FL demo complete.")

if __name__ == '__main__':
    from heterosense._core._config_manager import ConfigurationManager
    from heterosense.dataset_builder import DatasetBuilder

    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 3000
    cfg.config['random_seed'] = 42
    sim_cfg = cfg.to_sim_config()

    data = DatasetBuilder(sim_cfg).build()
    run_baseline_demo(data, seed=42)
