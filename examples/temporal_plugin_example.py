"""
HeteroSense-FL: plug-in template for FL algorithms.

Three clearly marked replacement points:
  [A] extract_features()  — temporal feature extractor
  [B] train_local()       — local training loop
  [C] aggregate()         — FL aggregation algorithm

Run:
  python examples/temporal_plugin_example.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM
from heterosense import DatasetBuilder, TemporalWindowSampler

N_CLIENTS, N_STEPS, WINDOW, N_ROUNDS = 5, 2000, 3, 3
clients = ClientFactory.make(N_CLIENTS, strategy="round_robin", seed=42)
data    = DatasetBuilder(CM.from_clients(clients, n_steps=N_STEPS).to_sim_config()).build()
STATE   = {s: i for i, s in enumerate(
           ["ABSENT","STATIONARY","WALKING","TRANSITION","ABNORMAL"])}

# ─── [A] REPLACE: temporal feature extractor ─────────────────────────────────
def extract_features(window, sampler):
    """Return a 1-D numpy feature vector from a temporal window.
    Replace with LSTM, Transformer, CNN, etc."""
    z = TemporalWindowSampler.lidar_z_series(window)    # (window,)
    p = TemporalWindowSampler.pressure_series(window)   # (window,)
    return np.concatenate([z, p])                        # (2*window,)


def build_xy(bundles):
    s = TemporalWindowSampler(bundles, window=WINDOW)
    Xs, ys = [], []
    for w in s:
        Xs.append(extract_features(w, s))
        ys.append(STATE.get(TemporalWindowSampler.center_label(w, s.center_idx()), 0))
    if not Xs:
        return np.empty((0, 2*WINDOW)), np.empty(0, int)
    return np.array(Xs, np.float32), np.array(ys, int)


# ─── [B] REPLACE: local training loop ────────────────────────────────────────
def train_local(weights, X, y):
    """Train locally for one FL round. Replace with PyTorch, TF, sklearn, etc.
    Input:  weights — current global model params (list[np.ndarray])
    Output: updated local params (same structure)"""
    n_cls = 5; n_in = X.shape[1] if len(X) else 2*WINDOW
    W, b = weights if weights else [np.zeros((n_in, n_cls)), np.zeros(n_cls)]
    if not len(X): return [W, b]
    lg = X @ W + b; lg -= lg.max(1,keepdims=True)
    pr = np.exp(lg)/np.exp(lg).sum(1,keepdims=True)
    dL = pr.copy(); dL[np.arange(len(y)),y] -= 1; dL /= len(y)
    return [W - 0.01*(X.T@dL), b - 0.01*dL.sum(0)]


# ─── [C] REPLACE: FL aggregation algorithm ───────────────────────────────────
def aggregate(local_weights_list):
    """Aggregate local params into global model. Replace with FedProx etc."""
    return [np.mean([ws[i] for ws in local_weights_list], 0)
            for i in range(len(local_weights_list[0]))]


# ── FL loop ───────────────────────────────────────────────────────────────────
print(f"Dataset: {N_CLIENTS} clients × {N_STEPS} steps")
for cid, buns in data.items():
    print(f"  client {cid}: lidar={any(b.lidar is not None for b in buns)} "
          f"bed={any(b.pressure is not None for b in buns)}")

cdata = {cid: build_xy(buns) for cid, buns in data.items()}
gw = None
for rnd in range(1, N_ROUNDS+1):
    lws = [train_local(list(gw) if gw else [], X, y)
           for X, y in cdata.values() if len(X)]
    gw = aggregate(lws)
    X0, y0 = cdata["0"]
    if len(X0):
        acc = (np.argmax(X0 @ gw[0] + gw[1], 1) == y0).mean()
        print(f"  round {rnd}: client-0 acc = {acc:.3f}")

print("\nReplace [A], [B], [C] with your own components.")
