"""
Fig. 3 — HeteroSense-FL reference benchmark.

Reproduces the benchmark chart for Table 3 in the paper.
All numbers are computed from heterosense-benchmark (seeds {42,123,7};
n_steps=3000; 3 FL rounds; TinyMLP encoder).

Run:
    python scripts/figures/generate_fig3_benchmark.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder
from heterosense import TemporalWindowSampler

# ── Benchmark configuration (must match heterosense-benchmark exactly) ──────
SEEDS    = [42, 123, 7]
N_STEPS  = 3000
N_ROUNDS = 3
WINDOW   = 3
N_FEAT   = 2 * WINDOW
N_CLS    = 5
STATE_MAP = {s: i for i, s in enumerate(
    ['ABSENT', 'STATIONARY', 'WALKING', 'TRANSITION', 'ABNORMAL'])}

CONDITIONS = [
    (3,  'N=3\nhomogeneous',  'round_robin', ['both']),
    (3,  'N=3\nhetero.',      'round_robin',  None),
    (10, 'N=10\nhetero.',     'round_robin',  None),
    (20, 'N=20\nhetero.',     'round_robin',  None),
    (50, 'N=50\nhetero.',     'round_robin',  None),
]

class TinyMLP:
    def __init__(self, rng):
        self.W1 = rng.normal(0, .1, (N_FEAT, 16))
        self.b1 = np.zeros(16)
        self.W2 = rng.normal(0, .1, (16, N_CLS))
        self.b2 = np.zeros(N_CLS)

    def predict(self, X):
        return np.argmax(
            np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2, 1)

    def weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_weights(self, ws):
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in ws]


def sgd(m, X, y, lr=0.01, epochs=5, batch=64, rng=None):
    rng = rng or np.random.default_rng()
    n = len(X)
    if n == 0: return
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, batch):
            xb, yb = X[idx[s:s+batch]], y[idx[s:s+batch]]
            h  = np.maximum(0, xb @ m.W1 + m.b1)
            lg = h @ m.W2 + m.b2
            lg -= lg.max(1, keepdims=True)
            pr  = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
            dL  = pr.copy()
            dL[np.arange(len(yb)), yb] -= 1
            dL /= len(yb)
            dW2 = h.T @ dL;       db2 = dL.sum(0)
            dh  = dL @ m.W2.T;    dh[h <= 0] = 0
            dW1 = xb.T @ dh;      db1 = dh.sum(0)
            m.W1 -= lr * dW1;  m.b1 -= lr * db1
            m.W2 -= lr * dW2;  m.b2 -= lr * db2


def build_xy(bundles):
    s = TemporalWindowSampler(bundles, WINDOW)
    Xs, ys = [], []
    for w in s:
        z = TemporalWindowSampler.lidar_z_series(w)
        p = TemporalWindowSampler.pressure_series(w)
        Xs.append(np.concatenate([z, p]))
        ys.append(STATE_MAP.get(
            TemporalWindowSampler.center_label(w, s.center_idx()), 0))
    if not Xs:
        return np.empty((0, N_FEAT)), np.empty(0, int)
    return np.array(Xs, np.float32), np.array(ys, int)


def run_one(N, strategy, seed, patterns):
    rm  = np.random.default_rng(seed + 1000)
    sc  = CM.from_clients(
        ClientFactory.make(N, strategy=strategy, patterns=patterns, seed=seed),
        n_steps=N_STEPS).to_sim_config()
    data = DatasetBuilder(sc).build()
    cdata = {cid: build_xy(buns) for cid, buns in data.items()}

    # Local baseline
    loc = []
    for X, y in cdata.values():
        if len(X) < 10: continue
        sp = int(.8 * len(X))
        m  = TinyMLP(np.random.default_rng(rm.integers(1 << 30)))
        sgd(m, X[:sp], y[:sp], rng=np.random.default_rng(rm.integers(1 << 30)))
        loc.append((m.predict(X[sp:]) == y[sp:]).mean())

    # FedAvg
    gm = TinyMLP(np.random.default_rng(rm.integers(1 << 30)))
    for _ in range(N_ROUNDS):
        lws = []
        for X, y in cdata.values():
            if len(X) < 10: continue
            m = TinyMLP(np.random.default_rng(rm.integers(1 << 30)))
            m.set_weights(gm.weights())
            sp = int(.8 * len(X))
            sgd(m, X[:sp], y[:sp], rng=np.random.default_rng(rm.integers(1 << 30)))
            lws.append(m.weights())
        if lws:
            gm.set_weights(
                [np.mean([w[i] for w in lws], 0) for i in range(4)])
    fed = []
    for X, y in cdata.values():
        if len(X) < 10: continue
        sp = int(.8 * len(X))
        fed.append((gm.predict(X[sp:]) == y[sp:]).mean())

    return np.nanmean(loc), np.nanmean(fed)


# ── Run benchmark ─────────────────────────────────────────────────────────────
print('Computing benchmark (this takes ~3 min)...')
local_means, local_stds, fedavg_means, fedavg_stds = [], [], [], []

for N, label, strategy, patterns in CONDITIONS:
    ls, fs = [], []
    for seed in SEEDS:
        l, f = run_one(N, strategy, seed, patterns)
        ls.append(l); fs.append(f)
    lm, ls_ = np.mean(ls), np.std(ls)
    fm, fs_ = np.mean(fs), np.std(fs)
    local_means.append(lm);  local_stds.append(ls_)
    fedavg_means.append(fm); fedavg_stds.append(fs_)
    print(f'  {label.replace(chr(10)," ")}: Local={lm:.3f}+/-{ls_:.3f}  '
          f'FedAvg={fm:.3f}+/-{fs_:.3f}  Δ={lm-fm:+.3f}')

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
conditions = [c[1] for c in CONDITIONS]
x = np.arange(len(conditions))
w = 0.32

ax.bar(x - w/2, local_means,  w, yerr=local_stds,
       color='#1F4E79', alpha=0.85, capsize=5,
       label='Local training', error_kw={'linewidth': 1.5})
ax.bar(x + w/2, fedavg_means, w, yerr=fedavg_stds,
       color='#E65100', alpha=0.85, capsize=5,
       label='FedAvg', error_kw={'linewidth': 1.5})

for i, (lm, fm) in enumerate(zip(local_means, fedavg_means)):
    delta = lm - fm
    ax.annotate(f'Δ={delta:+.3f}',
                xy=(x[i], max(lm, fm) + 0.025),
                ha='center', va='bottom', fontsize=8.5,
                color='#333333', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(conditions, fontsize=10)
ax.set_ylabel('Standard accuracy (mean ± std)', fontsize=11)
ax.set_ylim(0.28, 0.52)
ax.set_title(
    'Fig. 3.  Local vs FedAvg accuracy under N-client modality heterogeneity\n'
    '(seeds: {42, 123, 7};  n_steps = 3000;  3 FL rounds;  TinyMLP)',
    fontsize=10, fontweight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/tmp/fig3_benchmark.png', dpi=150,
            bbox_inches='tight', facecolor='white')
plt.close()
print('✓ Fig 3 saved  →  /tmp/fig3_benchmark.png')
