"""
HeteroSense-FL reference benchmark — entry point: heterosense-benchmark

Reproduces Table 3 from the SoftwareX paper.
Seeds {42, 123, 7}; n_steps=3000; 3 FL rounds; TinyMLP encoder.
"""
import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder
from heterosense import TemporalWindowSampler

SEEDS    = [42, 123, 7]
N_STEPS  = 3000
N_ROUNDS = 3
WINDOW   = 3
N_FEAT   = 2 * WINDOW
N_CLS    = 5

CONDITIONS = [
    (3,  "homogeneous",  "round_robin", ["both"]),
    (3,  "round-robin",  "round_robin", None),
    (10, "round-robin",  "round_robin", None),
    (20, "round-robin",  "round_robin", None),
    (50, "round-robin",  "round_robin", None),
]

STATE_MAP = {s: i for i, s in enumerate(
    ["ABSENT", "STATIONARY", "WALKING", "TRANSITION", "ABNORMAL"])}


class _TinyMLP:
    def __init__(self, rng):
        self.W1 = rng.normal(0, .1, (N_FEAT, 16))
        self.b1 = np.zeros(16)
        self.W2 = rng.normal(0, .1, (16, N_CLS))
        self.b2 = np.zeros(N_CLS)

    def predict(self, X):
        return np.argmax(np.maximum(0, X @ self.W1 + self.b1) @ self.W2 + self.b2, 1)

    def weights(self):
        return [self.W1.copy(), self.b1.copy(), self.W2.copy(), self.b2.copy()]

    def set_weights(self, ws):
        self.W1, self.b1, self.W2, self.b2 = [w.copy() for w in ws]


def _build_xy(bundles):
    sampler = TemporalWindowSampler(bundles, window=WINDOW)
    Xs, ys = [], []
    for w in sampler:
        z = TemporalWindowSampler.lidar_z_series(w)
        p = TemporalWindowSampler.pressure_series(w)
        lbl = TemporalWindowSampler.center_label(w, sampler.center_idx())
        Xs.append(np.concatenate([z, p]))
        ys.append(STATE_MAP.get(lbl, 0))
    if not Xs:
        return np.empty((0, N_FEAT)), np.empty(0, int)
    return np.array(Xs, np.float32), np.array(ys, int)


def _sgd(model, X, y, lr=0.01, epochs=5, batch=64, rng=None):
    rng = rng or np.random.default_rng()
    n = len(X)
    if n == 0:
        return
    for _ in range(epochs):
        for idx in [rng.permutation(n)[s:s+batch] for s in range(0, n, batch)]:
            xb, yb = X[idx], y[idx]
            h = np.maximum(0, xb @ model.W1 + model.b1)
            lg = h @ model.W2 + model.b2
            lg -= lg.max(1, keepdims=True)
            pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
            dL = pr.copy(); dL[np.arange(len(yb)), yb] -= 1; dL /= len(yb)
            dW2 = h.T @ dL; db2 = dL.sum(0)
            dh = dL @ model.W2.T; dh[h <= 0] = 0
            dW1 = xb.T @ dh; db1 = dh.sum(0)
            model.W1 -= lr * dW1; model.b1 -= lr * db1
            model.W2 -= lr * dW2; model.b2 -= lr * db2


def _run(N, strategy, seed, patterns):
    rm = np.random.default_rng(seed + 1000)
    sc = CM.from_clients(
        ClientFactory.make(N, strategy=strategy, patterns=patterns, seed=seed),
        n_steps=N_STEPS,
    ).to_sim_config()
    data = DatasetBuilder(sc).build()
    cdata = {cid: _build_xy(buns) for cid, buns in data.items()}

    # Local baseline
    loc = []
    for X, y in cdata.values():
        if len(X) < 10: continue
        sp = int(.8 * len(X))
        m = _TinyMLP(np.random.default_rng(rm.integers(1e9)))
        _sgd(m, X[:sp], y[:sp], rng=np.random.default_rng(rm.integers(1e9)))
        loc.append((m.predict(X[sp:]) == y[sp:]).mean())

    # FedAvg
    gm = _TinyMLP(np.random.default_rng(rm.integers(1e9)))
    for _ in range(N_ROUNDS):
        lws = []
        for X, y in cdata.values():
            if len(X) < 10: continue
            m = _TinyMLP(np.random.default_rng(rm.integers(1e9)))
            m.set_weights(gm.weights())
            sp = int(.8 * len(X))
            _sgd(m, X[:sp], y[:sp], rng=np.random.default_rng(rm.integers(1e9)))
            lws.append(m.weights())
        if lws:
            gm.set_weights([np.mean([w[i] for w in lws], 0) for i in range(4)])
    fed = []
    for X, y in cdata.values():
        if len(X) < 10: continue
        sp = int(.8 * len(X))
        fed.append((gm.predict(X[sp:]) == y[sp:]).mean())

    return np.nanmean(loc), np.nanmean(fed)


def main():
    print("HeteroSense-FL Reference Benchmark")
    print("=" * 60)
    print(f"{'N':>3}  {'Pattern':<22}  {'Local':>12}  {'FedAvg':>12}  {'Delta':>7}")
    print("-" * 60)
    for N, label, strategy, patterns in CONDITIONS:
        ls, fs = zip(*[_run(N, strategy, s, patterns) for s in SEEDS])
        lm, ls_ = np.mean(ls), np.std(ls)
        fm, fs_ = np.mean(fs), np.std(fs)
        print(f"{N:>3}  {label:<22}  {lm:.3f}+/-{ls_:.3f}  {fm:.3f}+/-{fs_:.3f}  {lm-fm:+.3f}")
    print("=" * 60)
    print(f"Reproduce: heterosense-benchmark")
    print(f"Seeds: {SEEDS}  n_steps: {N_STEPS}  rounds: {N_ROUNDS}")


if __name__ == "__main__":
    main()
