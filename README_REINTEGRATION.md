# Re-integration conditions — fall-motion feature → v2.x

This branch holds the "fall motion diversity" work (commit `0469ac3`, 大谷青羽). It was
**reverted from `main`** — not because the feature is unwanted, but because it changed the
**default (v1) simulator output** (golden digest `30c0e25…` → `603f00a…`), an accidental
backward-compatibility regression caused by consuming the **shared RNG stream** inside the
ABNORMAL branch of `BehaviorModel.generate()` / `ObservationModel`.

The fall-motion feature is **welcome in a future v2.x** once all three conditions hold:

1. **Dedicated child RNG.** The ABNORMAL/fall sampling must draw from an isolated child
   generator (e.g. `rng.spawn(1)[0]` or a separately-seeded `default_rng`) so it does **not**
   consume or shift the shared stream. Non-ABNORMAL output must be byte-identical to v1.
2. **`tests/test_v1_compat.py` passes** on the rebased branch (v1 default config → the pinned
   v1.0.0 golden digest `30c0e2537a8c539edf7e218efa5a2489b93b1173a3042eb0bbf619fe94c567b8`).
3. **Scope check.** Fall motion belongs to the **immediate-detection channel (LiDAR-based)**.
   Confirm it does not affect the **absence-inference layer** evaluation (the PIR/B2T anchor
   `tests/test_anchor.py` and the flagship study in `experiments/fl_observation_layer/`).

Meet these three → open a PR against `main` for a v2.x minor release. The two lines are
scope-separate (immediate-detection vs. absence-inference) and should not otherwise co-mingle.
