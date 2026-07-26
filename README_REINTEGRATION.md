# Re-integration conditions — fall-motion feature → `main`

This branch holds the "fall motion diversity" work (commit `0469ac3`, 大谷青羽). It was
**reverted from `main`** — not because the feature is unwanted, but because it changed the
**default (v1) simulator output** (golden digest `30c0e25…` → `603f00a…`), an accidental
backward-compatibility regression caused by consuming the **shared RNG stream** inside the
ABNORMAL branch of `BehaviorModel.generate()` / `ObservationModel`.

The fall-motion feature is **welcome on `main`** once both conditions hold:

1. **Dedicated child RNG.** The ABNORMAL/fall sampling must draw from an isolated child
   generator (e.g. `rng.spawn(1)[0]` or a separately-seeded `default_rng`) so it does **not**
   consume or shift the shared stream. Non-ABNORMAL output must be byte-identical to v1.
2. **v1 compatibility verified.** Running the v1 default config must reproduce the pinned
   v1.0.0 golden digest
   `30c0e2537a8c539edf7e218efa5a2489b93b1173a3042eb0bbf619fe94c567b8`.
   Note: the former `tests/test_v1_compat.py` is no longer in the repository — the check has
   to be re-created on this branch, comparing against tag `v1.0.0` (`f74a061`), which is the
   frozen SoftwareX release and must not be modified.

Meet both → open a PR against `main`. Fall motion belongs to the **immediate-detection
channel (LiDAR-based)**; keep the change confined to that channel.
