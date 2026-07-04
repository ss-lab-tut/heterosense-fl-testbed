# Changelog

All notable changes to HeteroSense-FL are documented here.

## [Unreleased] — v2 (branch `v2-dev`)

Backward-compatible extension adding three experiment knobs. All new config fields
default to v1 behavior; `tests/test_v1_compat.py` will enforce numeric-identical
output for v1 default configs. Design: see `EXTENSION_MAP.md`.

Scope decided (2026-07-04, Alt-C / split release): **v2.0 = temporal-resolution knob
only**; coverage + geometry deferred to v2.1 (insertion points reserved — see
EXTENSION_MAP.md §7 extensibility contract).

### Added (v2.0, Phase B — implemented, 43 tests passing)
- Temporal-resolution knob: PIR/motion modality with `refractory_s`, `report_period_s`,
  `bedroom_sensor_count` on `ClientConfig` (all default 0 → PIR disabled → v1 behavior).
  New modules `heterosense/_core/_pir_model.py`, `_b2t.py`, `_extractor.py`.
- Frozen, real-calibrated B2T snapshot `heterosense/_data/b2t_snapshot.json` (61-home CASAS,
  n=3996, median 1.93 min, 27% sub-minute) + regenerator `tools/make_b2t_snapshot.py`.
  Provenance embedded (longlie_study commit 0baaf8c, source DOI 10.5281/zenodo.15708568,
  CC-BY-4.0). **Snapshot updates are an explicit, CHANGELOG-documented operation only.**
- `LatentState.room_id` (default 0) — reserved seam for the v2.1 geometry knob.
- Anchor verification `tests/test_anchor.py`: single bedroom PIR recall 0.10–0.14 (within the
  observed 61-home range 0.03–0.55); recovers with a 2nd absence-confirming sensor. Snapshot
  data-hash pinned. Mechanism-driven, not tuned (base_gap = longlie_study G = 5 min).
- Backward-compat regression `tests/test_v1_compat.py`: v1 default config → byte-identical to
  v1.0.0 (golden digest 30c0e25…).

### Deferred to v2.1
- Coverage knob (PIR FOV / blind-spot; pressure coverage fraction).
- Geometry knob (multi-room topology; B2T durations derived from room distance).

## [1.0.0] — 2026-01-01

### Added
- `DatasetBuilder`: generates `{client_id: [ModalityBundle]}` time series for N clients
- `ClientFactory`: configures N-client modality availability via `round_robin`, `uniform`, `explicit`, `random` strategies
- `TemporalWindowSampler`: sliding-window iterator with plug-in encoder interface
- `run_validation`: automated observation integrity checks V1–V4
- Reference benchmarks reproducible via `heterosense-benchmark`
- GitHub Actions CI (Ubuntu / macOS / Windows × Python 3.9–3.12)
- ReadTheDocs documentation
- Jupyter notebook quickstart (`examples/quickstart.ipynb`)
