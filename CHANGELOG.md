# Changelog

All notable changes to HeteroSense-FL are documented here.

## [Unreleased] — v2 (branch `v2-dev`)

Backward-compatible extension adding three experiment knobs. All new config fields
default to v1 behavior; `tests/test_v1_compat.py` will enforce numeric-identical
output for v1 default configs. Design: see `EXTENSION_MAP.md`.

### Planned (pending PI approval — Phase A stop)
- Temporal-resolution knob: new PIR/motion modality (`refractory_s`, `report_period_s`,
  `bedroom_sensor_count`) — the physically meaningful home for these controls.
- Coverage knob: PIR FOV coverage / blind-spot rate; pressure coverage fraction.
- Geometry knob: multi-room topology + bed↔toilet (B2T) behavior, calibrated to the
  61-home empirical B2T distribution (anchor: single bedroom PIR misses sub-minute B2T).

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
