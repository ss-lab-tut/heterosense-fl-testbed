# Changelog

All notable changes to HeteroSense-FL are documented here.

## [2.0.0] — 2026-07-04

Backward-compatible extension. Scope (Alt-C / split release): **v2.0 = temporal-resolution
knob only**; coverage + geometry deferred to v2.1 (insertion points reserved — see
EXTENSION_MAP.md §7 extensibility contract). Design: see `EXTENSION_MAP.md`.

**Backward-compatibility guarantee**: any v1 config produces v1-identical output
(`tests/test_v1_compat.py` pins the v1.0.0 golden digest). To reproduce v1 exactly:
`git checkout v1.0.0`. All 43 tests pass (37 v1 + 4 anchor + 2 compat).

**Known limitations**: in-bed flicker / exit rate / morning cap are UNcalibrated behavioral
assumptions (only the B2T duration distribution is real-calibrated); the flagship study's
observation model is a simple windowed classifier; downstream injection uses a per-event
approximation (REPORT.md §7).

### Added (v2.0, Phase B — implemented, 43 tests passing)
- Temporal-resolution knob: PIR/motion modality with `refractory_s`, `report_period_s`,
  `bedroom_sensor_count` on `ClientConfig` (all default 0 → PIR disabled → v1 behavior).
  New modules `heterosense/_core/_pir_model.py`, `_b2t.py`, `_extractor.py`.
- Frozen, real-calibrated B2T snapshot `heterosense/_data/b2t_snapshot.json` (61-home CASAS,
  n=3996, median 1.93 min, 27% sub-minute) + regenerator `tools/make_b2t_snapshot.py`.
  Provenance embedded (longlie_study commit 0baaf8c, source DOI 10.5281/zenodo.15708568,
  CC-BY-4.0). **Snapshot updates are an explicit, CHANGELOG-documented operation only.**
- `LatentState.room_id` (default 0) — reserved seam for the v2.1 geometry knob.
- Anchor verification `tests/test_anchor.py`: single bedroom PIR extractor "anchor recall"
  within the observed 61-home range 0.03–0.55; recovers with a 2nd absence-confirming sensor.
  Snapshot data-hash pinned. Mechanism-driven, not tuned (base_gap = longlie_study G = 5 min).
  NOTE: adding the in-bed flicker consumed the shared RNG stream, shifting the extractor anchor
  recall from 0.138 to 0.179 (both inside the 61-home range). The anchor test is now two-layer:
  an exact drift guard (0.179 ± 0.005) plus the separate 61-home-range consistency check.
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
