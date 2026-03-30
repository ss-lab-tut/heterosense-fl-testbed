# Changelog

All notable changes to HeteroSense-FL are documented here.

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
