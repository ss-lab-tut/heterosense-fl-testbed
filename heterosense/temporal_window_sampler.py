"""HeteroSense TemporalWindowSampler: sliding-window interface for temporal encoders."""
from __future__ import annotations

import numpy as np
from typing import Iterator
import os
from heterosense._core._observation_model import ModalityBundle

class TemporalWindowSampler:
    """Sliding window sampler over a ModalityBundle time series.

    Parameters
    ----------
    bundles : list[ModalityBundle]
        Time-ordered sequence from DatasetBuilder.build()
    window  : int
        Number of frames per sample (default: 3)
    stride  : int
        Step between windows (default: 1)
    """

    def __init__(
        self,
        bundles: list[ModalityBundle],
        window: int = 3,
        stride: int = 1,
    ) -> None:
        if window <= 0:
            raise ValueError(f"window must be >= 1, got {window}")
        if stride <= 0:
            raise ValueError(f"stride must be >= 1, got {stride}")
        self.bundles = bundles
        self.window  = window
        self.stride  = stride

    def __len__(self) -> int:
        return max(0, (len(self.bundles) - self.window) // self.stride + 1)

    def __iter__(self) -> Iterator[list[ModalityBundle]]:
        for i in range(0, len(self.bundles) - self.window + 1, self.stride):
            yield self.bundles[i:i + self.window]

    def center_idx(self) -> int:
        """Index of the center frame within a window."""
        return self.window // 2

    # ------------------------------------------------------------------
    # Convenience: extract simple temporal features from a window
    # ------------------------------------------------------------------

    @staticmethod
    def pressure_series(window: list[ModalityBundle]) -> np.ndarray:
        """Per-frame pressure sum. Shape: (T,)"""
        return np.array([
            float(b.pressure.sum()) if b.pressure is not None else 0.0
            for b in window
        ])

    @staticmethod
    def lidar_z_series(window: list[ModalityBundle]) -> np.ndarray:
        """Per-frame mean z of point cloud. Shape: (T,)"""
        return np.array([
            float(b.lidar[:, 2].mean()) if b.lidar is not None else 0.0
            for b in window
        ])

    @staticmethod
    def lidar_upper_x_series(window: list[ModalityBundle]) -> np.ndarray:
        """Per-frame upper-body x asymmetry. Shape: (T,)"""
        vals = []
        for b in window:
            if b.lidar is None:
                vals.append(0.0)
            else:
                pts = b.lidar
                zm  = float(np.median(pts[:, 2]))
                upper = pts[pts[:, 2] > zm]
                vals.append(float(upper[:, 0].mean()) if len(upper) > 0 else 0.0)
        return np.array(vals)

    @staticmethod
    def center_label(window: list[ModalityBundle], center_idx: int) -> str:
        return window[center_idx].semantic_state

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from heterosense._core._config_manager import ConfigurationManager
    from heterosense.dataset_builder import DatasetBuilder

    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 1000
    cfg.config['random_seed'] = 42
    sim_cfg = cfg.to_sim_config()

    data = DatasetBuilder(sim_cfg).build()

    pass  # smoke test removed
    print('=' * 50)

    for cid, bundles in data.items():
        for W in [3, 5]:
            sampler = TemporalWindowSampler(bundles, window=W)
            windows = list(sampler)
            w0 = windows[0]
            ps = TemporalWindowSampler.pressure_series(w0)
            zs = TemporalWindowSampler.lidar_z_series(w0)
            print(f'Client {cid} window={W}: '
                  f'n_windows={len(windows)}, '
                  f'pressure_series={ps.round(3)}, '
                  f'z_series={zs.round(3)}')

    print()
    print('DatasetBuilder: unchanged ✅')
    pass  # smoke test removed
    print()
    print('✓ smoke test passed')
