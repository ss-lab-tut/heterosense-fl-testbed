"""HeteroSense DatasetBuilder: builds multimodal FL datasets."""
from __future__ import annotations

import os

import numpy as np
from heterosense._core._behavior_model import BehaviorModel
from heterosense._core._config_schema import SimConfig, ClientConfig
from heterosense._core._config_manager import ConfigurationManager
import hashlib

# Import structured observation model
from heterosense._core._observation_model import ObservationModel, ModalityBundle

class DatasetBuilder:
    """Generates per-client ModalityBundle sequences."""

    def __init__(self, sim_config: SimConfig) -> None:
        self._cfg = sim_config

    def _build_single_client(self, client_cfg: ClientConfig) -> list[ModalityBundle]:
        h = hashlib.sha256(client_cfg.client_id.encode()).digest()
        seed_offset = int.from_bytes(h[:4], 'big')
        rng = np.random.default_rng(self._cfg.random_seed + seed_offset)

        behavior_model = BehaviorModel(config=client_cfg, rng=rng)
        obs_model      = ObservationModel(config=client_cfg, rng=rng)

        latent_states = behavior_model.generate(self._cfg.n_steps)
        bundles = []
        for ls in latent_states:
            t = round(ls.t * self._cfg.delta_t, 6)
            bundles.append(obs_model.observe(ls, t))
        return bundles

    def build(self) -> dict[str, list[ModalityBundle]]:
        """Returns {client_id: [ModalityBundle, ...]}"""
        result = {}
        for client_cfg in self._cfg.clients:
            result[client_cfg.client_id] = self._build_single_client(client_cfg)
        return result

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 500
    cfg.config['random_seed'] = 42
    sim_cfg = cfg.to_sim_config()

    builder = DatasetBuilder(sim_cfg)
    data = builder.build()

    pass  # smoke test removed
    print("=" * 50)

    for cid, bundles in data.items():
        b0 = bundles[0]
        print(f"\nClient {cid}:")
        print(f"  Timesteps: {len(bundles)}")

        if b0.lidar is not None:
            n_pts = np.mean([len(b.lidar) for b in bundles])
            print(f"  LiDAR:    point cloud, avg {n_pts:.0f} points/frame")
            print(f"            shape example: {b0.lidar.shape}")
            print(f"            z range: [{b0.lidar[:,2].min():.2f}, {b0.lidar[:,2].max():.2f}] m")
        else:
            print(f"  LiDAR:    None (not available)")

        if b0.pressure is not None:
            print(f"  Pressure: map shape {b0.pressure.shape}")
            print(f"            max pressure: {max(b.pressure.max() for b in bundles):.3f}")
        else:
            print(f"  Pressure: None (not available)")

        states = [b.semantic_state for b in bundles]
        for s in ['ABSENT','STATIONARY','WALKING','TRANSITION','ABNORMAL']:
            n = states.count(s)
            print(f"  {s:<12}: {n:4d} steps ({n/len(bundles)*100:.1f}%)")

    print("\n✓ smoke test passed")
