"""HeteroSense ObservationModelV3: renders LatentStateV3 to ModalityBundle."""
from __future__ import annotations

import numpy as np
import os

from heterosense._core._observation_model import ObservationModel, ModalityBundle
from heterosense._core._behavior_model_ext import LatentStateV3, SupportState

_PRESSURE_MAP_SIZE = 16

class ObservationModelV3(ObservationModel):
    """Extended observation model with support_state rendering."""

    def observe_v3(self, ls: LatentStateV3, timestamp: float) -> ModalityBundle:
        """Generate ModalityBundle from v3 latent state."""
        # Generate base observation using structured observation model
        bundle = self.observe(ls.base, timestamp)

        # Apply support_state modulation
        if ls.support_state == SupportState.PARTIAL:
            bundle = self._apply_partial_support(bundle, ls)

        return bundle

    def _apply_partial_support(
        self,
        bundle: ModalityBundle,
        ls: LatentStateV3,
    ) -> ModalityBundle:
        """Modulate pressure and point cloud for PARTIAL_SUPPORT.

        差は意図的に弱く設定する：
        - pressure：中央から端への軽い重心シフト
        - point cloud：体幹の軽い傾き
        """
        rng = self._rng
        new_pressure = bundle.pressure
        new_lidar    = bundle.lidar

        # ── pressure: shift distribution toward edge ───────────────────────
        if bundle.pressure is not None:
            M = _PRESSURE_MAP_SIZE
            p = bundle.pressure.copy()

            # Shift weight toward bed edge direction
            bx, by = self._bed_pos
            dx = ls.x - bx
            dy = ls.y - by
            dist = float(np.sqrt(dx**2 + dy**2)) + 1e-6

            # edge direction in grid coordinates
            edge_gx = (dx / dist) * M * 0.15   # 15% shift (weak)
            edge_gy = (dy / dist) * M * 0.15

            # Apply shift via roll + blend（弱い）
            shifted = np.roll(p, int(round(edge_gx)), axis=0)
            shifted = np.roll(shifted, int(round(edge_gy)), axis=1)
            alpha = 0.35   # blend factor: 35% shifted, 65% original
            new_pressure = (1 - alpha) * p + alpha * shifted
            new_pressure = np.clip(new_pressure, 0.0, 1.0)

        # ── point cloud: slight trunk lean toward edge ─────────────────────
        if bundle.lidar is not None:
            pts = bundle.lidar.copy()

            bx, by = self._bed_pos
            dx = ls.x - bx
            dy = ls.y - by
            dist = float(np.sqrt(dx**2 + dy**2)) + 1e-6

            # Upper body only, weak lean
            lean_dx = (dx / dist) * 0.08   # 8cm lean (weak)
            lean_dy = (dy / dist) * 0.08

            z_median = float(np.median(pts[:, 2]))
            upper_mask = pts[:, 2] > z_median
            pts[upper_mask, 0] += lean_dx + rng.normal(0, 0.02, upper_mask.sum())
            pts[upper_mask, 1] += lean_dy + rng.normal(0, 0.02, upper_mask.sum())
            pts[:, 2] = np.clip(pts[:, 2], 0.0, 2.5)
            new_lidar = pts

        return ModalityBundle(
            client_id      = bundle.client_id,
            timestamp      = bundle.timestamp,
            lidar          = new_lidar,
            pressure       = new_pressure,
            semantic_state = bundle.semantic_state,
            posture_state  = bundle.posture_state,
            bed_zone       = bundle.bed_zone,
            abnormal_phase = bundle.abnormal_phase,
        )

# ---------------------------------------------------------------------------
# DatasetBuilder v3
# ---------------------------------------------------------------------------

class DatasetBuilderV3:
    """Generates per-client ModalityBundle sequences with support_state."""

    def __init__(self, sim_config) -> None:
        self._cfg = sim_config

    def build(self) -> dict[str, list]:
        import hashlib
        from heterosense._core._behavior_model_ext import _BehaviorModelExt

        result = {}
        for client_cfg in self._cfg.clients:
            h = hashlib.sha256(client_cfg.client_id.encode()).digest()
            seed_offset = int.from_bytes(h[:4], 'big')
            rng = np.random.default_rng(self._cfg.random_seed + seed_offset)

            bm  = _BehaviorModelExt(config=client_cfg, rng=rng)
            obs = ObservationModelV3(config=client_cfg, rng=rng)

            states = bm.generate(self._cfg.n_steps)
            bundles = []
            for ls in states:
                t = round(ls.t * self._cfg.delta_t, 6)
                bundles.append(obs.observe_v3(ls, t))
            result[client_cfg.client_id] = bundles
        return result

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from heterosense._core._config_manager import ConfigurationManager
    from heterosense._core._behavior_model_ext import SupportState
    from heterosense._core._behavior_model import Posture, BedZone

    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 20000
    cfg.config['random_seed'] = 42

    builder = DatasetBuilderV3(cfg.to_sim_config())
    data = builder.build()
    all_bundles = [b for bundles in data.values() for b in bundles]

    pass  # smoke test removed
    print('=' * 50)
    print(f'Total bundles: {len(all_bundles)}')
    print()

    # task candidates
    # Note: support_state is embedded in generation, not in ModalityBundle directly
    # We re-generate with explicit tracking for diagnosis
    from heterosense._core._config_manager import ConfigurationManager
    from heterosense._core._behavior_model_ext import _BehaviorModelExt
    import hashlib

    cfg2 = ConfigurationManager(None)
    cfg2.config['n_steps'] = 20000
    cfg2.config['random_seed'] = 42
    sim_cfg = cfg2.to_sim_config()

    class_a_bundles, class_b_bundles = [], []
    for client_cfg in sim_cfg.clients:
        h = hashlib.sha256(client_cfg.client_id.encode()).digest()
        seed_offset = int.from_bytes(h[:4], 'big')
        rng_bm  = np.random.default_rng(42 + seed_offset)
        rng_obs = np.random.default_rng(42 + seed_offset)
        bm  = _BehaviorModelExt(client_cfg, rng_bm)
        obs = ObservationModelV3(client_cfg, rng_obs)
        states = bm.generate(20000)
        for ls in states:
            if ls.posture == Posture.SEATED and ls.bed_zone == BedZone.BED_EDGE:
                b = obs.observe_v3(ls, ls.t)
                if ls.support_state == SupportState.PARTIAL:
                    class_a_bundles.append(b)
                else:
                    class_b_bundles.append(b)

    print(f'Task: SEATED+BED_EDGE+PARTIAL vs SEATED+BED_EDGE+FULL')
    print(f'  Class A (PARTIAL): {len(class_a_bundles)}')
    print(f'  Class B (FULL):    {len(class_b_bundles)}')

    # Quick pressure check
    def mean_edge_ratio(bundles):
        ratios = []
        for b in bundles:
            if b.pressure is None: continue
            p = b.pressure
            edge = (p[:4,:].sum() + p[12:,:].sum() + p[:,0:4].sum() + p[:,12:].sum())
            total = p.sum()
            if total > 0: ratios.append(edge/total)
        return np.mean(ratios) if ratios else 0

    ra = mean_edge_ratio(class_a_bundles)
    rb = mean_edge_ratio(class_b_bundles)
    print(f'\n  Pressure edge ratio:')
    print(f'    PARTIAL: {ra:.3f}')
    print(f'    FULL:    {rb:.3f}')
    print(f'    diff:    {ra-rb:+.3f}', '✅' if abs(ra-rb) > 0.01 else '⚠️ small')

    # Quick point cloud check
    def mean_x_spread(bundles):
        spreads = []
        for b in bundles:
            if b.lidar is None: continue
            spreads.append(float(np.std(b.lidar[:,0])))
        return np.mean(spreads) if spreads else 0

    sa = mean_x_spread(class_a_bundles)
    sb = mean_x_spread(class_b_bundles)
    print(f'\n  Point cloud x-spread (std):')
    print(f'    PARTIAL: {sa:.3f}')
    print(f'    FULL:    {sb:.3f}')
    print(f'    diff:    {sa-sb:+.3f}', '✅' if abs(sa-sb) > 0.005 else '⚠️ small')
    print()
    pass  # removed
