"""HeteroSense observation model: structured multimodal sensor data generation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from heterosense._core._behavior_model import (
    LatentState, SemanticState, Posture, BedZone,
)
from heterosense._core._config_schema import ClientConfig

# ---------------------------------------------------------------------------
# Constants: point cloud
# ---------------------------------------------------------------------------

# Number of points per body part per state (approximate)
_N_TORSO: dict[Posture, int] = {
    Posture.UPRIGHT: 80,
    Posture.SEATED:  70,
    Posture.LYING:   90,
}
_N_HEAD: int  = 30
_N_LEGS: int  = 40   # total for both legs

# Body-part ellipsoid half-axes (metres): (x_radius, y_radius, z_radius)
# z = vertical axis
_TORSO_AXES: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.20, 0.15, 0.35),   # tall, narrow
    Posture.SEATED:  (0.22, 0.18, 0.25),   # shorter
    Posture.LYING:   (0.50, 0.18, 0.12),   # long, flat
}
_HEAD_AXES: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.10, 0.10, 0.12),
    Posture.SEATED:  (0.10, 0.10, 0.12),
    Posture.LYING:   (0.10, 0.10, 0.10),
}
# Centre offsets from body origin (x_offset, y_offset, z_offset)
_TORSO_CENTRE: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.0, 0.0, 0.90),
    Posture.SEATED:  (0.0, 0.0, 0.55),
    Posture.LYING:   (0.0, 0.0, 0.12),
}
_HEAD_CENTRE: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.0, 0.0, 1.65),
    Posture.SEATED:  (0.0, 0.0, 1.20),
    Posture.LYING:   (0.30, 0.0, 0.20),
}
_LEGS_CENTRE: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.0, 0.0, 0.45),
    Posture.SEATED:  (0.35, 0.0, 0.25),
    Posture.LYING:   (-0.55, 0.0, 0.10),
}
_LEGS_AXES: dict[Posture, tuple] = {
    Posture.UPRIGHT: (0.12, 0.20, 0.45),
    Posture.SEATED:  (0.35, 0.20, 0.15),
    Posture.LYING:   (0.55, 0.20, 0.10),
}

# ABNORMAL phase 1: splatter pattern (impact)
_N_SPLATTER: int = 120
_SPLATTER_AXES: tuple = (0.60, 0.50, 0.20)

# Background noise points
_N_BACKGROUND: int = 10

# Pressure map
_PRESSURE_MAP_SIZE: int = 16

# ---------------------------------------------------------------------------
# ModalityBundle
# ---------------------------------------------------------------------------

@dataclass
class ModalityBundle:
    """One timestep of structured sensor data.

    lidar:    (N, 3) float64 point cloud in room coordinates, or None
    pressure: (16, 16) float64 pressure map over bed area, or None
    """
    client_id:      str
    timestamp:      float
    lidar:          Optional[np.ndarray]   # shape (N, 3) or None
    pressure:       Optional[np.ndarray]   # shape (16, 16) or None

    # Ground truth labels (from BehaviorModel)
    semantic_state: str
    posture_state:  str
    bed_zone:       str
    abnormal_phase: int

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_ellipsoid(
    centre: tuple[float, float, float],
    axes: tuple[float, float, float],
    n: int,
    rng: np.random.Generator,
    noise_scale: float = 1.0,
) -> np.ndarray:
    """Sample n points from a 3D Gaussian ellipsoid."""
    sigma = np.array(axes) * noise_scale
    pts = rng.normal(loc=centre, scale=sigma, size=(n, 3))
    return pts.astype(np.float64)

def _add_motion_blur(
    pts: np.ndarray,
    velocity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add motion blur: translate points by random fraction of velocity."""
    if velocity < 0.05:
        return pts
    blur = rng.uniform(-velocity * 0.3, velocity * 0.3, size=(1, 3))
    blur[:, 2] = 0.0  # no vertical blur
    return pts + blur

def _apply_occlusion(
    pts: np.ndarray,
    occlusion: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly drop points to simulate occlusion."""
    if occlusion <= 0.0:
        return pts
    keep = rng.random(len(pts)) > occlusion
    return pts[keep] if keep.any() else pts[:1]

# ---------------------------------------------------------------------------
# ObservationModel (internal implementation)
# ---------------------------------------------------------------------------

class ObservationModel:
    """Maps LatentState -> ModalityBundle.

    Parameters
    ----------
    config : ClientConfig
    rng    : np.random.Generator
    """

    def __init__(self, config: ClientConfig, rng: np.random.Generator) -> None:
        self._has_lidar   = config.has_channel('lidar')
        self._has_bed     = config.has_channel('bed')
        self._noise       = config.sensor_noise_level
        self._occlusion   = config.lidar_occlusion
        self._bed_pos     = config.bed_position      # (bx, by) centre
        self._bed_radius  = config.bed_radius
        self._rng         = rng
        self._client_id   = config.client_id

    # ------------------------------------------------------------------
    # LiDAR point cloud generation
    # ------------------------------------------------------------------

    def _generate_lidar(self, ls: LatentState) -> Optional[np.ndarray]:
        """Generate 3D point cloud for this timestep."""
        if not self._has_lidar:
            return None

        rng = self._rng
        ns = self._noise

        if ls.state == SemanticState.ABSENT:
            # Background noise only
            bg = rng.uniform(
                low=[0.0, 0.0, 0.0],
                high=[5.0, 5.0, 0.1],
                size=(_N_BACKGROUND, 3)
            )
            return bg.astype(np.float64)

        # Person origin in room coordinates
        ox, oy = ls.x, ls.y
        posture = ls.posture

        parts = []

        # ABNORMAL phase 1: splatter pattern
        if ls.state == SemanticState.ABNORMAL and ls.abnormal_phase == 1:
            splatter = _sample_ellipsoid(
                (ox, oy, 0.15), _SPLATTER_AXES, _N_SPLATTER, rng, ns
            )
            splatter[:, 2] = np.clip(splatter[:, 2], 0.0, 0.5)
            parts.append(splatter)
        else:
            # Normal body parts
            n_torso = _N_TORSO[posture]
            tc = _TORSO_CENTRE[posture]
            ta = _TORSO_AXES[posture]
            torso = _sample_ellipsoid(
                (ox + tc[0], oy + tc[1], tc[2]), ta, n_torso, rng, ns * 0.5
            )
            parts.append(torso)

            hc = _HEAD_CENTRE[posture]
            head = _sample_ellipsoid(
                (ox + hc[0], oy + hc[1], hc[2]), _HEAD_AXES[posture],
                _N_HEAD, rng, ns * 0.4
            )
            parts.append(head)

            lc = _LEGS_CENTRE[posture]
            legs = _sample_ellipsoid(
                (ox + lc[0], oy + lc[1], lc[2]), _LEGS_AXES[posture],
                _N_LEGS, rng, ns * 0.6
            )
            parts.append(legs)

        # Background
        bg = rng.uniform(low=[0, 0, 0], high=[5, 5, 0.05],
                         size=(_N_BACKGROUND, 3))
        parts.append(bg.astype(np.float64))

        pts = np.vstack(parts)

        # Motion blur for WALKING
        if ls.state == SemanticState.WALKING:
            pts = _add_motion_blur(pts, ls.velocity, rng)

        # TRANSITION: asymmetric body spread toward away-from-bed direction
        # Person is mid-movement (sitting up, standing up) -> upper body leans
        # away from bed center, creating a consistent geometric asymmetry
        # vs STATIONARY (symmetric, stable upright/seated posture)
        if ls.state == SemanticState.TRANSITION:
            bx, by = self._bed_pos
            dx = ls.x - bx
            dy = ls.y - by
            dist = float(np.sqrt(dx**2 + dy**2)) + 1e-6
            # Lean direction: away from bed center (consistent, not random)
            lean_dx = (dx / dist) * 0.20
            lean_dy = (dy / dist) * 0.20
            # Apply to upper body only
            z_median = float(np.median(pts[:, 2]))
            upper_mask = pts[:, 2] > z_median
            pts[upper_mask, 0] += lean_dx + rng.normal(0, 0.04, upper_mask.sum())
            pts[upper_mask, 1] += lean_dy + rng.normal(0, 0.04, upper_mask.sum())

        # Occlusion
        pts = _apply_occlusion(pts, self._occlusion, rng)

        # Clip to room boundaries (z >= 0)
        pts[:, 2] = np.clip(pts[:, 2], 0.0, 2.5)

        return pts.astype(np.float64)

    # ------------------------------------------------------------------
    # Pressure map generation
    # ------------------------------------------------------------------

    def _generate_pressure(self, ls: LatentState) -> Optional[np.ndarray]:
        """Generate 16x16 pressure map over bed area."""
        if not self._has_bed:
            return None

        rng = self._rng
        ns  = self._noise
        M   = _PRESSURE_MAP_SIZE
        pmap = np.zeros((M, M), dtype=np.float64)

        if ls.state == SemanticState.ABSENT:
            # Small noise
            pmap += rng.normal(0.0, 0.01 * ns, size=(M, M))
            return np.clip(pmap, 0.0, 1.0)

        bx, by = self._bed_pos
        br     = self._bed_radius

        if ls.bed_zone == BedZone.OFF_BED:
            # No pressure on bed
            pmap += rng.normal(0.0, 0.02 * ns, size=(M, M))
            return np.clip(pmap, 0.0, 1.0)

        # Map person position to grid coordinates
        gx = (ls.x - (bx - br)) / (2 * br) * M
        gy = (ls.y - (by - br)) / (2 * br) * M
        gx = float(np.clip(gx, 0, M - 1))
        gy = float(np.clip(gy, 0, M - 1))

        # Pressure amplitude depends on posture and zone
        if ls.posture == Posture.LYING and ls.bed_zone == BedZone.ON_BED:
            amplitude = 0.90
            sigma_x, sigma_y = 3.5, 2.0   # spread along body axis
        elif ls.posture == Posture.SEATED:
            amplitude = 0.65
            sigma_x, sigma_y = 2.0, 2.0
        elif ls.bed_zone == BedZone.BED_EDGE:
            amplitude = 0.45
            sigma_x, sigma_y = 1.5, 1.5
        else:
            amplitude = 0.50
            sigma_x, sigma_y = 2.0, 2.0

        # 2D Gaussian
        xs = np.arange(M)
        ys = np.arange(M)
        xx, yy = np.meshgrid(xs, ys, indexing='ij')
        gauss = amplitude * np.exp(
            -(((xx - gx) / sigma_x) ** 2 + ((yy - gy) / sigma_y) ** 2) / 2.0
        )

        # ABNORMAL: sudden pressure drop (person fell off bed)
        if ls.state == SemanticState.ABNORMAL and ls.abnormal_phase == 1:
            gauss *= 0.1   # impact -> pressure collapses

        pmap = gauss + rng.normal(0.0, 0.03 * ns, size=(M, M))
        return np.clip(pmap, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def observe(self, ls: LatentState, timestamp: float) -> ModalityBundle:
        return ModalityBundle(
            client_id      = self._client_id,
            timestamp      = timestamp,
            lidar          = self._generate_lidar(ls),
            pressure       = self._generate_pressure(ls),
            semantic_state = ls.state.value,
            posture_state  = ls.posture.value,
            bed_zone       = ls.bed_zone.value,
            abnormal_phase = ls.abnormal_phase,
        )
