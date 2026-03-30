"""HeteroSense validation: automated observation integrity checks."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

# import ModalityBundle
import os
from heterosense._core._observation_model import ModalityBundle

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name:   str
    passed: bool
    reason: str

    def __str__(self):
        status = 'PASS' if self.passed else 'FAIL'
        return f'[{status}] {self.name}: {self.reason}'

# ---------------------------------------------------------------------------
# V1  Geometry Validity
# ---------------------------------------------------------------------------

def validate_geometry(
    bundles: list[ModalityBundle],
    min_samples: int = 5,
) -> CheckResult:
    """STANDING mean_z > LYING mean_z."""

    standing_z, lying_z = [], []

    for b in bundles:
        if b.lidar is None:
            continue
        if b.posture_state == 'UPRIGHT' and b.semantic_state != 'ABSENT':
            standing_z.append(float(b.lidar[:, 2].mean()))
        elif b.posture_state == 'LYING':
            lying_z.append(float(b.lidar[:, 2].mean()))

    if len(standing_z) < min_samples:
        return CheckResult('V1 Geometry Validity', False,
            f'Not enough UPRIGHT samples (got {len(standing_z)}, need {min_samples})')
    if len(lying_z) < min_samples:
        return CheckResult('V1 Geometry Validity', False,
            f'Not enough LYING samples (got {len(lying_z)}, need {min_samples})')

    mean_standing = float(np.mean(standing_z))
    mean_lying    = float(np.mean(lying_z))
    passed = mean_standing > mean_lying

    return CheckResult(
        'V1 Geometry Validity', passed,
        f'UPRIGHT mean_z={mean_standing:.3f}m, LYING mean_z={mean_lying:.3f}m'
        + ('' if passed else ' -- expected UPRIGHT > LYING')
    )

# ---------------------------------------------------------------------------
# V2  Fall Pattern Validity
# ---------------------------------------------------------------------------

def validate_fall_pattern(
    bundles: list[ModalityBundle],
    z_threshold: float = 0.3,
    floor_ratio_min: float = 0.30,
    min_samples: int = 1,
) -> CheckResult:
    """ABNORMAL phase1: z < z_threshold の点が floor_ratio_min 以上。

    Note: z_threshold=0.3 は Phase 0 prior に基づく固定値。
    将来の 3D-aware 化や スケール変更時に brittle になる可能性あり。
    Phase 2 以降で relative threshold への変更を検討すること。
    """

    phase1_bundles = [
        b for b in bundles
        if b.semantic_state == 'ABNORMAL'
        and b.abnormal_phase == 1
        and b.lidar is not None
    ]

    if len(phase1_bundles) < min_samples:
        return CheckResult('V2 Fall Pattern Validity', False,
            f'No ABNORMAL phase1 frames with LiDAR (got {len(phase1_bundles)})')

    ratios = []
    for b in phase1_bundles:
        pts = b.lidar
        floor_pts = np.sum(pts[:, 2] < z_threshold)
        ratios.append(floor_pts / len(pts))

    mean_ratio = float(np.mean(ratios))
    passed = mean_ratio >= floor_ratio_min

    return CheckResult(
        'V2 Fall Pattern Validity', passed,
        f'ABNORMAL phase1: mean floor-point ratio={mean_ratio:.3f} '
        f'(threshold z<{z_threshold}m, need >={floor_ratio_min:.2f})'
        + ('' if passed else ' -- FAIL')
    )

# ---------------------------------------------------------------------------
# V3  Pressure Separability
# ---------------------------------------------------------------------------

def validate_pressure_separability(
    bundles: list[ModalityBundle],
    min_samples: int = 5,
) -> CheckResult:
    """ON_BED pressure_sum > OFF_BED pressure_sum."""

    on_sums, off_sums = [], []

    for b in bundles:
        if b.pressure is None:
            continue
        psum = float(b.pressure.sum())
        if b.bed_zone == 'ON_BED':
            on_sums.append(psum)
        elif b.bed_zone == 'OFF_BED':
            off_sums.append(psum)

    if len(on_sums) < min_samples:
        return CheckResult('V3 Pressure Separability', False,
            f'Not enough ON_BED samples (got {len(on_sums)}, need {min_samples})')
    if len(off_sums) < min_samples:
        return CheckResult('V3 Pressure Separability', False,
            f'Not enough OFF_BED samples (got {len(off_sums)}, need {min_samples})')

    mean_on  = float(np.mean(on_sums))
    mean_off = float(np.mean(off_sums))
    passed = mean_on > mean_off

    return CheckResult(
        'V3 Pressure Separability', passed,
        f'ON_BED mean_sum={mean_on:.3f}, OFF_BED mean_sum={mean_off:.3f}'
        + ('' if passed else ' -- expected ON_BED > OFF_BED')
    )

# ---------------------------------------------------------------------------
# V4  Missing Modality Integrity
# ---------------------------------------------------------------------------

def validate_missing_modality(
    data: dict[str, list[ModalityBundle]],
    client_modalities: dict[str, list[str]],
) -> CheckResult:
    """modality がない client では該当フィールドが全 timestep で None。"""

    violations = []

    for client_id, bundles in data.items():
        modalities = client_modalities.get(client_id, [])
        has_lidar = 'lidar' in modalities
        has_bed   = 'bed'   in modalities

        for b in bundles:
            if not has_lidar and b.lidar is not None:
                violations.append(
                    f'Client {client_id}: lidar should be None but got data')
                break
            if not has_bed and b.pressure is not None:
                violations.append(
                    f'Client {client_id}: pressure should be None but got data')
                break
            if has_lidar and b.lidar is None and b.semantic_state != 'ABSENT':
                violations.append(
                    f'Client {client_id}: lidar is None but client has lidar')
                break
            if has_bed and b.pressure is None:
                violations.append(
                    f'Client {client_id}: pressure is None but client has bed')
                break

    passed = len(violations) == 0
    reason = 'All modality presence/absence correct' if passed else '; '.join(violations)

    return CheckResult('V4 Missing Modality Integrity', passed, reason)

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_validation(
    data: dict[str, list[ModalityBundle]],
    client_modalities: dict[str, list[str]],
) -> list[CheckResult]:
    """Run all observation integrity checks. Returns list of CheckResult."""

    all_bundles = [b for bundles in data.values() for b in bundles]
    lidar_bundles = [b for b in all_bundles if b.lidar is not None]

    results = [
        validate_geometry(lidar_bundles),
        validate_fall_pattern(lidar_bundles),
        validate_pressure_separability(all_bundles),
        validate_missing_modality(data, client_modalities),
    ]
    return results

# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from heterosense._core._config_manager import ConfigurationManager
    from heterosense.dataset_builder import DatasetBuilder

    cfg = ConfigurationManager(None)
    cfg.config['n_steps'] = 2000
    cfg.config['random_seed'] = 42
    sim_cfg = cfg.to_sim_config()

    data = DatasetBuilder(sim_cfg).build()

    client_modalities = {
        c.client_id: c.channel_availability
        for c in sim_cfg.clients
    }

    results = run_validation(data, client_modalities)

    pass  # smoke test removed
    print('=' * 50)
    for r in results:
        print(r)
    print()
    n_pass = sum(r.passed for r in results)
    print(f'Result: {n_pass}/{len(results)} PASS')
