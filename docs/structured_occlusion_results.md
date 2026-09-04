# Structured occlusion and held-out-geometry pilot

## What changed

The legacy `lidar_occlusion` mechanism independently removes each point with a
fixed probability. The new optional `lidar_occluders` configuration describes
walls or furniture as axis-aligned rectangular footprints with a height.

For every LiDAR point, the simulator traces the horizontal line of sight from
the configured LiDAR position. A point is removed when that segment intersects
an obstacle before reaching the point and the point is below the obstacle top.
This produces contiguous, geometry-dependent shadows rather than independent
random missing points.

The operation is deterministic and consumes no random numbers. An empty
`lidar_occluders` list is the default, so the existing default observation path
is unchanged. The legacy random point-drop parameter remains available for
backward compatibility and direct comparison.

## Research question

Can the observable statistics identify meaningful observation degradation when
both the trajectory seed and the wall/furniture geometry are absent from the
estimator's training data?

Five geometries are paired with five held-out seeds. Each fold excludes every
training sample with the test seed and every training sample with the test
geometry. Clear and occluded observations are generated from the same latent
behavior sequence. A frame is labelled degraded only when the paired obstacle
removes at least 10% of its LiDAR points. This paired point-loss value is a
target construction mechanism and is never included in estimator inputs.

## Provisional result

| Observable feature set | Balanced accuracy | Degraded recall | False-positive rate |
|---|---:|---:|---:|
| All 24 features | 0.753 ± 0.202 | 0.673 ± 0.383 | 0.168 ± 0.027 |
| Point count and ratio removed | 0.735 ± 0.205 | 0.638 ± 0.390 | 0.168 ± 0.027 |
| LiDAR shape statistics only | 0.729 ± 0.198 | 0.626 ± 0.376 | 0.168 ± 0.027 |

Unlike the random point-drop pilot, performance remains above chance after raw
point-count features are removed. Structured shadows therefore alter observable
shape statistics, not only the number of points. However, the large standard
deviation shows that this signal does not generalize uniformly.

## Held-out condition analysis

| Held-out geometry | Balanced accuracy | Degraded recall | False-positive rate |
|---|---:|---:|---:|
| Near vertical wall | 0.895 | 0.949 | 0.160 |
| Mid-room vertical wall | 0.854 | 0.854 | 0.147 |
| Near horizontal wall | 0.919 | 1.000 | 0.163 |
| Mid-room horizontal wall | 0.654 | 0.462 | 0.154 |
| Central low furniture | 0.443 | 0.101 | 0.214 |

The estimator transfers reasonably to large wall shadows but fails on the
localized shadow of low central furniture. A single frame often does not contain
enough evidence that a local region is hidden, especially when the current body
position is outside that region.

## Interpretation and next hypothesis

This result rejects the implicit hypothesis that one frame of global summary
statistics is sufficient for all occluder types. The next model should estimate
condition from a temporal calibration window and retain spatial occupancy
information. Candidate additions are sector-wise point counts and sector-wise z
statistics computed relative to an observed baseline. These remain deployable
observations and do not expose simulator parameters.

Only after held-out furniture performance improves should this estimate control
a fall-score threshold in P1. Otherwise dynamic threshold results would be
dominated by condition-estimation errors.

## Limitations

- The current ray test is 2.5-D because LiDAR mounting height is not represented.
- Obstacles are axis-aligned rectangles; rotated walls are not yet supported.
- The result is a simulation pilot, not a real-LiDAR result.
- Ceiling-view conclusions remain provisional until real point clouds are used.
- The recorded run used Python 3.14, outside the versions listed in the package classifiers.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m heterosense._scripts.structured_occlusion_pilot `
  --n-steps 800 `
  --calibration-steps 100 `
  --minimum-loss-ratio 0.10 `
  --output results\structured_occlusion_pilot.json
```
