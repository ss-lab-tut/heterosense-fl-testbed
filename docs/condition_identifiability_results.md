# Observable condition identifiability: controlled pilot

## Question

Can sensor availability and LiDAR occlusion be inferred from the 24 observable
statistics on a trajectory seed that was not used to fit the estimator?

This is the first necessary sub-question for P1. If observation condition is not
identifiable, switching the fall detector by an estimated condition cannot be
expected to improve the recall--false-alert trade-off.

## Design

- Five simulation seeds; each seed is held out once.
- The four other seeds are used to fit a deterministic nearest-centroid model.
- The same behavior-generation identity and seed are used across condition
  profiles, so the comparison changes observation conditions rather than the
  intended latent behavior process.
- Each profile has 800 frames. The first 100 observed frames fit its calibration
  baseline; the remaining 700 frames are evaluated.
- Simulator settings are used only to construct targets. Estimator inputs are
  exactly the values in `FEATURE_NAMES`.
- Availability and occlusion are evaluated separately. Pressure-only frames are
  excluded from the occlusion task.

The compared profiles are both-sensors/clear, both-sensors/occluded,
LiDAR-only/clear, LiDAR-only/occluded, and pressure-only. In this controlled
pilot, occluded means 45% random LiDAR point removal.

## Provisional results

| Task / feature set | Balanced accuracy, mean ± sample SD |
|---|---:|
| Sensor availability, all observable features | 1.000 ± 0.000 |
| Occlusion, all observable features | 1.000 ± 0.000 |
| Occlusion, point count and count ratio removed | 0.515 ± 0.026 |
| Occlusion, LiDAR shape statistics only | 0.516 ± 0.027 |

Sensor availability is an expected sanity check: the permitted feature contract
contains explicit modality-availability flags.

The occlusion result is more important. Perfect classification disappears when
`point_count` and `point_count_ratio` are removed, falling to approximately the
binary chance level of 0.5. The current simulator implements occlusion as random
point removal, so the estimator can recover the generating rule by counting
points. The remaining z-distribution, floor-ratio, and spatial-spread features
do not identify this condition reliably.

## Interpretation

This pilot does **not** establish that wall occlusion can be inferred in a real
room. It establishes that the current random point-drop simulator makes its own
occlusion parameter trivially recoverable through point count. Reporting only
the 1.000 result would overstate evidence for P1.

The negative ablation result changes the next research step:

1. Add structured occluders that remove contiguous, geometry-dependent regions
   rather than independent random points.
2. Hold out occluder geometry, room layout, and severity during evaluation.
3. Test whether features beyond raw point count identify the held-out condition.
4. Only then test whether predicted condition improves a fixed detector through
   dynamic thresholds.

Installation geometry and ceiling-view effects are not represented by this
pilot. Any future ceiling simulation result remains provisional until compared
with real ceiling-mounted LiDAR point clouds.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m heterosense._scripts.condition_identifiability `
  --n-steps 800 `
  --calibration-steps 100 `
  --output results\condition_identifiability_pilot.json
```

The recorded raw result is `results/condition_identifiability_pilot.json`.
