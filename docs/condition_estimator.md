# Condition estimator: observable feature contract

This document fixes the input contract for P1/P2. It prevents simulator-only
information from leaking into the condition estimator.

## Allowed input

`ObservationStatisticsExtractor` produces 24 values from the current sensor
frame and an optional calibration baseline:

- modality flags: LiDAR, pressure, and baseline availability;
- LiDAR statistics: point count, count relative to the calibration baseline,
  z mean/standard deviation/quantiles, z baseline deltas, floor-point ratios,
  x/y spread, and combined xy spread;
- pressure statistics: sum, mean, standard deviation, maximum, active-cell
  ratio, and baseline delta.

The point-count ratio is **current observed count / observed calibration count**.
It is not divided by a hidden pre-occlusion point count. The calibration
baseline uses medians from caller-selected sensor frames and never reads labels.

Missing modalities are encoded as zero-valued statistics plus explicit
availability flags. This lets one estimator handle LiDAR-only, pressure-only,
and multimodal rooms without treating a missing sensor as a real zero reading.

## Prohibited input

The following may be used as training/evaluation targets but must never be
included in the estimator input matrix:

- simulator occlusion or sensor-layout parameters;
- room/facility identity used by B3;
- ground-truth semantic state, posture, bed zone, or abnormal phase;
- any quantity unavailable from a deployed sensor or its calibration period.

Tests explicitly verify that changing all ground-truth fields while keeping the
sensor observations fixed cannot change the extracted features.

## Leakage-free use

1. Select the calibration period using only the training partition.
2. Fit one `ObservationBaseline` per deployment site using that period.
3. Freeze the baseline before validation and test evaluation.
4. Fit preprocessing, the condition estimator, and threshold calibration on
   training/validation data only.
5. Record the feature order (`FEATURE_NAMES`), calibration selection rule, and
   `pressure_active_threshold` in every experiment configuration.

P1 should consume only this feature matrix. B3 may use facility identity because
it is intentionally the facility-specific threshold baseline, but P1/P2 may not.
