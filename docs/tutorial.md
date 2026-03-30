# HeteroSense-FL Design Tutorial

**Version 1.0.0**  
Toyohashi University of Technology — Smart Systems Laboratory

---

## Contents

1. [Overview](#1-overview)
2. [Architecture](#2-architecture)
3. [Installation](#3-installation)
4. [Core Concepts](#4-core-concepts)
5. [API Walkthrough](#5-api-walkthrough)
   - 5.1 ClientFactory
   - 5.2 ConfigurationManager
   - 5.3 DatasetBuilder
   - 5.4 TemporalWindowSampler
   - 5.5 run_validation
6. [FL Algorithm Plug-in](#6-fl-algorithm-plug-in)
7. [Reference Benchmarks](#7-reference-benchmarks)
8. [Configuration Reference](#8-configuration-reference)
9. [Extending HeteroSense-FL](#9-extending-heterosense-fl)

---

## 1. Overview

HeteroSense-FL simulates an indoor smart-room sensing environment where each
federated learning (FL) client site is equipped with a **different sensor subset**.
It generates structured multimodal observations — 3D LiDAR point clouds and
16×16 bed pressure maps — for N client sites with configurable per-client
modality availability.

**What it is:**  
A simulation testbed that generates the heterogeneous sensor data needed
for FL research under realistic modality-heterogeneous conditions.

**What it is not:**  
An FL execution framework. Use [Flower](https://flower.dev/) or
[FedML](https://fedml.ai/) to run FL algorithms on top of the data
HeteroSense-FL generates.

### Key capabilities

| Capability | Description |
|-----------|-------------|
| `ClientFactory` | Configure N clients with different sensor subsets |
| `DatasetBuilder` | Generate `{client_id: [ModalityBundle]}` time series |
| `TemporalWindowSampler` | Sliding-window iterator for temporal encoder development |
| `run_validation` | Automated observation integrity checks V1–V4 |
| `heterosense-benchmark` | One-command reproducible reference benchmarks |

---

## 2. Architecture

HeteroSense-FL is structured in three layers:

```
┌─────────────────────────────────────────────────────┐
│  INTERFACE LAYER  (public API)                       │
│  ClientFactory · ConfigurationManager               │
│  DatasetBuilder · TemporalWindowSampler · run_validation │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  OBSERVATION LAYER  (heterosense/_core/)             │
│  _ObservationModel  — 3D point cloud rendering       │
│  _ObservationModelExt — pressure map rendering       │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│  LATENT WORLD LAYER  (heterosense/_core/)            │
│  _BehaviorModel  — Markov-chain state generator      │
│  _ConfigSchema   — SimConfig / ClientConfig           │
│  _ConfigManager  — YAML loading + programmatic API   │
└─────────────────────────────────────────────────────┘
```

### Latent states

Each time step produces a `LatentState` with five semantic states:

| State | Description |
|-------|-------------|
| `ABSENT` | No resident present |
| `STATIONARY` | Resident standing or sitting still |
| `WALKING` | Resident moving around the room |
| `TRANSITION` | Resident moving between postures |
| `ABNORMAL` | Fall event (2-phase: impact → rest) |

Posture: `UPRIGHT`, `SEATED`, `LYING`  
Bed zone: `ON_BED`, `BED_EDGE`, `OFF_BED`

### Observation models

**LiDAR point cloud** — Gaussian ellipsoid per body part (torso, head, legs).
Shape and centroid vary with posture. Returns `numpy.ndarray` of shape `(N, 3)`.
Returns `None` when the client has no LiDAR sensor.

**Bed pressure map** — 16×16 grid; Gaussian kernel centred on the body contact
point, modulated by posture and bed zone. Returns `numpy.ndarray` of shape
`(16, 16)`. Returns `None` when the client has no pressure sensor.

---

## 3. Installation

```bash
pip install heterosense-fl
```

**Requirements:** Python 3.9+, numpy ≥ 1.24, PyYAML ≥ 6.0

**Development install:**

```bash
git clone https://github.com/ss-lab-tut/heterosense-fl
cd heterosense-fl
pip install -e ".[dev]"
pytest tests/ -v
```

**Verify:**

```python
import heterosense
print(heterosense.__version__)   # 1.0.0
```

---

## 4. Core Concepts

### ModalityBundle

A `ModalityBundle` is the output for one client at one time step:

```python
@dataclass
class ModalityBundle:
    client_id:      str
    timestamp:      float
    lidar:          Optional[np.ndarray]   # shape (N, 3) or None
    pressure:       Optional[np.ndarray]   # shape (16, 16) or None
    semantic_state: str   # 'ABSENT' | 'STATIONARY' | 'WALKING' | 'TRANSITION' | 'ABNORMAL'
    posture_state:  str   # 'UPRIGHT' | 'SEATED' | 'LYING'
    bed_zone:       str   # 'ON_BED' | 'BED_EDGE' | 'OFF_BED'
    abnormal_phase: int   # 0 = normal; 1 = impact; ≥2 = rest
```

### Modality patterns

```python
patterns = ['both', 'lidar', 'bed']
# 'both'  → lidar=array, pressure=array
# 'lidar' → lidar=array, pressure=None
# 'bed'   → lidar=None,  pressure=array
```

### Dataset structure

```python
data: dict[str, list[ModalityBundle]]
# key   = client_id  ('0', '1', ..., 'N-1')
# value = list of ModalityBundle, length = n_steps
```

---

## 5. API Walkthrough

### 5.1 ClientFactory

`ClientFactory.make(n, strategy, patterns, seed)` returns a list of N client
configuration dicts.

**Strategies:**

| Strategy | Description | Example |
|----------|-------------|---------|
| `'round_robin'` | Cycle through `patterns` list (default) | 0→both, 1→lidar, 2→bed, 3→both, … |
| `'uniform'` | All clients get `patterns[0]` | All → both |
| `'random'` | Random assignment (deterministic with `seed`) | |
| `'explicit'` | `patterns` must have exactly N entries | |

```python
from heterosense import ClientFactory

# 10-client round-robin (default patterns: both, lidar, bed)
clients = ClientFactory.make(10, strategy='round_robin')

# All clients with both sensors
clients = ClientFactory.make(5, strategy='round_robin', patterns=['both'])

# Explicit per-client assignment
clients = ClientFactory.make(3, strategy='explicit',
                             patterns=['both', 'lidar', 'bed'])

# Random (reproducible)
clients = ClientFactory.make(10, strategy='random', seed=42)
```

**Output structure:**

```python
clients[0]
# {
#   'client_id': '0',
#   'channel_availability': ['lidar', 'bed'],
#   'sensor_noise_level': 1.0,
#   'abnormal_rate': 0.01,
#   ...
# }
```

---

### 5.2 ConfigurationManager

`ConfigurationManager.from_clients(clients, n_steps)` builds a `SimConfig`
from a ClientFactory output without manual YAML editing.

```python
from heterosense import ConfigurationManager as CM

cfg = CM.from_clients(clients, n_steps=20_000)
sc  = cfg.to_sim_config()   # SimConfig object

# Inspect per-client config
for c in sc.clients:
    print(c.client_id, c.channel_availability, c.abnormal_rate)
```

**Loading from YAML** (advanced):

```python
cm = ConfigurationManager('configs/default.yaml')
sc = cm.to_sim_config()
```

---

### 5.3 DatasetBuilder

`DatasetBuilder(sim_config).build()` generates the full dataset.

```python
from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder

clients = ClientFactory.make(10, strategy='round_robin', seed=42)
cfg     = CM.from_clients(clients, n_steps=20_000)
data    = DatasetBuilder(cfg.to_sim_config()).build()
# data: dict[str, list[ModalityBundle]]
```

**Determinism:** Each client uses a deterministic seed derived from the global
`random_seed` and `client_id`. Identical inputs always produce identical outputs.

**Inspecting output:**

```python
# Check modalities per client
for cid, bundles in data.items():
    has_lidar    = any(b.lidar    is not None for b in bundles)
    has_pressure = any(b.pressure is not None for b in bundles)
    print(f'client {cid}: lidar={has_lidar}  bed={has_pressure}')

# Sample a ModalityBundle
b = data['0'][100]
print(b.semantic_state, b.lidar.shape if b.lidar is not None else None)
```

---

### 5.4 TemporalWindowSampler

`TemporalWindowSampler(bundles, window)` slides a window of length `k` over
a ModalityBundle list. The centre bundle at `center_idx()` provides the label.

```python
from heterosense import TemporalWindowSampler

sampler = TemporalWindowSampler(data['0'], window=3)
print(f'windows: {sum(1 for _ in sampler)}')   # n_steps - window + 1
print(f'center_idx: {sampler.center_idx()}')    # 1

for window in sampler:
    # Built-in scalar helpers (replace with your own encoder)
    z     = TemporalWindowSampler.lidar_z_series(window)    # (3,) mean z per step
    p     = TemporalWindowSampler.pressure_series(window)   # (3,) mean pressure per step
    label = TemporalWindowSampler.center_label(window, sampler.center_idx())
    # label ∈ {'ABSENT','STATIONARY','WALKING','TRANSITION','ABNORMAL'}
```

**Replacing the built-in helpers:**

```python
import numpy as np

def my_temporal_encoder(window: list) -> np.ndarray:
    """Replace this with LSTM, Transformer, CNN, etc."""
    # window: list of ModalityBundle, length = window_size
    # Return: fixed-length feature vector
    features = []
    for bundle in window:
        if bundle.lidar is not None:
            features.append(bundle.lidar[:, 2].mean())   # mean z
        else:
            features.append(0.0)
    return np.array(features)
```

---

### 5.5 run_validation

`run_validation(data, client_modalities)` runs four automated integrity checks
on the generated dataset.

```python
from heterosense import run_validation

client_modalities = {c.client_id: list(c.channel_availability)
                     for c in cfg.to_sim_config().clients}

results = run_validation(data, client_modalities)
for r in results:
    icon = 'OK' if r.passed else 'FAIL'
    print(f'[{icon}] {r.name}: {r.reason}')
```

**The four checks:**

| Check | What it verifies |
|-------|-----------------|
| **V1 Geometry Validity** | UPRIGHT mean_z > LYING mean_z in LiDAR clouds |
| **V2 Fall Pattern Validity** | ABNORMAL phase-1 frames have high floor-point ratio (z < 0.3 m) |
| **V3 Pressure Separability** | ON_BED pressure mean >> OFF_BED pressure mean |
| **V4 Modality Integrity** | Absent sensors consistently return `None`; present sensors never return `None` |

---

## 6. FL Algorithm Plug-in

`examples/temporal_plugin_example.py` provides a complete FL loop with three
clearly marked replacement points. Run it as a starting point:

```bash
python examples/temporal_plugin_example.py
```

### Replacement points

```python
# [A] REPLACE: temporal feature extractor
def extract_features(window, sampler) -> np.ndarray:
    """Return fixed-length feature vector. Replace with LSTM, Transformer, etc."""
    z = TemporalWindowSampler.lidar_z_series(window)
    p = TemporalWindowSampler.pressure_series(window)
    return np.concatenate([z, p])


# [B] REPLACE: local training loop
def train_local(weights, X, y) -> list:
    """One FL round of local training. Replace with PyTorch, TF, sklearn, etc."""
    ...


# [C] REPLACE: FL aggregation algorithm
def aggregate(local_weights_list) -> list:
    """Aggregate local params. Replace with FedProx, SCAFFOLD, FedNova, etc."""
    return [np.mean([ws[i] for ws in local_weights_list], 0)
            for i in range(len(local_weights_list[0]))]
```

---

## 7. Reference Benchmarks

```bash
heterosense-benchmark          # full Table 3
heterosense-demo               # quick 3-client demo
```

**Reproducing Table 3 manually:**

```python
from heterosense._scripts.run_benchmark import main
main()
```

**Benchmark configuration:**  
Seeds: `{42, 123, 7}` · n_steps: `3000` · FL rounds: `3` · Encoder: TinyMLP (numpy-only)

| N | Pattern | Local std. | FedAvg std. | Local bal. | FedAvg bal. |
|---|---------|------------|-------------|------------|-------------|
| 3 | homogeneous | 0.366±0.009 | 0.419±0.023 | 0.252±0.004 | 0.298±0.017 |
| 3 | round-robin | 0.379±0.021 | 0.383±0.009 | 0.285±0.011 | 0.283±0.007 |
| 10 | round-robin | 0.361±0.006 | 0.368±0.017 | 0.280±0.003 | 0.287±0.014 |
| 20 | round-robin | 0.362±0.005 | 0.368±0.006 | 0.282±0.004 | 0.299±0.012 |
| 50 | round-robin | 0.366±0.003 | 0.381±0.017 | 0.277±0.003 | 0.300±0.015 |

> **Note:** Standard accuracy is dominated by the ABSENT class (~40% of steps).
> Balanced accuracy corrects for this and reveals FedAvg's advantage on minority
> activity classes.

---

## 8. Configuration Reference

### SimConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `random_seed` | 42 | Global RNG seed |
| `n_steps` | 1000 | Time steps per client |
| `delta_t` | 1.0 | Time step duration (seconds) |
| `room.width` | 5.0 | Room width (metres) |
| `room.height` | 5.0 | Room height (metres) |

### ClientConfig fields

| Field | Default | Description |
|-------|---------|-------------|
| `channel_availability` | `['lidar','bed']` | Active sensors |
| `sensor_noise_level` | 1.0 | Noise multiplier (1.0 = default) |
| `abnormal_rate` | 0.01 | ABNORMAL state entry probability |
| `lidar_sigma` | 0.03 | LiDAR point cloud noise std |
| `bed_sigma` | 0.04 | Pressure map noise std |
| `bed_position` | (2.5, 2.5) | Bed centre (x, y) in room coordinates |
| `bed_radius` | 0.8 | ON_BED zone radius (metres) |

### Markov transition matrix

Default transition probabilities (rows = current state, cols = next state):

| | ABSENT | STATIONARY | WALKING | TRANSITION | ABNORMAL |
|-|--------|------------|---------|------------|----------|
| **ABSENT** | 0.70 | 0.00 | 0.30 | 0.00 | 0.00 |
| **STATIONARY** | 0.00 | 0.60 | 0.25 | 0.14 | 0.01 |
| **WALKING** | 0.05 | 0.25 | 0.55 | 0.14 | 0.01 |
| **TRANSITION** | 0.00 | 0.40 | 0.40 | 0.19 | 0.01 |
| **ABNORMAL** | 0.00 | 0.00 | 0.00 | 0.80 | 0.20 |

Override via YAML `markov.transition_matrix` key.

---

## 9. Extending HeteroSense-FL

### Adding a new modality

1. Add a field to `ModalityBundle` in `heterosense/_core/_observation_model.py`
2. Add a rendering function in the same file
3. Add a channel name to `ClientConfig._ALLOWED_CHANNELS`
4. Update `DatasetBuilder` to call the new renderer
5. Add a V-check in `heterosense/validation.py`

### Using with Flower

```python
import flwr as fl
from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder

# Generate data
data = DatasetBuilder(CM.from_clients(ClientFactory.make(10), n_steps=50_000)
                      .to_sim_config()).build()

# Wrap each client's data in a Flower NumPyClient
class HeteroSenseClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.bundles = data[cid]
    # implement get_parameters / fit / evaluate
    ...

fl.simulation.start_simulation(
    client_fn=lambda cid: HeteroSenseClient(cid),
    num_clients=10,
)
```

---

*HeteroSense-FL v1.0.0 · MIT License · https://github.com/ss-lab-tut/heterosense-fl*
