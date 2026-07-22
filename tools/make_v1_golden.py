"""make_v1_golden.py — regenerate the v1-compat golden fixture.

The v1 backward-compatibility test (tests/test_v1_compat.py) compares the default dataset
against a stored golden: an exact hash of the platform-independent structure, plus float32
LiDAR/pressure arrays checked with a tolerance (see that file's module docstring for why
byte-exact SHA is not portable across CPU architectures).

The current v2 default path is byte-identical to v1.0.0 on a fixed platform, so this
generates the golden from the current default build. Regenerating is an EXPLICIT,
CHANGELOG-documented operation (it redefines the backward-compat reference).

Usage:
    python tools/make_v1_golden.py
Then paste the printed GOLDEN_STRUCT_HASH into tests/test_v1_compat.py.
"""
from pathlib import Path

import numpy as np

from tests.test_v1_compat import _build, _collect, _struct_hash

OUT = Path(__file__).resolve().parent.parent / "tests" / "golden" / "v1_default.npz"


def main() -> None:
    lidar, pressure, structure = _collect(_build())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        lidar=lidar.astype(np.float32),
        pressure=pressure.astype(np.float32),
    )
    size_mb = OUT.stat().st_size / 1e6
    print(f"wrote {OUT}  ({size_mb:.2f} MB)")
    print(f"lidar {lidar.shape}  pressure {pressure.shape}  samples {len(structure)}")
    print(f'GOLDEN_STRUCT_HASH = "{_struct_hash(structure)}"')


if __name__ == "__main__":
    main()
