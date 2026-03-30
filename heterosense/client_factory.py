"""HeteroSense ClientFactory: N-client modality configuration."""
from __future__ import annotations

import copy
import numpy as np

# ---------------------------------------------------------------------------
# Modality presets
# ---------------------------------------------------------------------------

MODALITY_PRESETS: dict[str, list[str]] = {
    'both':          ['lidar', 'bed'],
    'lidar_only':    ['lidar'],
    'pressure_only': ['bed'],
}

# Default sensor parameters per modality preset
_SENSOR_DEFAULTS: dict[str, dict] = {
    'both': {
        'lidar_height_gain': 1.0,
        'lidar_motion_gain': 1.0,
        'lidar_floor_gain':  1.0,
        'lidar_occlusion':   0.0,
        'bed_pressure_gain': 1.0,
        'bed_edge_sensitivity': 0.8,
        'sensor_noise_level': 1.0,
        'abnormal_rate': 0.01,
    },
    'lidar_only': {
        'lidar_height_gain': 0.9,
        'lidar_motion_gain': 0.9,
        'lidar_floor_gain':  0.8,
        'lidar_occlusion':   0.1,
        'sensor_noise_level': 2.0,
        'abnormal_rate': 0.005,
    },
    'pressure_only': {
        'bed_pressure_gain': 1.0,
        'bed_edge_sensitivity': 0.8,
        'sensor_noise_level': 1.0,
        'abnormal_rate': 0.002,
    },
}

# Default assignment order for round-robin
_DEFAULT_PATTERNS = ['both', 'lidar_only', 'pressure_only']


class ClientFactory:
    """Factory for generating N-client configuration dictionaries.

    All methods return a list of client config dicts that can be passed
    directly to ConfigurationManager.from_clients().
    """

    @classmethod
    def make(
        cls,
        n: int,
        strategy: str = 'round_robin',
        patterns: list[str] | None = None,
        seed: int = 0,
        client_id_prefix: str = '',
        base_overrides: dict | None = None,
    ) -> list[dict]:
        """Generate N client configuration dicts.

        Parameters
        ----------
        n : int
            Number of clients. Must be >= 1.
        strategy : str
            Assignment strategy. One of:
              'round_robin' -- cycle through patterns (default)
              'random'      -- randomly assign patterns (uses seed)
              'explicit'    -- patterns must have exactly n entries
              'uniform'     -- all clients get patterns[0]
        patterns : list[str] or None
            Modality pattern names. Defaults to
            ['both', 'lidar_only', 'pressure_only'] for round_robin/random,
            required for 'explicit' and 'uniform'.
        seed : int
            Random seed for 'random' strategy.
        client_id_prefix : str
            Prefix for client IDs. IDs will be '{prefix}{i}'.
        base_overrides : dict or None
            Additional config keys applied to all clients after defaults.

        Returns
        -------
        list[dict]
            List of client config dicts.
        """
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        if strategy not in ('round_robin', 'random', 'explicit', 'uniform'):
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                "Choose: 'round_robin', 'random', 'explicit', 'uniform'")

        if patterns is None:
            if strategy == 'explicit':
                raise ValueError("'explicit' strategy requires patterns list")
            if strategy == 'uniform':
                raise ValueError("'uniform' strategy requires patterns list")
            patterns = _DEFAULT_PATTERNS

        for p in patterns:
            if p not in MODALITY_PRESETS:
                raise ValueError(
                    f"Unknown modality preset {p!r}. "
                    f"Choose: {list(MODALITY_PRESETS)}")

        # Assign pattern to each client
        assigned = cls._assign(n, strategy, patterns, seed)

        clients = []
        for i, pattern in enumerate(assigned):
            cid = f"{client_id_prefix}{i}"
            client = cls._make_single(cid, pattern, base_overrides or {})
            clients.append(client)
        return clients

    @classmethod
    def _assign(cls, n: int, strategy: str,
                patterns: list[str], seed: int) -> list[str]:
        if strategy == 'round_robin':
            return [patterns[i % len(patterns)] for i in range(n)]
        if strategy == 'uniform':
            return [patterns[0]] * n
        if strategy == 'explicit':
            if len(patterns) != n:
                raise ValueError(
                    f"'explicit' strategy: patterns has {len(patterns)} entries "
                    f"but n={n}")
            return list(patterns)
        if strategy == 'random':
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, len(patterns), size=n)
            return [patterns[int(i)] for i in idx]
        raise ValueError(f"Unknown strategy: {strategy}")

    @classmethod
    def _make_single(cls, client_id: str, pattern: str,
                     overrides: dict) -> dict:
        defaults = copy.deepcopy(_SENSOR_DEFAULTS[pattern])
        client = {
            'client_id': client_id,
            'channel_availability': MODALITY_PRESETS[pattern],
            'bed_position':  [2.5, 2.5],
            'bed_radius':    0.8,
        }
        client.update(defaults)
        client.update(overrides)
        return client

    # ------------------------------------------------------------------
    # Convenience presets
    # ------------------------------------------------------------------

    @classmethod
    def heterogeneous(cls, n: int, **kwargs) -> list[dict]:
        """N clients with round-robin modality assignment (default)."""
        return cls.make(n, strategy='round_robin', **kwargs)

    @classmethod
    def homogeneous(cls, n: int, modality: str = 'both', **kwargs) -> list[dict]:
        """N clients all with the same modality."""
        return cls.make(n, strategy='uniform',
                        patterns=[modality], **kwargs)

    @classmethod
    def explicit(cls, patterns: list[str], **kwargs) -> list[dict]:
        """Clients with explicitly specified per-client modalities."""
        return cls.make(len(patterns), strategy='explicit',
                        patterns=patterns, **kwargs)
