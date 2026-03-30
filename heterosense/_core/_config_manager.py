"""BehaSim ConfigurationManager: loads and merges YAML config."""
from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml
from heterosense._core._config_schema import SimConfig

DEFAULT_CONFIG: dict = {
    "random_seed": 42,
    "n_steps": 1000,
    "delta_t": 1.0,
    "room":    {"width": 5.0, "height": 5.0},
    "sensors": {
        "lidar_position": [0.0, 0.0],
    },
    "perturbation": {
        "lidar_sigma": 0.03,
        "bed_sigma":   0.04,
    },
    "clients": [
        {   # Full setup: both LiDAR and bed
            "client_id": "0",
            "room_width": 5.0, "room_height": 5.0,
            "channel_availability": ["lidar", "bed"],
            "sensor_noise_level": 1.0,
            "abnormal_rate": 0.01,
            "bed_position": [2.5, 2.5], "bed_radius": 0.8,
            "lidar_height_gain": 1.0, "lidar_motion_gain": 1.0,
            "lidar_floor_gain": 1.0,  "lidar_occlusion": 0.0,
            "bed_pressure_gain": 1.0, "bed_edge_sensitivity": 0.8,
        },
        {   # Bed only (e.g., no ceiling sensor)
            "client_id": "1",
            "room_width": 4.0, "room_height": 4.0,
            "channel_availability": ["bed"],
            "sensor_noise_level": 1.0,
            "abnormal_rate": 0.002,
            "bed_position": [2.0, 2.0], "bed_radius": 0.8,
            "bed_pressure_gain": 1.0, "bed_edge_sensitivity": 0.8,
        },
        {   # LiDAR only, high noise (occluded room)
            "client_id": "2",
            "room_width": 6.0, "room_height": 6.0,
            "channel_availability": ["lidar"],
            "sensor_noise_level": 3.0,
            "abnormal_rate": 0.0005,
            "lidar_height_gain": 0.8, "lidar_motion_gain": 0.9,
            "lidar_floor_gain": 0.7,  "lidar_occlusion": 0.2,
        },
    ],
}


class ConfigurationManager:
    def __init__(self, config_path: str | Path | None = None) -> None:
        import copy
        self.config: dict = copy.deepcopy(DEFAULT_CONFIG)
        if config_path is not None:
            self._load_yaml(Path(config_path))

    def _load_yaml(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Config file not found: {path}\n"
                f"Use ConfigurationManager(None) for built-in defaults, "
                f"or provide a valid YAML path."
            )
        try:
            with open(path, encoding="utf-8") as f:
                user_config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML in {path}: {exc}") from exc
        if user_config:
            self._deep_update(self.config, user_config)

    def _deep_update(self, base: dict, override: dict) -> None:
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def to_sim_config(self) -> SimConfig:
        return SimConfig.from_dict(self.config)

    def get(self, *keys: str, default: Any = None) -> Any:
        node: Any = self.config
        for k in keys:
            if isinstance(node, dict) and k in node:
                node = node[k]
            else:
                return default
        return node

    @classmethod
    def from_clients(
        cls,
        clients: list,
        n_steps: int = 20000,
        random_seed: int = 42,
        config_path=None,
    ) -> 'ConfigurationManager':
        """Create a ConfigurationManager from a client list.

        Designed to work with ClientFactory.make() output.

        Parameters
        ----------
        clients : list[dict]
            List of client config dicts, e.g. from ClientFactory.make().
        n_steps : int
            Number of simulation steps. Default 20000.
        random_seed : int
            Global random seed. Default 42.
        config_path : str or None
            Optional YAML config path as base. None uses defaults.

        Returns
        -------
        ConfigurationManager

        Example
        -------
            from heterosense.client_factory import ClientFactory
            from heterosense._core._config_manager import ConfigurationManager
            from heterosense.dataset_builder import DatasetBuilder

            cfg = ConfigurationManager.from_clients(
                ClientFactory.make(10, strategy='round_robin'),
                n_steps=20000,
            )
            data = DatasetBuilder(cfg.to_sim_config()).build()
        """
        mgr = cls(config_path)
        mgr.config['clients'] = clients
        mgr.config['n_steps'] = n_steps
        mgr.config['random_seed'] = random_seed
        return mgr
