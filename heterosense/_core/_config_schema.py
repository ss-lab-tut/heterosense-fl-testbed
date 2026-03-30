"""BehaSim config schema: typed dataclasses for simulator configuration."""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _require_positive(value: float, name: str) -> float:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _require_nonnegative(value: float, name: str) -> float:
    """Allow zero (noise-free oracle / ablation baseline) but reject negative values."""
    if value < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return value


def _require_unit_interval(value: float, name: str) -> float:
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _require_positive_int(value: int, name: str) -> int:
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class RoomConfig:
    width:  float = 5.0
    height: float = 5.0

    def __post_init__(self) -> None:
        _require_positive(self.width,  "room.width")
        _require_positive(self.height, "room.height")

    @classmethod
    def from_dict(cls, d: dict) -> RoomConfig:
        return cls(width=float(d.get("width", 5.0)), height=float(d.get("height", 5.0)))


@dataclass
class SensorLayoutConfig:
    """Physical sensor layout in the room.

    v1.0: LiDAR position only. Bed position is per-client (ClientConfig.bed_position).
    """
    lidar_position: list[float] = field(default_factory=lambda: [0.0, 0.0])

    def __post_init__(self) -> None:
        if len(self.lidar_position) != 2:
            raise ValueError(f"lidar_position must be [x, y], got {self.lidar_position}")

    @classmethod
    def from_dict(cls, d: dict) -> SensorLayoutConfig:
        return cls(
            lidar_position=d.get("lidar_position", [0.0, 0.0]),
        )


@dataclass
class PerturbationConfig:
    """Noise sigmas per modality.

    lidar_sigma  applies to all LiDAR-like channels.
    bed_sigma    applies to all bed-pressure-like channels.
    """
    lidar_sigma: float = 0.03
    bed_sigma:   float = 0.04

    def __post_init__(self) -> None:
        _require_nonnegative(self.lidar_sigma, "perturbation.lidar_sigma")
        _require_nonnegative(self.bed_sigma,   "perturbation.bed_sigma")

    @classmethod
    def from_dict(cls, d: dict) -> PerturbationConfig:
        return cls(
            lidar_sigma=float(d.get("lidar_sigma", 0.03)),
            bed_sigma=float(d.get("bed_sigma", 0.04)),
        )


@dataclass
class ClientConfig:
    """Per-client configuration (v1.0).

    channel_availability : sequence of "lidar" | "bed"
    sensor_noise_level   : global noise multiplier (applied to all sigmas)
    abnormal_rate        : direct entry probability into ABNORMAL from any
                             non-abnormal state (STATIONARY, WALKING, TRANSITION).
                             rate=0 -> zero ABNORMAL; ABNORMAL self-loop (0.20) is independent.

    LiDAR-specific
    ~~~~~~~~~~~~~~
    lidar_height_gain    : alpha_h  -- client sensitivity for height channel
    lidar_motion_gain    : alpha_m  -- client sensitivity for motion channel
    lidar_floor_gain     : alpha_f  -- client sensitivity for floor-proximity channel
    lidar_occlusion      : fraction of LiDAR signal lost due to occlusion (range 0 to 1)

    Bed-specific
    ~~~~~~~~~~~~
    bed_pressure_gain    : beta_s  -- overall bed pressure sensitivity
    bed_edge_sensitivity : beta_e  -- edge-zone detection sensitivity
    bed_position         : (x, y) centre of the bed in room coordinates
    bed_radius           : radius defining ON_BED zone [m]
    """
    client_id:           str
    room_width:          float = 5.0
    room_height:         float = 5.0
    channel_availability: tuple[str, ...] = ("lidar", "bed")
    sensor_noise_level:  float = 1.0
    abnormal_rate:       float = 0.01

    # Resolved from global layout/perturbation
    lidar_position:      tuple[float, float] = (0.0, 0.0)
    lidar_sigma:         float = 0.03
    bed_sigma:           float = 0.04

    # LiDAR gains (client heterogeneity)
    lidar_height_gain:   float = 1.0
    lidar_motion_gain:   float = 1.0
    lidar_floor_gain:    float = 1.0
    lidar_occlusion:     float = 0.0   # (range 0 to 1)

    # Bed gains (client heterogeneity)
    bed_pressure_gain:   float = 1.0
    bed_edge_sensitivity: float = 0.8
    bed_position:        tuple[float, float] = (2.5, 2.5)
    bed_radius:          float = 0.8

    _ALLOWED_CHANNELS: frozenset = frozenset({"lidar", "bed"})

    def __post_init__(self) -> None:
        _require_positive(self.room_width,         f"client {self.client_id}: room_width")
        _require_positive(self.room_height,        f"client {self.client_id}: room_height")
        _require_positive(self.bed_radius,         f"client {self.client_id}: bed_radius")
        if self.sensor_noise_level < 0:
            raise ValueError(f"client {self.client_id}: sensor_noise_level must be >= 0")
        _require_unit_interval(self.abnormal_rate,  f"client {self.client_id}: abnormal_rate")
        _require_unit_interval(self.lidar_occlusion,f"client {self.client_id}: lidar_occlusion")
        unknown = set(self.channel_availability) - self._ALLOWED_CHANNELS
        if unknown:
            raise ValueError(
                f"client {self.client_id}: unknown channel(s) {unknown}. "
                f"Allowed: {self._ALLOWED_CHANNELS}"
            )
        if len(self.bed_position) != 2:
            raise ValueError(f"client {self.client_id}: bed_position must be (x, y), got {self.bed_position}")
        if len(self.lidar_position) != 2:
            raise ValueError(f"client {self.client_id}: lidar_position must be (x, y), got {self.lidar_position}")

    @classmethod
    def from_dict(
        cls,
        d: dict,
        layout: SensorLayoutConfig,
        perturbation: PerturbationConfig,
        global_room: RoomConfig,
    ) -> ClientConfig:
        availability = d.get("channel_availability", ["lidar", "bed"])
        lp = layout.lidar_position
        bp = d.get("bed_position", [2.5, 2.5])
        return cls(
            client_id=str(d["client_id"]),
            room_width=float(d.get("room_width",  global_room.width)),
            room_height=float(d.get("room_height", global_room.height)),
            channel_availability=tuple(availability),
            sensor_noise_level=float(d.get("sensor_noise_level", 1.0)),
            abnormal_rate=float(d.get("abnormal_rate", 0.01)),
            lidar_position=(float(lp[0]), float(lp[1])),
            lidar_sigma=perturbation.lidar_sigma,
            bed_sigma=perturbation.bed_sigma,
            lidar_height_gain=float(d.get("lidar_height_gain", 1.0)),
            lidar_motion_gain=float(d.get("lidar_motion_gain", 1.0)),
            lidar_floor_gain=float(d.get("lidar_floor_gain",  1.0)),
            lidar_occlusion=float(d.get("lidar_occlusion", 0.0)),
            bed_pressure_gain=float(d.get("bed_pressure_gain", 1.0)),
            bed_edge_sensitivity=float(d.get("bed_edge_sensitivity", 0.8)),
            bed_position=(float(bp[0]), float(bp[1])),
            bed_radius=float(d.get("bed_radius", 0.8)),
        )

    def has_channel(self, ch: str) -> bool:
        return ch in self.channel_availability


# ---------------------------------------------------------------------------
# Top-level SimConfig
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    random_seed:  int   = 42
    n_steps:      int   = 1000
    delta_t:      float = 1.0
    room:         RoomConfig         = field(default_factory=RoomConfig)
    sensors:      SensorLayoutConfig = field(default_factory=SensorLayoutConfig)
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    clients:      list[ClientConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        _require_positive_int(self.n_steps, "n_steps")
        _require_positive(self.delta_t,     "delta_t")
        if not self.clients:
            raise ValueError("SimConfig.clients must not be empty")

    @classmethod
    def from_dict(cls, d: dict) -> SimConfig:
        room        = RoomConfig.from_dict(d.get("room", {}))
        sensors     = SensorLayoutConfig.from_dict(d.get("sensors", {}))
        perturbation = PerturbationConfig.from_dict(d.get("perturbation", {}))
        clients     = [
            ClientConfig.from_dict(c, sensors, perturbation, room)
            for c in d.get("clients", [])
        ]
        return cls(
            random_seed=int(d.get("random_seed", 42)),
            n_steps=int(d.get("n_steps", 1000)),
            delta_t=float(d.get("delta_t", 1.0)),
            room=room, sensors=sensors, perturbation=perturbation, clients=clients,
        )
