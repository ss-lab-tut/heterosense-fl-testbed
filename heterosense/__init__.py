"""
HeteroSense-FL: A multimodal simulation testbed for
modality-heterogeneous federated learning research.

Public API
----------
>>> from heterosense import ClientFactory, ConfigurationManager
>>> from heterosense import DatasetBuilder, TemporalWindowSampler
>>> from heterosense import ModalityBundle, run_validation
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("heterosense-fl")
except PackageNotFoundError:
    import re
    import pathlib
    _toml = (pathlib.Path(__file__).parent.parent / "pyproject.toml").read_text()
    m = re.search(r'version\s*=\s*"([^"]+)"', _toml)
    __version__ = m.group(1) if m else "unknown"

# Public API
from heterosense._core._config_schema import SimConfig, ClientConfig
from heterosense._core._config_manager import ConfigurationManager
from heterosense._core._behavior_model import (
    BehaviorModel, LatentState, SemanticState, Posture, BedZone, AbnormalType,
)
from heterosense._core._observation_model import ModalityBundle
from heterosense.client_factory import ClientFactory
from heterosense.dataset_builder import DatasetBuilder
from heterosense.temporal_window_sampler import TemporalWindowSampler
from heterosense.validation import run_validation

__all__ = [
    "ClientFactory",
    "ConfigurationManager",
    "DatasetBuilder",
    "TemporalWindowSampler",
    "ModalityBundle",
    "run_validation",
    "BehaviorModel",
    "LatentState",
    "SemanticState",
    "Posture",
    "BedZone",
    "AbnormalType",
    "SimConfig",
    "ClientConfig",
    "__version__",
]
