"""Shared fixtures for pytest."""
import sys, os, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from heterosense import ClientFactory, ConfigurationManager as CM, DatasetBuilder

@pytest.fixture(scope="session")
def data3():
    sc = CM.from_clients(ClientFactory.make(3, strategy="round_robin", seed=0),
                         n_steps=200).to_sim_config()
    return DatasetBuilder(sc).build(), sc

@pytest.fixture(scope="session")
def bundles0(data3):
    data, _ = data3
    return data["0"]
