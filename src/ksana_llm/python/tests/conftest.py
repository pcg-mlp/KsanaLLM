# tests/conftest.py

import os
import pytest
from utils import read_from_csv


@pytest.fixture(scope="session")
def default_ksana_yaml_path():
    """
    Fixture to provide the default ksana_yaml file path.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, "../../../../examples/ksana_llm.yaml")


@pytest.fixture(scope="session")
def benchmark_inputs():
    """
    Fixture to provide benchmark inputs from CSV.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(
        current_dir, "../../../../benchmarks/benchmark_input.csv"
    )
    return read_from_csv(csv_path)


def pytest_configure(config):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
