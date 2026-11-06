"""Smoke tests: confirm the environment and project layout are in place."""

import importlib

import pytest

from src import config

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "scipy",
    "requests",
    "sklearn",
    "xgboost",
    "joblib",
    "matplotlib",
    "openpyxl",
    "streamlit",
    "altair",
]


@pytest.mark.parametrize("package", REQUIRED_PACKAGES)
def test_required_package_importable(package):
    importlib.import_module(package)


@pytest.mark.parametrize(
    "path",
    [
        config.RAW_DATA_DIR,
        config.PROCESSED_DATA_DIR,
        config.FIGURES_DIR,
        config.MODELS_DIR,
        config.PREDICTIONS_DIR,
    ],
)
def test_project_directories_exist(path):
    assert path.is_dir(), f"Missing directory: {path}"
