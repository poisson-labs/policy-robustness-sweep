"""Scaffold sanity: the component packages exist and import cleanly (empty for now —
gate-fenced code arrives after G1-G5 per spec §4)."""

import importlib

import pytest

COMPONENT_PACKAGES = [
    "configs",
    "train",
    "sweep",
    "probe",
    "reduce",
    "render",
    "app",
    "analytics",
    "ci",
]


@pytest.mark.parametrize("package", COMPONENT_PACKAGES)
def test_component_package_imports(package: str) -> None:
    importlib.import_module(package)
