"""Load the plugin as a private package; unit tests need no running Comfy server."""
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
package = types.ModuleType("flashvsr_plus_plugin")
package.__path__ = [str(ROOT)]
sys.modules.setdefault(package.__name__, package)


def pytest_addoption(parser):
    parser.addoption("--run-cuda", action="store_true", default=False)


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: exercises real official CUDA math without pretrained weights")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--run-cuda"):
        for item in items:
            if "cuda" in item.keywords:
                item.add_marker(pytest.mark.skip(reason="Pass --run-cuda to run CUDA integration checks"))
