"""Pytest configuration shared by every test module in tests/.

Adds src/ to sys.path so `import emotion_logic` (and other src/ modules)
work the same way app.py already makes them work via its own
sys.path.insert - tests run from the project root with plain `pytest`,
with no package/install step needed.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"

sys.path.insert(0, str(SRC_DIR))

import pytest  # noqa: E402 (must follow sys.path insert)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "requires_production_models: needs real results/*.pkl artefacts to run"
    )
