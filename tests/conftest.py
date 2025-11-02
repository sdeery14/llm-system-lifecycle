"""Pytest configuration for the test suite."""

import sys
from pathlib import Path

# Add src to path for all tests
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
