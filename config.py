"""Shared paths and defaults (no hardcoded user files)."""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
JOBS_CSV = PACKAGE_ROOT / "jobs.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_N = 5
DEFAULT_THRESHOLD = 0.45
