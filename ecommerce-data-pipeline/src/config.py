"""Configuration and constants for the pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Config:
    """Project configuration and defaults."""

    ROOT: Path = Path(__file__).resolve().parents[1]
    DATA_DIR: Path = ROOT / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # Defaults
    N_CUSTOMERS: int = 100_000
    N_PRODUCTS: int = 10_000
    N_ORDERS: int = 1_000_000

    SPARK_APP_NAME: str = "ecommerce-data-pipeline"
    RANDOM_SEED: int = 42


CONFIG = Config()
