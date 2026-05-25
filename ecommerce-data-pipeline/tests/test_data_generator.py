"""Tests for data generation."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from src.data_generator import SyntheticDataGenerator
from src.config import CONFIG


def test_generate_small(tmp_path: Path) -> None:
    gen = SyntheticDataGenerator(seed=123)
    # override raw_dir to tmp
    gen.raw_dir = tmp_path
    customers, products, orders = gen.generate_all(n_customers=1000, n_products=200, n_orders=5000)

    assert isinstance(customers, pd.DataFrame)
    assert isinstance(products, pd.DataFrame)
    assert isinstance(orders, pd.DataFrame)
    assert len(customers) == 1000
    assert len(products) == 200
    assert len(orders) == 5000

    # test parquet write
    gen.save_parquet(customers, "customers_test")
    assert (tmp_path / "customers_test.parquet").exists()

    # cleanup
    shutil.rmtree(tmp_path)
