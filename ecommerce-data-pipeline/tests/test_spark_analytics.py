"""Tests for spark analytics using a small dataset."""
from __future__ import annotations

import os
import shutil
import subprocess

from pathlib import Path
import pytest
from pyspark.sql import SparkSession

from src.data_generator import SyntheticDataGenerator
from src.spark_analytics import run_analytics, create_spark_session


def _java_available() -> bool:
    """Return True if Java is available in the environment."""
    if os.environ.get("JAVA_HOME"):
        return Path(os.environ["JAVA_HOME"]).exists()
    try:
        result = subprocess.run(["java", "-version"], capture_output=True, text=True)
        return result.returncode == 0
    except FileNotFoundError:
        return False


def test_spark_analytics_small(tmp_path: Path) -> None:
    gen = SyntheticDataGenerator(seed=42)
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    gen.raw_dir = raw_dir
    customers, products, orders = gen.generate_all(n_customers=500, n_products=100, n_orders=2000)

    if not _java_available():
        pytest.skip("Java is required for PySpark analytics tests.")

    spark = create_spark_session("test-ecom")
    try:
        run_analytics(spark, raw_dir=raw_dir, processed_dir=processed_dir)
        assert (processed_dir / "revenue_by_category.parquet").exists()
        assert (processed_dir / "top_customers.parquet").exists()
        assert (processed_dir / "monthly_sales.parquet").exists()
        assert (processed_dir / "sales_by_country.parquet").exists()
    finally:
        spark.stop()
        shutil.rmtree(tmp_path)
