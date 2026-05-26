"""Compare Pandas vs PySpark performance on revenue aggregation."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pandas as pd
from tabulate import tabulate

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError:
    SparkSession = None  # type: ignore
    F = None  # type: ignore


def format_seconds(value: float) -> str:
    return f"{value:.3f} s"


def pandas_workflow(orders_path: Path, products_path: Path, max_rows: int = 1_000_000) -> dict[str, Any]:
    start = time.perf_counter()
    orders_df = pd.read_parquet(orders_path, engine="pyarrow")
    if len(orders_df) > max_rows:
        orders_df = orders_df.head(max_rows)
    products_df = pd.read_parquet(products_path, engine="pyarrow")
    load_time = time.perf_counter() - start

    start = time.perf_counter()
    merged = orders_df.merge(products_df, on="product_id", how="left")
    merged["revenue"] = merged["quantity"] * merged["price"]
    top_customers = (
        merged.groupby("customer_id", as_index=False)["revenue"]
        .sum()
        .rename(columns={"revenue": "total_revenue"})
        .sort_values("total_revenue", ascending=False)
        .head(10)
    )
    agg_time = time.perf_counter() - start

    return {
        "engine": "pandas",
        "load_time_s": load_time,
        "compute_time_s": agg_time,
        "total_time_s": load_time + agg_time,
        "result": top_customers,
    }


def pyspark_workflow(orders_path: Path, products_path: Path, max_rows: int = 1_000_000) -> dict[str, Any]:
    if SparkSession is None:
        raise RuntimeError("PySpark is not installed in this environment.")

    spark = (
        SparkSession.builder.appName("PandasVsPySpark")
        .master("local[*]")
        .config("spark.driver.memory", "4g")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )

    try:
        start = time.perf_counter()
        orders_df = spark.read.parquet(str(orders_path)).limit(max_rows)
        products_df = spark.read.parquet(str(products_path))
        load_time = time.perf_counter() - start

        start = time.perf_counter()
        joined = orders_df.join(products_df, on="product_id", how="left")
        revenue_df = joined.withColumn("revenue", F.col("quantity") * F.col("price"))
        top_customers = (
            revenue_df.groupBy("customer_id")
            .agg(F.sum("revenue").alias("total_revenue"))
            .orderBy(F.desc("total_revenue"))
            .limit(10)
        )
        result = top_customers.toPandas()
        compute_time = time.perf_counter() - start

        return {
            "engine": "pyspark",
            "load_time_s": load_time,
            "compute_time_s": compute_time,
            "total_time_s": load_time + compute_time,
            "result": result,
        }
    finally:
        spark.stop()


def build_comparison_table(results: list[dict[str, Any]]) -> str:
    headers = ["Engine", "Load time", "Compute time", "Total time"]
    rows = [
        [
            result["engine"],
            format_seconds(result["load_time_s"]),
            format_seconds(result["compute_time_s"]),
            format_seconds(result["total_time_s"]),
        ]
        for result in results
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    orders_path = raw_dir / "orders.parquet"
    products_path = raw_dir / "products.parquet"

    if not orders_path.exists() or not products_path.exists():
        raise FileNotFoundError(
            f"Required parquet files not found in {raw_dir}."
        )

    print("Running pandas workflow...")
    pandas_result = pandas_workflow(orders_path, products_path)
    print(pandas_result["result"].to_string(index=False))
    print()

    print("Running PySpark workflow...")
    pyspark_result = pyspark_workflow(orders_path, products_path)
    print(pyspark_result["result"].to_string(index=False))
    print()

    print("Performance comparison:")
    print(build_comparison_table([pandas_result, pyspark_result]))


if __name__ == "__main__":
    main()
