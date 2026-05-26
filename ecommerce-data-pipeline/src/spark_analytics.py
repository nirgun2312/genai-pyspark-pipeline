"""PySpark analytics for the ecommerce-data-pipeline project.

This module defines SalesAnalytics for loading data and computing revenue
analytics using local Spark and optimized serialization.
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class SalesAnalytics:
    """Sales analytics tools for PySpark revenue analysis."""

    def __init__(self, app_name: str = "SalesAnalytics", master: str = "local[*]", memory: str = "4g") -> None:
        """Initialize the analytics helper with Spark settings."""
        self.app_name = app_name
        self.master = master
        self.memory = memory
        self.spark: Optional[SparkSession] = None

    def _verify_java(self) -> None:
        """Ensure Java is installed and available before creating a SparkSession."""
        if shutil.which("java") is None:
            raise RuntimeError(
                "Java executable not found. Install a JDK, set JAVA_HOME, or add java to PATH. "
                "Then restart your terminal and rerun the script."
            )

    def create_spark_session(self) -> SparkSession:
        """Create a SparkSession configured for local execution and Kryo serialization."""
        self._verify_java()
        logger.info("Creating SparkSession: app=%s master=%s memory=%s", self.app_name, self.master, self.memory)
        self.spark = (
            SparkSession.builder.appName(self.app_name)
            .master(self.master)
            .config("spark.driver.memory", self.memory)
            .config("spark.executor.memory", self.memory)
            .config("spark.sql.adaptive.enabled", "true")
            .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
            .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
            .config("spark.kryoserializer.buffer.max", "512m")
            .getOrCreate()
        )
        return self.spark

    def stop_spark(self) -> None:
        """Stop the SparkSession if it is running."""
        if self.spark is not None:
            logger.info("Stopping SparkSession")
            self.spark.stop()
            self.spark = None

    def load_parquet(self, path: str | Path) -> DataFrame:
        """Load a Parquet dataset from the specified path."""
        if self.spark is None:
            self.create_spark_session()

        logger.info("Loading Parquet file from %s", path)
        return self.spark.read.parquet(str(path))

    def top_customers_by_revenue(self, orders_df: DataFrame, products_df: DataFrame, n: int = 10) -> DataFrame:
        """Return the top N customers ranked by total revenue."""
        logger.info("Calculating top %d customers by revenue", n)
        enriched = orders_df.join(products_df, on="product_id", how="left")
        revenue_df = enriched.withColumn("revenue", F.col("price") * F.col("quantity"))
        result = (
            revenue_df.groupBy("customer_id")
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.sum("quantity").alias("units_purchased"),
            )
            .orderBy(F.desc("total_revenue"))
            .limit(n)
        )
        return result

    def sales_by_category(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Aggregate total revenue and units sold by product category."""
        logger.info("Calculating sales by category")
        enriched = orders_df.join(products_df, on="product_id", how="left")
        result = (
            enriched.withColumn("revenue", F.col("price") * F.col("quantity"))
            .groupBy("category")
            .agg(
                F.sum("revenue").alias("total_revenue"),
                F.sum("quantity").alias("units_sold"),
            )
            .orderBy(F.desc("total_revenue"))
        )
        return result

    def monthly_trends(self, orders_df: DataFrame, products_df: DataFrame) -> DataFrame:
        """Calculate month-over-month revenue growth percentage."""
        logger.info("Calculating monthly revenue trends")
        enriched = orders_df.join(products_df, on="product_id", how="left")
        revenue_by_month = (
            enriched.withColumn("month", F.date_format(F.col("order_date"), "yyyy-MM"))
            .withColumn("revenue", F.col("price") * F.col("quantity"))
            .groupBy("month")
            .agg(F.sum("revenue").alias("month_revenue"))
            .orderBy("month")
        )

        month_window = Window.orderBy("month")
        result = (
            revenue_by_month
            .withColumn("previous_month_revenue", F.lag("month_revenue").over(month_window))
            .withColumn(
                "revenue_growth_pct",
                F.when(F.col("previous_month_revenue").isNull(), None).otherwise(
                    (F.col("month_revenue") - F.col("previous_month_revenue"))
                    / F.col("previous_month_revenue")
                    * 100.0
                ),
            )
        )
        return result


def main() -> None:
    """Demonstrate Spark session creation and class initialization."""
    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()
    try:
        logger.info("Spark version: %s", spark.version)
    finally:
        analytics.stop_spark()


if __name__ == "__main__":
    main()
