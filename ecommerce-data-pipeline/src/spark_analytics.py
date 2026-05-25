"""PySpark analytics for the ecommerce-data-pipeline project.

Loads raw Parquet files, computes business insights, and writes processed Parquet results.
"""
from __future__ import annotations

import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from .config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


def create_spark_session(app_name: str = CONFIG.SPARK_APP_NAME) -> SparkSession:
    """Create and return a SparkSession configured for local execution."""
    return (
        SparkSession.builder.appName(app_name)
        .master('local[*]')
        .config('spark.sql.shuffle.partitions', '8')
        .getOrCreate()
    )


def _save_table(df, path: Path) -> None:
    """Save a Spark DataFrame to Parquet with overwrite mode."""
    df.write.mode('overwrite').parquet(str(path))
    logger.info('Wrote analytics output to %s', path)


def run_analytics(spark: SparkSession, raw_dir: Path = CONFIG.RAW_DIR, processed_dir: Path = CONFIG.PROCESSED_DIR) -> None:
    """Load raw data, compute analytics, and save processed outputs."""
    logger.info('Starting analytics from %s to %s', raw_dir, processed_dir)

    customers = spark.read.parquet(str(raw_dir / 'customers.parquet'))
    products = spark.read.parquet(str(raw_dir / 'products.parquet'))
    orders = spark.read.parquet(str(raw_dir / 'orders.parquet'))

    enriched = orders.join(products, on='product_id', how='left').withColumn(
        'revenue', F.col('price') * F.col('quantity')
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    revenue_by_category = (
        enriched.groupBy('category')
        .agg(F.sum('revenue').alias('revenue'))
        .orderBy(F.desc('revenue'))
    )
    _save_table(revenue_by_category, processed_dir / 'revenue_by_category.parquet')

    customer_sales = enriched.join(customers, on='customer_id', how='left')
    top_customers = (
        customer_sales.groupBy('customer_id', 'name', 'email')
        .agg(F.sum('revenue').alias('total_revenue'))
        .orderBy(F.desc('total_revenue'))
    )
    _save_table(top_customers, processed_dir / 'top_customers.parquet')

    monthly_sales = (
        enriched.withColumn('month', F.date_format(F.col('order_date'), 'yyyy-MM'))
        .groupBy('month')
        .agg(F.sum('revenue').alias('revenue'))
        .orderBy('month')
    )
    _save_table(monthly_sales, processed_dir / 'monthly_sales.parquet')

    sales_by_country = (
        customer_sales.groupBy('country')
        .agg(F.sum('revenue').alias('revenue'))
        .orderBy(F.desc('revenue'))
    )
    _save_table(sales_by_country, processed_dir / 'sales_by_country.parquet')

    logger.info('Completed analytics and saved processed results.')


def main() -> None:
    """Run the Spark analytics pipeline from the command line."""
    spark = create_spark_session()
    try:
        run_analytics(spark)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
