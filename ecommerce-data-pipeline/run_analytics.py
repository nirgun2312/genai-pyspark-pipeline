"""Run Spark analytics for the ecommerce-data-pipeline project."""
from __future__ import annotations

import time
from pathlib import Path

from src.spark_analytics import SalesAnalytics


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    raw_dir = base_dir / "data" / "raw"
    customer_path = raw_dir / "customers.parquet"
    product_path = raw_dir / "products.parquet"
    order_path = raw_dir / "orders.parquet"

    analytics = SalesAnalytics()
    spark = analytics.create_spark_session()

    try:
        start = time.perf_counter()
        customers_df = analytics.load_parquet(customer_path)
        customers_time = time.perf_counter() - start
        print(f"Loaded customers.parquet in {customers_time:.2f} seconds")

        start = time.perf_counter()
        products_df = analytics.load_parquet(product_path)
        products_time = time.perf_counter() - start
        print(f"Loaded products.parquet in {products_time:.2f} seconds")

        start = time.perf_counter()
        orders_df = analytics.load_parquet(order_path)
        orders_time = time.perf_counter() - start
        print(f"Loaded orders.parquet in {orders_time:.2f} seconds")

        start = time.perf_counter()
        top_customers = analytics.top_customers_by_revenue(orders_df, products_df)
        top_customers_time = time.perf_counter() - start
        print(f"Computed top customers in {top_customers_time:.2f} seconds")
        top_customers.show(truncate=False)

        start = time.perf_counter()
        category_sales = analytics.sales_by_category(orders_df, products_df)
        category_sales_time = time.perf_counter() - start
        print(f"Computed sales by category in {category_sales_time:.2f} seconds")
        category_sales.show(truncate=False)

        start = time.perf_counter()
        monthly_trends = analytics.monthly_trends(orders_df, products_df)
        monthly_trends_time = time.perf_counter() - start
        print(f"Computed monthly trends in {monthly_trends_time:.2f} seconds")
        monthly_trends.show(truncate=False)

    finally:
        analytics.stop_spark()


if __name__ == "__main__":
    main()
