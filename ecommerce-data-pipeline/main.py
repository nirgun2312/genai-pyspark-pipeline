"""Main entry point for the ecommerce-data-pipeline project."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from src.config import CONFIG
from src.data_generator import SyntheticDataGenerator

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def format_size(path: Path) -> str:
    """Return a human-readable file size for a given path."""
    size = path.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} TB"


def main() -> None:
    """Generate raw datasets and output runtime and file size metrics."""
    try:
        CONFIG.RAW_DIR.mkdir(parents=True, exist_ok=True)
        generator = SyntheticDataGenerator()

        start_time = time.perf_counter()
        customers = generator.generate_customers(CONFIG.N_CUSTOMERS)
        customers_path = generator.save_parquet(customers, "customers")

        products = generator.generate_products(CONFIG.N_PRODUCTS)
        products_path = generator.save_parquet(products, "products")

        orders = generator.generate_orders(customers, products, CONFIG.N_ORDERS)
        orders_path = generator.save_parquet(orders, "orders")
        elapsed = time.perf_counter() - start_time

        print("Data generation completed successfully.")
        print(f"Elapsed time: {elapsed:.2f} seconds")
        print(f"Customers file: {customers_path} ({format_size(customers_path)})")
        print(f"Products file: {products_path} ({format_size(products_path)})")
        print(f"Orders file: {orders_path} ({format_size(orders_path)})")
    except Exception as error:
        logger.exception("Failed to generate ecommerce data: %s", error)
        raise


if __name__ == "__main__":
    main()
