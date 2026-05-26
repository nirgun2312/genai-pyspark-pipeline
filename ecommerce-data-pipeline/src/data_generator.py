"""Synthetic data generator for the ecommerce-data-pipeline project.

Generates customers, products, and orders as pandas DataFrames and writes raw Parquet
files to the configured data/raw directory.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm

from .config import CONFIG

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')


class SyntheticDataGenerator:
    """Generate synthetic e-commerce datasets and save them as Parquet files."""

    def __init__(self, seed: int | None = CONFIG.RANDOM_SEED) -> None:
        """Initialize the data generator with an optional reproducibility seed."""
        self.faker = Faker()
        if seed is not None:
            np.random.seed(seed)
            Faker.seed(seed)
        self.raw_dir: Path = CONFIG.RAW_DIR
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def generate_customers(self, n_customers: int = CONFIG.N_CUSTOMERS) -> pd.DataFrame:
        """Generate the customers dataset as a pandas DataFrame."""
        logger.info('Generating %d customers', n_customers)

        ids = np.arange(1, n_customers + 1, dtype=np.int32)
        names: list[str] = []
        emails: list[str] = []
        cities: list[str] = []
        countries: list[str] = []
        registration_dates: list[pd.Timestamp] = []

        for _ in tqdm(range(n_customers), desc='customers'):
            profile = self.faker.simple_profile()
            names.append(profile['name'])  # type: ignore[index]
            emails.append(profile['mail'])  # type: ignore[index]
            cities.append(self.faker.city())
            countries.append(self.faker.country())
            registration_dates.append(self.faker.date_between(start_date='-5y', end_date='today'))

        ages = np.random.normal(loc=35, scale=10, size=n_customers).round().astype(np.int32)
        ages = np.clip(ages, 18, 90)

        return pd.DataFrame(
            {
                'customer_id': ids,
                'name': names,
                'email': emails,
                'age': ages,
                'city': cities,
                'country': countries,
                'registration_date': pd.to_datetime(registration_dates),
            }
        )

    def generate_products(self, n_products: int = CONFIG.N_PRODUCTS) -> pd.DataFrame:
        """Generate the products dataset as a pandas DataFrame."""
        logger.info('Generating %d products', n_products)

        categories = ['Electronics', 'Clothing', 'Home', 'Sports', 'Books']
        ids = np.arange(1, n_products + 1, dtype=np.int32)
        names: list[str] = []
        categories_out: list[str] = []
        prices: list[float] = []
        stocks: list[int] = []
        ratings: list[float] = []

        for _ in tqdm(range(n_products), desc='products'):
            category = self.faker.random_element(elements=categories)
            names.append(f'{self.faker.word().capitalize()} {category[:-1]}')
            categories_out.append(category)
            prices.append(round(float(np.random.uniform(10, 500)), 2))
            stocks.append(int(np.random.poisson(lam=50)))
            ratings.append(round(float(np.clip(np.random.normal(4.0, 0.8), 1.0, 5.0)), 1))

        return pd.DataFrame(
            {
                'product_id': ids,
                'name': names,
                'category': categories_out,
                'price': prices,
                'stock': stocks,
                'rating': ratings,
            }
        )

    def _pareto_order_counts(self, n_customers: int, total_orders: int, top_share: float = 0.8) -> np.ndarray:
        """Create customer order counts using a Pareto-style distribution."""
        weights = np.random.pareto(1.5, size=n_customers) + 1.0
        weights = np.clip(weights, 0.001, None)

        sorted_idx = np.argsort(weights)[::-1]
        top_count = max(1, int(n_customers * 0.2))
        top_idx = sorted_idx[:top_count]
        rest_idx = sorted_idx[top_count:]

        top_sum = weights[top_idx].sum()
        rest_sum = weights[rest_idx].sum() if rest_idx.size else 0.0

        if rest_sum > 0:
            weights[top_idx] *= 4.0 * rest_sum / max(top_sum, 1e-9)

        probabilities = weights / weights.sum()
        counts = np.floor(probabilities * total_orders).astype(np.int32)

        diff = int(total_orders - counts.sum())
        if diff > 0:
            choices = np.argsort(probabilities)[::-1][:diff]
            counts[choices] += 1
        elif diff < 0:
            choices = np.argsort(probabilities)[: -diff]
            counts[choices] -= 1

        return np.maximum(counts, 0)

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        n_orders: int = CONFIG.N_ORDERS,
    ) -> pd.DataFrame:
        """Generate orders dataset with a Pareto distribution for customer order volume."""
        logger.info('Generating %d orders', n_orders)

        customer_counts = self._pareto_order_counts(len(customers), n_orders)
        customer_ids = np.repeat(customers['customer_id'].to_numpy(dtype=np.int32), customer_counts)
        if customer_ids.size != n_orders:
            if customer_ids.size > n_orders:
                customer_ids = customer_ids[:n_orders]
            else:
                extra = n_orders - customer_ids.size
                choices = np.random.choice(customers['customer_id'].to_numpy(dtype=np.int32), size=extra)
                customer_ids = np.concatenate([customer_ids, choices])
        np.random.shuffle(customer_ids)

        product_ids = np.random.randint(1, len(products) + 1, size=n_orders, dtype=np.int32)
        quantities = np.random.randint(1, 11, size=n_orders, dtype=np.int16)

        registration_lookup = customers.set_index('customer_id')['registration_date'].to_dict()
        reg_array = pd.to_datetime(
            np.array([registration_lookup[int(cid)] for cid in customer_ids])
        ).astype('datetime64[ns]')

        now = np.datetime64('now', 'ns')
        available_days = np.maximum((now - reg_array).astype('timedelta64[D]').astype(int), 0)
        rand_days = np.array([np.random.randint(0, int(days) + 1) for days in available_days], dtype=np.int32)
        rand_seconds = np.random.randint(0, 24 * 3600, size=n_orders, dtype=np.int32)

        order_dates = reg_array + rand_days.astype('timedelta64[D]') + rand_seconds.astype('timedelta64[s]')

        return pd.DataFrame(
            {
                'order_id': np.arange(1, n_orders + 1, dtype=np.int64),
                'customer_id': customer_ids,
                'product_id': product_ids,
                'quantity': quantities,
                'order_date': order_dates,
            }
        )

    def save_parquet(self, df: pd.DataFrame, name: str) -> Path:
        """Save a DataFrame to Parquet in the configured raw directory."""
        path = self.raw_dir / f'{name}.parquet'
        df.to_parquet(path, engine='pyarrow', index=False)
        logger.info('Saved %s rows to %s', len(df), path)
        return path

    def generate_all(
        self,
        n_customers: int = CONFIG.N_CUSTOMERS,
        n_products: int = CONFIG.N_PRODUCTS,
        n_orders: int = CONFIG.N_ORDERS,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all datasets and persist them as Parquet files."""
        customers = self.generate_customers(n_customers)
        products = self.generate_products(n_products)
        orders = self.generate_orders(customers, products, n_orders)

        self.save_parquet(customers, 'customers')
        self.save_parquet(products, 'products')
        self.save_parquet(orders, 'orders')

        return customers, products, orders


def main() -> None:
    """Run full dataset generation from the command line."""
    generator = SyntheticDataGenerator()
    generator.generate_all()


if __name__ == '__main__':
    main()
