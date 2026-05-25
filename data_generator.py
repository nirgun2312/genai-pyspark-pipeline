"""Synthetic e-commerce data generator.

Provides `SyntheticDataGenerator` which produces customers, products, and orders
as pandas DataFrames. Uses Faker for realistic names/emails, NumPy for
distributions (including a Pareto-based weighting for orders), tqdm for
progress bars, and logging for progress reporting.

Defaults generate 100_000 customers, 10_000 products, and 1_000_000 orders.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from faker import Faker
from tqdm.auto import tqdm


class SyntheticDataGenerator:
    """Generate synthetic e-commerce datasets.

    Attributes:
        faker: Faker instance used for generating names, emails and locations.
    """

    def __init__(self, seed: int | None = 42) -> None:
        """Initialize the generator.

        Args:
            seed: Optional random seed for reproducibility.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.faker = Faker()
        if seed is not None:
            np.random.seed(seed)
            Faker.seed(seed)

    def generate_customers(self, n_customers: int = 100_000) -> pd.DataFrame:
        """Generate customers DataFrame.

        Columns: customer_id, name, email, age, city, country, registration_date

        Args:
            n_customers: Number of customers to generate.

        Returns:
            pd.DataFrame with customer records.
        """
        self.logger.info("Generating %d customers", n_customers)
        ids = np.arange(1, n_customers + 1, dtype=np.int32)

        names = []
        emails = []
        cities = []
        countries = []
        registration_dates = []

        for _ in tqdm(range(n_customers), desc="customers"):
            profile = self.faker.simple_profile()
            names.append(profile["name"])  # type: ignore[index]
            emails.append(profile["mail"])  # type: ignore[index]
            city = self.faker.city()
            country = self.faker.country()
            cities.append(city)
            countries.append(country)
            # registration date in last 5 years
            registration_dates.append(self.faker.date_between(start_date="-5y", end_date="today"))

        # Age: normal around 35, clip to 18..90 and convert to int
        ages = np.random.normal(loc=35, scale=10, size=n_customers).round().astype(int)
        ages = np.clip(ages, 18, 90)

        customers = pd.DataFrame(
            {
                "customer_id": ids,
                "name": names,
                "email": emails,
                "age": ages,
                "city": cities,
                "country": countries,
                "registration_date": pd.to_datetime(registration_dates),
            }
        )

        self.logger.info("Generated customers: %d rows", len(customers))
        return customers

    def generate_products(self, n_products: int = 10_000) -> pd.DataFrame:
        """Generate products DataFrame.

        Columns: product_id, name, category, price, stock, rating

        Args:
            n_products: Number of products to generate.

        Returns:
            pd.DataFrame with product records.
        """
        self.logger.info("Generating %d products", n_products)
        categories = ["Electronics", "Clothing", "Home", "Sports", "Books"]

        product_ids = np.arange(1, n_products + 1, dtype=np.int32)
        names = []
        cats = []
        prices = []
        stocks = []
        ratings = []

        for _ in tqdm(range(n_products), desc="products"):
            cat = np.random.choice(categories)
            # product name can be a short phrase
            name = f"{self.faker.word().capitalize()} {cat[:-1]}"
            names.append(name)
            cats.append(cat)
            prices.append(round(float(np.random.uniform(10, 500)), 2))
            # stock: 0-1000 skewed with poisson
            stocks.append(int(np.random.poisson(lam=50)))
            # rating: 1.0 - 5.0 with one decimal
            ratings.append(round(float(np.clip(np.random.normal(4.0, 0.8), 1.0, 5.0)), 1))

        products = pd.DataFrame(
            {
                "product_id": product_ids,
                "name": names,
                "category": cats,
                "price": prices,
                "stock": stocks,
                "rating": ratings,
            }
        )

        self.logger.info("Generated products: %d rows", len(products))
        return products

    def _pareto_order_weights(self, n_customers: int, total_orders: int, top_share: float = 0.8) -> np.ndarray:
        """Create order counts per customer using Pareto weights.

        Ensures that `top_share` of orders come from the top 20% customers.
        """
        # base pareto weights
        weights = np.random.pareto(a=1.5, size=n_customers) + 1e-6

        # enforce exact 80/20 style split: top 20% customers make `top_share` of orders
        cutoff = int(max(1, n_customers * 0.2))
        sorted_idx = np.argsort(weights)[::-1]
        top_idx = sorted_idx[:cutoff]
        rest_idx = sorted_idx[cutoff:]

        S_top = weights[top_idx].sum()
        S_rest = weights[rest_idx].sum()
        if S_top <= 0 or S_rest <= 0:
            # fallback: uniform distribution
            weights = np.ones(n_customers, dtype=float)
        else:
            # multiplier to scale top weights so their share becomes `top_share`
            m = (top_share * S_rest) / ((1 - top_share) * S_top)
            weights[top_idx] = weights[top_idx] * m

        # normalize to total_orders
        probs = weights / weights.sum()
        counts = np.floor(probs * total_orders).astype(int)

        # adjust rounding error to match total_orders
        diff = total_orders - counts.sum()
        if diff > 0:
            # add remaining to customers with largest fractional parts
            fractional = probs * total_orders - np.floor(probs * total_orders)
            order = np.argsort(fractional)[::-1]
            counts[order[:diff]] += 1
        elif diff < 0:
            # remove extras from smallest fractional parts
            fractional = probs * total_orders - np.floor(probs * total_orders)
            order = np.argsort(fractional)
            remove = -diff
            counts[order[:remove]] -= 1

        # ensure non-negative
        counts[counts < 0] = 0
        return counts

    def generate_orders(
        self,
        customers: pd.DataFrame,
        products: pd.DataFrame,
        n_orders: int = 1_000_000,
    ) -> pd.DataFrame:
        """Generate orders DataFrame.

        Columns: order_id, customer_id, product_id, quantity, order_date

        Args:
            customers: DataFrame produced by `generate_customers`.
            products: DataFrame produced by `generate_products`.
            n_orders: Total number of orders to generate.

        Returns:
            pd.DataFrame with orders.
        """
        self.logger.info("Generating %d orders", n_orders)
        n_customers = len(customers)
        n_products = len(products)

        # compute orders per customer using Pareto-derived weights
        counts = self._pareto_order_weights(n_customers, n_orders)

        # build customer_id array for orders
        customer_ids = np.repeat(customers["customer_id"].values, counts)
        if len(customer_ids) != n_orders:
            # safety: if mismatch due to numeric issues, adjust
            if len(customer_ids) > n_orders:
                customer_ids = customer_ids[:n_orders]
            else:
                extra = n_orders - len(customer_ids)
                choices = np.random.choice(customers["customer_id"].values, size=extra)
                customer_ids = np.concatenate([customer_ids, choices])

        # randomize order of orders so they're not grouped by customer
        order_perm = np.random.permutation(n_orders)
        customer_ids = customer_ids[order_perm]

        # product ids and quantities
        product_ids = np.random.randint(1, n_products + 1, size=n_orders, dtype=np.int32)
        quantities = np.random.randint(1, 11, size=n_orders, dtype=np.int16)

        # order dates: uniform between customer's registration_date and today
        now = pd.Timestamp.now()
        reg_dates = customers.set_index("customer_id")["registration_date"].to_dict()

        # vectorized compute: get registration_date for each order
        reg_array = np.array([reg_dates[int(cid)] for cid in tqdm(customer_ids, desc="mapping reg dates")])
        # days from registration to now
        days_range = (now - pd.to_datetime(reg_array)).dt.days.clip(lower=0).values
        # sample offset days and seconds
        rand_days = np.floor(np.random.uniform(0, 1, size=n_orders) * (days_range + 1)).astype(int)
        rand_seconds = np.random.randint(0, 24 * 3600, size=n_orders)

        order_dates = pd.to_datetime(reg_array) + pd.to_timedelta(rand_days, unit="D") + pd.to_timedelta(rand_seconds, unit="s")

        orders = pd.DataFrame(
            {
                "order_id": np.arange(1, n_orders + 1, dtype=np.int64),
                "customer_id": customer_ids,
                "product_id": product_ids,
                "quantity": quantities,
                "order_date": order_dates,
            }
        )

        # ensure order_date is sorted (optional) — keep shuffled to mimic real data
        self.logger.info("Generated orders: %d rows", len(orders))
        return orders

    def generate_all(
        self,
        n_customers: int = 100_000,
        n_products: int = 10_000,
        n_orders: int = 1_000_000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate customers, products, and orders DataFrames.

        Returns:
            Tuple of (customers_df, products_df, orders_df)
        """
        customers = self.generate_customers(n_customers)
        products = self.generate_products(n_products)
        orders = self.generate_orders(customers, products, n_orders)
        return customers, products, orders


if __name__ == "__main__":
    # quick smoke test with small sizes
    gen = SyntheticDataGenerator(seed=123)
    c, p = gen.generate_customers(1000), gen.generate_products(500)
    o = gen.generate_orders(c, p, 5000)
    print(c.head())
    print(p.head())
    print(o.head())
