# ecommerce-data-pipeline

A scalable e-commerce data engineering pipeline for generating synthetic customer, product, and order datasets, storing raw Parquet files, and analyzing data with PySpark.

## Project Overview

This project generates:
- 100,000 synthetic customers
- 10,000 synthetic products
- 1,000,000 synthetic orders

It stores raw datasets in `data/raw/`, performs analytics using PySpark, and writes processed results to `data/processed/`.

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Note: PySpark requires a Java JDK installation. Install Java 17+ and set `JAVA_HOME` to the JDK root before running analytics.

## Execution

Generate raw data:

```powershell
python main.py
```

Run PySpark analytics:

```powershell
python -m src.spark_analytics
```

## Analytics Explanation

The analytics module computes:
- Revenue by product category
- Top customers by revenue
- Monthly sales trends
- Revenue by country

Processed Parquet output files are written to `data/processed/`.

## Project Structure

```text
ecommerce-data-pipeline/
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_generator.py
│   └── spark_analytics.py
└── tests/
    ├── test_data_generator.py
    └── test_spark_analytics.py
```

## Sample Outputs

After generation, you will find:
- `data/raw/customers.parquet`
- `data/raw/products.parquet`
- `data/raw/orders.parquet`

After analytics, you will find:
- `data/processed/revenue_by_category.parquet`
- `data/processed/top_customers.parquet`
- `data/processed/monthly_sales.parquet`
- `data/processed/sales_by_country.parquet`
