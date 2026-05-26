"""Benchmark multiple file formats for analytics and data engineering workloads.

This script compares CSV, XLSX, Parquet, ORC, and Feather using file size,
write/read performance, memory usage, CPU time, and estimated energy.
"""
from __future__ import annotations

import logging
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.orc as orc
from faker import Faker
from tabulate import tabulate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

T = TypeVar("T")
CATEGORIES = ["Electronics", "Clothing", "Home", "Sports", "Books"]
OUTPUT_DIR = Path("benchmark_output")
METRICS_CSV = OUTPUT_DIR / "benchmark_results.csv"
CPU_TDP_WATTS = 65.0
RANDOM_SEED = 42
ROW_COUNT = 500_000


@dataclass(frozen=True)
class BenchmarkResult:
    """Holds benchmark metrics for a single file format."""

    format_name: str
    file_path: Path
    file_size_mb: float
    write_time_s: float
    read_time_s: float
    peak_memory_mb: float
    cpu_time_s: float
    energy_wh: float
    storage_saving_pct: float = 0.0
    write_improvement_pct: float = 0.0
    read_improvement_pct: float = 0.0
    energy_saving_pct: float = 0.0


def create_output_directory(output_dir: Path) -> None:
    """Create the benchmark output directory if it does not exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.debug("Created benchmark output directory: %s", output_dir)


def generate_dataset(rows: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Generate a reproducible synthetic dataset for benchmarking."""
    logger.info("Generating synthetic dataset with %d rows", rows)
    faker = Faker()
    Faker.seed(seed)
    faker.seed_instance(seed)
    np.random.seed(seed)

    ids = np.arange(1, rows + 1, dtype=np.int64)
    names: list[str] = []
    emails: list[str] = []

    for _ in range(rows):
        names.append(faker.name())
        emails.append(faker.email())

    categories = np.random.choice(CATEGORIES, size=rows)
    amounts = np.round(np.random.uniform(10, 5000, size=rows), 2)

    start_timestamp = int(pd.Timestamp("2018-01-01").timestamp())
    end_timestamp = int(pd.Timestamp.now().timestamp())
    dates = pd.to_datetime(np.random.randint(start_timestamp, end_timestamp + 1, size=rows), unit="s")

    dataset = pd.DataFrame(
        {
            "id": ids,
            "name": names,
            "email": emails,
            "amount": amounts,
            "date": dates,
            "category": categories,
        }
    )
    logger.info("Dataset generation complete")
    return dataset


def measure_operation(operation: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[T, float, float, float]:
    """Measure time, CPU, and peak memory for a callable operation."""
    tracemalloc.start()
    tracemalloc.reset_peak()
    start_cpu = time.process_time()
    start_time = time.perf_counter()
    result = operation(*args, **kwargs)
    elapsed_time = time.perf_counter() - start_time
    cpu_time = time.process_time() - start_cpu
    peak_memory = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    return result, elapsed_time, cpu_time, peak_memory


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a CSV file."""
    df.to_csv(path, index=False)


def read_csv(path: Path) -> pd.DataFrame:
    """Read a DataFrame from a CSV file."""
    return pd.read_csv(path)


def write_xlsx(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to an XLSX file."""
    df.to_excel(path, index=False, engine="openpyxl")


def read_xlsx(path: Path) -> pd.DataFrame:
    """Read a DataFrame from an XLSX file."""
    return pd.read_excel(path, engine="openpyxl")


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a Parquet file using pyarrow."""
    df.to_parquet(path, engine="pyarrow", index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a DataFrame from a Parquet file using pyarrow."""
    return pd.read_parquet(path, engine="pyarrow")


def write_orc(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to an ORC file using pyarrow."""
    table = pa.Table.from_pandas(df, preserve_index=False)
    orc.write_table(table, str(path))


def read_orc(path: Path) -> pd.DataFrame:
    """Read a DataFrame from an ORC file using pyarrow."""
    with orc.ORCFile(str(path)) as source:
        return source.read().to_pandas()


def write_feather(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a Feather file."""
    df.reset_index(drop=True).to_feather(path)


def read_feather(path: Path) -> pd.DataFrame:
    """Read a DataFrame from a Feather file."""
    return pd.read_feather(path)


def calculate_energy(cpu_time_s: float) -> float:
    """Estimate energy consumption in Watt-hours from CPU time."""
    return (cpu_time_s * CPU_TDP_WATTS) / 3600.0


def benchmark_format(
    format_name: str,
    path: Path,
    write_fn: Callable[[pd.DataFrame, Path], None],
    read_fn: Callable[[Path], pd.DataFrame],
    df: pd.DataFrame,
) -> BenchmarkResult:
    """Benchmark one file format by writing and reading a dataset."""
    logger.info("Benchmarking format: %s", format_name)
    _, write_time, write_cpu, write_memory = measure_operation(write_fn, df, path)
    file_size_mb = path.stat().st_size / 1024**2

    _, read_time, read_cpu, read_memory = measure_operation(read_fn, path)
    peak_memory_mb = max(write_memory, read_memory) / 1024**2
    cpu_time_s = write_cpu + read_cpu
    energy_wh = calculate_energy(cpu_time_s)

    return BenchmarkResult(
        format_name=format_name,
        file_path=path,
        file_size_mb=file_size_mb,
        write_time_s=write_time,
        read_time_s=read_time,
        peak_memory_mb=peak_memory_mb,
        cpu_time_s=cpu_time_s,
        energy_wh=energy_wh,
    )


def compare_against_csv(results: list[BenchmarkResult]) -> list[BenchmarkResult]:
    """Compute percentage savings versus CSV for all benchmark results."""
    baseline = next(result for result in results if result.format_name == "CSV")

    updated_results: list[BenchmarkResult] = []
    for result in results:
        storage_saving = ((baseline.file_size_mb - result.file_size_mb) / baseline.file_size_mb) * 100.0
        write_improvement = ((baseline.write_time_s - result.write_time_s) / baseline.write_time_s) * 100.0 if baseline.write_time_s else 0.0
        read_improvement = ((baseline.read_time_s - result.read_time_s) / baseline.read_time_s) * 100.0 if baseline.read_time_s else 0.0
        energy_saving = ((baseline.energy_wh - result.energy_wh) / baseline.energy_wh) * 100.0 if baseline.energy_wh else 0.0

        updated_results.append(
            BenchmarkResult(
                format_name=result.format_name,
                file_path=result.file_path,
                file_size_mb=result.file_size_mb,
                write_time_s=result.write_time_s,
                read_time_s=result.read_time_s,
                peak_memory_mb=result.peak_memory_mb,
                cpu_time_s=result.cpu_time_s,
                energy_wh=result.energy_wh,
                storage_saving_pct=storage_saving,
                write_improvement_pct=write_improvement,
                read_improvement_pct=read_improvement,
                energy_saving_pct=energy_saving,
            )
        )

    return updated_results


def build_results_table(results: list[BenchmarkResult]) -> str:
    """Build a formatted benchmark comparison table."""
    headers = [
        "Format",
        "Size (MB)",
        "Write (s)",
        "Read (s)",
        "Peak Mem (MB)",
        "CPU (s)",
        "Energy (Wh)",
        "Storage Savings (%)",
        "Write Improvement (%)",
        "Read Improvement (%)",
        "Energy Savings (%)",
    ]
    rows = [
        [
            result.format_name,
            f"{result.file_size_mb:.2f}",
            f"{result.write_time_s:.2f}",
            f"{result.read_time_s:.2f}",
            f"{result.peak_memory_mb:.2f}",
            f"{result.cpu_time_s:.2f}",
            f"{result.energy_wh:.4f}",
            f"{result.storage_saving_pct:.2f}",
            f"{result.write_improvement_pct:.2f}",
            f"{result.read_improvement_pct:.2f}",
            f"{result.energy_saving_pct:.2f}",
        ]
        for result in results
    ]
    return tabulate(rows, headers=headers, tablefmt="github")


def save_results(results: list[BenchmarkResult], output_path: Path) -> None:
    """Persist benchmark results to a CSV file."""
    data = [
        {
            "format": result.format_name,
            "file_path": str(result.file_path),
            "file_size_mb": result.file_size_mb,
            "write_time_s": result.write_time_s,
            "read_time_s": result.read_time_s,
            "peak_memory_mb": result.peak_memory_mb,
            "cpu_time_s": result.cpu_time_s,
            "energy_wh": result.energy_wh,
            "storage_saving_pct": result.storage_saving_pct,
            "write_improvement_pct": result.write_improvement_pct,
            "read_improvement_pct": result.read_improvement_pct,
            "energy_saving_pct": result.energy_saving_pct,
        }
        for result in results
    ]
    pd.DataFrame(data).to_csv(output_path, index=False)
    logger.info("Saved benchmark results to %s", output_path)


def main() -> None:
    """Run all file format benchmarks and print a summary table."""
    try:
        create_output_directory(OUTPUT_DIR)
        dataframe = generate_dataset(ROW_COUNT, seed=RANDOM_SEED)

        benchmark_definitions = [
            ("CSV", OUTPUT_DIR / "benchmark.csv", write_csv, read_csv),
            ("XLSX", OUTPUT_DIR / "benchmark.xlsx", write_xlsx, read_xlsx),
            ("Parquet", OUTPUT_DIR / "benchmark.parquet", write_parquet, read_parquet),
            ("ORC", OUTPUT_DIR / "benchmark.orc", write_orc, read_orc),
            ("Feather", OUTPUT_DIR / "benchmark.feather", write_feather, read_feather),
        ]

        results: list[BenchmarkResult] = []
        for format_name, path, write_fn, read_fn in benchmark_definitions:
            results.append(benchmark_format(format_name, path, write_fn, read_fn, dataframe))

        results = compare_against_csv(results)
        print("\nBenchmark summary:\n")
        print(build_results_table(results))
        save_results(results, METRICS_CSV)
    except Exception as error:
        logger.exception("Benchmark failed: %s", error)
        raise


if __name__ == "__main__":
    main()
