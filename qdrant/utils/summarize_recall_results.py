#!/usr/bin/env python3
"""Summarize Qdrant recall sweep result coverage by dataset and parameter."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


DEFAULT_RESULTS = (
    Path(__file__).resolve().parents[1]
    / "local_sweep_results_from_pbs"
    / "results.csv"
)

PARAMETER_FIELDS = [
    "number_of_segments",
    "quantization_variant",
    "hnsw_m",
    "ef_construct",
    "ef_search",
    "top_k",
]

SPREAD_REPORT_FIELDS = [
    ("number_of_segments", "segments", "value"),
    ("quantization_variant", "quantization", "count"),
    ("hnsw_m", "hnsw_m", "value"),
    ("ef_construct", "ef_construct", "value"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Print row counts per dataset and the distinct values present for "
            "each sweep parameter."
        )
    )
    parser.add_argument(
        "results_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS,
        help=f"Results CSV to summarize (default: {DEFAULT_RESULTS})",
    )
    parser.add_argument(
        "--include-status",
        action="store_true",
        help="Include non-success rows. By default only status=success rows are counted.",
    )
    parser.add_argument(
        "--show-counts",
        action="store_true",
        help="Show per-value counts for each parameter, not just distinct values.",
    )
    parser.add_argument(
        "--spread-report",
        action="store_true",
        help=(
            "Print a compact Dataset Spread / Parameter Spread report and exit. "
            "By default only status=success rows are counted."
        ),
    )
    parser.add_argument(
        "--spread-sort-count",
        action="store_true",
        help=(
            "With --spread-report or --spread-per-dataset, sort every section by "
            "count descending instead of using semantic/numeric parameter ordering."
        ),
    )
    parser.add_argument(
        "--spread-per-dataset",
        action="store_true",
        help=(
            "Print a per-dataset Parameter Spread report and exit. "
            "Shows the same parameter distributions as --spread-report but "
            "repeated for each dataset individually."
        ),
    )
    return parser.parse_args()


def sort_key(value: str) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except ValueError:
        return (1, value)


def quantization_sort_key(value: str) -> tuple[int, tuple[int, int | str]]:
    if value == "none":
        return (0, sort_key(value))
    if value.startswith("scalar"):
        return (1, sort_key(value))
    if value.startswith("binary"):
        return (2, sort_key(value))
    if value.startswith("product"):
        return (3, sort_key(value))
    if value.startswith("turbo"):
        return (4, sort_key(value))
    return (5, sort_key(value))


def format_values(values: set[str]) -> str:
    return ", ".join(sorted(values, key=sort_key))


def format_counts(counter: Counter[str]) -> str:
    return ", ".join(
        f"{value}:{counter[value]}"
        for value in sorted(counter, key=sort_key)
    )


def dataset_sort_key(item: tuple[str, int]) -> tuple[int, str]:
    dataset, count = item
    return (-count, dataset)


def read_rows(path: Path, include_status: bool) -> list[dict[str, str]]:
    with path.expanduser().open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        missing = ["dataset", *PARAMETER_FIELDS]
        missing = [field for field in missing if field not in reader.fieldnames]
        if missing:
            raise ValueError(f"{path}: missing required columns: {', '.join(missing)}")
        rows = list(reader)
    if include_status:
        return rows
    return [row for row in rows if row.get("status") == "success"]


def print_aligned_counts(counter: Counter[str], *, key=None) -> None:
    items = counter.items()
    if key is None:
        key = lambda item: (-item[1], sort_key(item[0]))
    sorted_items = sorted(items, key=key)
    if not sorted_items:
        return
    label_width = max(len(str(label)) for label, _ in sorted_items)
    count_width = max(len(str(count)) for _, count in sorted_items)
    for label, count in sorted_items:
        print(f"  {label:<{label_width}}  {count:>{count_width}}")


def print_spread_report(rows: list[dict[str, str]], *, sort_count: bool) -> None:
    dataset_counts = Counter(row["dataset"] for row in rows)
    print("Dataset Spread")
    print()
    if sort_count:
        print_aligned_counts(dataset_counts)
    else:
        print_aligned_counts(dataset_counts, key=dataset_sort_key)
    print()
    print("Parameter Spread")
    print()
    for field, label, order in SPREAD_REPORT_FIELDS:
        counter = Counter(row[field] for row in rows)
        print(f"{label}:")
        if sort_count:
            print_aligned_counts(counter)
        elif order == "value":
            print_aligned_counts(counter, key=lambda item: sort_key(item[0]))
        elif field == "quantization_variant":
            print_aligned_counts(counter, key=lambda item: quantization_sort_key(item[0]))
        else:
            print_aligned_counts(counter)
        print()


def print_spread_per_dataset(rows: list[dict[str, str]], *, sort_count: bool) -> None:
    dataset_counts = Counter(row["dataset"] for row in rows)
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    for dataset, _ in sorted(dataset_counts.items(), key=dataset_sort_key):
        dataset_rows = by_dataset[dataset]
        print(f"Dataset: {dataset}  (n={len(dataset_rows)})")
        print()
        for field, label, order in SPREAD_REPORT_FIELDS:
            counter = Counter(row[field] for row in dataset_rows)
            print(f"  {label}:")
            if sort_count:
                print_aligned_counts(counter)
            elif order == "value":
                print_aligned_counts(counter, key=lambda item: sort_key(item[0]))
            elif field == "quantization_variant":
                print_aligned_counts(counter, key=lambda item: quantization_sort_key(item[0]))
            else:
                print_aligned_counts(counter)
            print()
        print()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.results_csv, args.include_status)

    if args.spread_per_dataset:
        print_spread_per_dataset(rows, sort_count=args.spread_sort_count)
        return 0

    if args.spread_report:
        print_spread_report(rows, sort_count=args.spread_sort_count)
        return 0

    print(f"results_csv: {args.results_csv.expanduser().resolve()}")
    print(f"rows_counted: {len(rows)}")
    print()

    dataset_counts = Counter(row["dataset"] for row in rows)
    print("rows_per_dataset")
    for dataset, count in sorted(dataset_counts.items(), key=dataset_sort_key):
        print(f"  {dataset}: {count}")
    print()

    print("overall_spread")
    for field in PARAMETER_FIELDS:
        values = {row[field] for row in rows}
        print(f"  {field}: {len(values)} values [{format_values(values)}]")
    print()

    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    print("per_dataset_spread")
    for dataset, _ in sorted(dataset_counts.items(), key=dataset_sort_key):
        dataset_rows = by_dataset[dataset]
        print(f"  {dataset}: rows={len(dataset_rows)}")
        for field in PARAMETER_FIELDS:
            counter = Counter(row[field] for row in dataset_rows)
            values = set(counter)
            if args.show_counts:
                detail = format_counts(counter)
            else:
                detail = format_values(values)
            print(f"    {field}: {len(values)} values [{detail}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
