#!/usr/bin/env python3
"""Report quantization coverage from a Qdrant recall-sweep results CSV."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import fmean


EXPECTED_TYPES = {"NONE", "SCALAR", "BINARY", "PRODUCT", "TURBO"}
DEFAULT_RESULTS_CSV = (
    Path(__file__).resolve().parents[1]
    / "local_sweep_results_from_pbs"
    / "results.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Show quantization coverage and compare mean recall by dataset."
        )
    )
    parser.add_argument(
        "results_csv",
        nargs="?",
        type=Path,
        default=DEFAULT_RESULTS_CSV,
        help=f"results CSV to inspect (default: {DEFAULT_RESULTS_CSV})",
    )
    return parser.parse_args()


def print_recall_comparison(successful_rows: list[dict[str, str]]) -> None:
    graph_fields = (
        "dataset",
        "number_of_segments",
        "hnsw_m",
        "ef_construct",
        "top_k",
    )
    recalls: dict[tuple[str, ...], dict[int, dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    for row in successful_rows:
        variant = row["quantization_variant"].strip()
        try:
            graph = tuple(row[field].strip() for field in graph_fields)
            ef_search = int(row["ef_search"])
            recall = float(row["mean_recall_at_k"])
        except ValueError:
            continue

        if all(graph) and variant:
            recalls[graph][ef_search][variant].append(recall)

    datasets = sorted({graph[0] for graph in recalls})
    print("\nMatched recall comparison:")
    print("Each dataset uses one fixed graph and up to three ef_search values.")

    for dataset in datasets:
        candidates = []
        for graph, ef_groups in recalls.items():
            if graph[0] != dataset:
                continue
            max_variants = max(
                (len(variants) for variants in ef_groups.values()),
                default=0,
            )
            comparable_efs = sum(
                len(variants) >= 2 for variants in ef_groups.values()
            )
            if max_variants >= 2:
                candidates.append((max_variants, comparable_efs, graph, ef_groups))

        if not candidates:
            print(f"\n{dataset}: no exact matched quantization comparison")
            continue

        _, _, graph, ef_groups = max(
            candidates,
            key=lambda candidate: (
                candidate[0],
                candidate[1],
                candidate[2][4] == "10",
                tuple(int(value) for value in candidate[2][1:4]),
            ),
        )
        common_variants = set.intersection(
            *(
                set(variants)
                for variants in ef_groups.values()
                if len(variants) >= 2
            )
        )
        if len(common_variants) < 2:
            best_ef = max(ef_groups, key=lambda ef: len(ef_groups[ef]))
            common_variants = set(ef_groups[best_ef])

        variants = sorted(
            common_variants,
            key=lambda variant: (variant != "none", variant),
        )[:3]
        matching_efs = sorted(
            ef
            for ef, values in ef_groups.items()
            if all(variant in values for variant in variants)
        )
        selected_efs = select_representative_values(matching_efs, limit=3)

        _, segments, hnsw_m, ef_construct, top_k = graph
        print(
            f"\n{dataset}: segments={segments}, m={hnsw_m}, "
            f"ef_construct={ef_construct}, recall@{top_k}"
        )
        for ef_search in selected_efs:
            print(f"  ef_search={ef_search}")
            for variant in variants:
                values = ef_groups[ef_search][variant]
                print(f"    {variant:<20} {fmean(values):.6f}")


def select_representative_values(values: list[int], limit: int) -> list[int]:
    if len(values) <= limit:
        return values
    indexes = (0, len(values) // 2, len(values) - 1)
    return [values[index] for index in indexes]


def main() -> int:
    results_csv = parse_args().results_csv.expanduser().resolve()

    try:
        csv_file = results_csv.open(newline="", encoding="utf-8")
    except OSError as error:
        raise SystemExit(f"Could not open {results_csv}: {error}") from error

    with csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {
            "dataset",
            "ef_construct",
            "ef_search",
            "hnsw_m",
            "mean_recall_at_k",
            "number_of_segments",
            "quantization",
            "quantization_variant",
            "status",
            "top_k",
        }
        missing_columns = required_columns.difference(reader.fieldnames or ())
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise SystemExit(f"{results_csv} is missing columns: {missing}")

        successful_rows = [
            row
            for row in reader
            if row["status"].strip().lower() == "success"
        ]

    type_counts = Counter(
        row["quantization"].strip().upper()
        for row in successful_rows
        if row["quantization"].strip()
    )
    variant_counts = Counter(
        row["quantization_variant"].strip()
        for row in successful_rows
        if row["quantization_variant"].strip()
    )

    covered_types = set(type_counts)
    print(f"Results file: {results_csv}")
    print(f"Successful rows: {len(successful_rows)}")
    print(f"Covered types: {', '.join(sorted(covered_types)) or 'none'}")
    print(
        "Missing expected types: "
        f"{', '.join(sorted(EXPECTED_TYPES - covered_types)) or 'none'}"
    )

    print("\nSuccessful rows by type:")
    for quantization_type, count in sorted(type_counts.items()):
        print(f"  {quantization_type}: {count}")

    print("\nSuccessful rows by variant:")
    for variant, count in sorted(variant_counts.items()):
        print(f"  {variant}: {count}")

    print_recall_comparison(successful_rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
