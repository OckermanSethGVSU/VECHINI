#!/usr/bin/env python3
"""Compute recall@k from exact and observed neighbor-ID NPY matrices."""

from __future__ import annotations

import argparse
import ast
import csv
import math
import re
import struct
import sys
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import BinaryIO, Iterator


NPY_MAGIC = b"\x93NUMPY"


@dataclass(frozen=True)
class NpyMatrix:
    path: Path
    rows: int
    columns: int
    data_offset: int
    row_struct: struct.Struct

    def iter_rows(self) -> Iterator[tuple[int, ...]]:
        with self.path.open("rb") as handle:
            handle.seek(self.data_offset)
            for row_index in range(self.rows):
                raw = handle.read(self.row_struct.size)
                if len(raw) != self.row_struct.size:
                    raise ValueError(
                        f"{self.path}: truncated data while reading row {row_index}"
                    )
                yield self.row_struct.unpack(raw)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer, got {value!r}"
        )
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a 2D exact ground-truth ID .npy file with a 2D query-result "
            "ID .npy file and write aggregate recall@k statistics."
        )
    )
    parser.add_argument(
        "ground_truth",
        type=Path,
        help="2D integer .npy matrix of exact neighbor IDs",
    )
    parser.add_argument(
        "query_ids",
        type=Path,
        nargs="+",
        help=(
            "One or more 2D integer .npy matrices of IDs returned by the vector "
            "database. Multiple files are concatenated in worker/client order."
        ),
    )
    parser.add_argument(
        "--top-k",
        "-k",
        dest="top_k",
        type=positive_int,
        nargs="+",
        required=True,
        metavar="K",
        help="One or more top-k values to compute recall for (one output row each)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("recall.csv"),
        help="Output CSV path (default: recall.csv)",
    )
    return parser.parse_args()


def read_exact(handle: BinaryIO, size: int, path: Path, label: str) -> bytes:
    value = handle.read(size)
    if len(value) != size:
        raise ValueError(f"{path}: truncated NPY {label}")
    return value


def parse_integer_dtype(path: Path, descriptor: str) -> tuple[str, int]:
    if len(descriptor) < 3:
        raise ValueError(f"{path}: unsupported NPY dtype descriptor {descriptor!r}")

    byte_order = descriptor[0]
    kind = descriptor[1]
    try:
        item_size = int(descriptor[2:])
    except ValueError as exc:
        raise ValueError(
            f"{path}: unsupported NPY dtype descriptor {descriptor!r}"
        ) from exc

    if kind not in {"i", "u"} or item_size not in {1, 2, 4, 8}:
        raise ValueError(
            f"{path}: expected an integer NPY dtype, got {descriptor!r}"
        )

    if item_size == 1:
        endian = ""
    elif byte_order in {"<", "="}:
        endian = "<"
    elif byte_order == ">":
        endian = ">"
    else:
        raise ValueError(
            f"{path}: unsupported NPY byte order in dtype {descriptor!r}"
        )

    format_codes = {
        ("i", 1): "b",
        ("u", 1): "B",
        ("i", 2): "h",
        ("u", 2): "H",
        ("i", 4): "i",
        ("u", 4): "I",
        ("i", 8): "q",
        ("u", 8): "Q",
    }
    return endian + format_codes[(kind, item_size)], item_size


def load_npy_matrix(path: Path) -> NpyMatrix:
    resolved = path.expanduser().resolve()
    with resolved.open("rb") as handle:
        if read_exact(handle, len(NPY_MAGIC), resolved, "magic") != NPY_MAGIC:
            raise ValueError(f"{resolved}: not an NPY file")

        major, minor = read_exact(handle, 2, resolved, "version")
        if major == 1:
            header_size = struct.unpack(
                "<H", read_exact(handle, 2, resolved, "header length")
            )[0]
            encoding = "latin1"
        elif major in {2, 3}:
            header_size = struct.unpack(
                "<I", read_exact(handle, 4, resolved, "header length")
            )[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"{resolved}: unsupported NPY version {major}.{minor}")

        raw_header = read_exact(handle, header_size, resolved, "header")
        try:
            header = ast.literal_eval(raw_header.decode(encoding).strip())
        except (SyntaxError, ValueError, UnicodeDecodeError) as exc:
            raise ValueError(f"{resolved}: invalid NPY header") from exc

        if not isinstance(header, dict):
            raise ValueError(f"{resolved}: invalid NPY header dictionary")
        if header.get("fortran_order") is not False:
            raise ValueError(f"{resolved}: Fortran-order NPY arrays are not supported")

        shape = header.get("shape")
        if (
            not isinstance(shape, tuple)
            or len(shape) != 2
            or not all(isinstance(value, int) and value >= 0 for value in shape)
        ):
            raise ValueError(f"{resolved}: expected a 2D NPY matrix, got shape {shape!r}")

        descriptor = header.get("descr")
        if not isinstance(descriptor, str):
            raise ValueError(f"{resolved}: invalid NPY dtype descriptor")
        value_format, item_size = parse_integer_dtype(resolved, descriptor)

        rows, columns = shape
        if rows == 0 or columns == 0:
            raise ValueError(f"{resolved}: matrix must have at least one row and column")

        row_struct = struct.Struct(value_format[0:1] + value_format[-1] * columns)
        expected_size = rows * columns * item_size
        data_offset = handle.tell()
        handle.seek(0, 2)
        actual_size = handle.tell() - data_offset
        if actual_size != expected_size:
            raise ValueError(
                f"{resolved}: expected {expected_size} data bytes, found {actual_size}"
            )

    return NpyMatrix(
        path=resolved,
        rows=rows,
        columns=columns,
        data_offset=data_offset,
        row_struct=row_struct,
    )


def query_result_sort_key(matrix: NpyMatrix) -> tuple[int, int, str]:
    match = re.search(r"_w(\d+)_c(\d+)\.npy$", matrix.path.name)
    if match:
        return int(match.group(1)), int(match.group(2)), matrix.path.name
    return 0, 0, matrix.path.name


def compute_recall_multi(
    ground_truth: NpyMatrix,
    query_id_matrices: list[NpyMatrix],
    top_ks: list[int],
) -> list[dict[str, int | float]]:
    """Compute recall@k for every k in top_ks in a single pass over the data."""
    if not query_id_matrices:
        raise ValueError("at least one query-ID matrix is required")
    if not top_ks:
        raise ValueError("at least one top-k value is required")

    query_id_matrices = sorted(query_id_matrices, key=query_result_sort_key)
    query_rows = sum(matrix.rows for matrix in query_id_matrices)
    if ground_truth.rows != query_rows:
        raise ValueError(
            "row-count mismatch: "
            f"ground truth has {ground_truth.rows}, query IDs have {query_rows}"
        )
    max_k = max(top_ks)
    if max_k > ground_truth.columns:
        raise ValueError(
            f"top-k {max_k} exceeds ground-truth columns {ground_truth.columns}"
        )
    for matrix in query_id_matrices:
        if max_k > matrix.columns:
            raise ValueError(
                f"top-k {max_k} exceeds query-ID columns {matrix.columns} "
                f"in {matrix.path}"
            )

    # Per-k accumulators — initialise min_recall to 1.0 (worst possible stays 0).
    accums: dict[int, list[float]] = {
        k: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]  # recall_sum, sq_sum, min, max, matches, perfect
        for k in top_ks
    }

    for row_index, (expected_row, actual_row) in enumerate(
        zip(
            ground_truth.iter_rows(),
            chain.from_iterable(matrix.iter_rows() for matrix in query_id_matrices),
        )
    ):
        # Validate against max_k once per row; smaller k values are a subset.
        max_expected = expected_row[:max_k]
        if any(point_id < 0 for point_id in max_expected):
            raise ValueError(
                f"{ground_truth.path}: negative ground-truth ID in row {row_index}"
            )
        if len(set(max_expected)) != max_k:
            raise ValueError(
                f"{ground_truth.path}: duplicate ground-truth ID in row {row_index}"
            )

        for k in top_ks:
            expected_set = set(expected_row[:k])
            actual_set = {pid for pid in actual_row[:k] if pid >= 0}
            matches = len(expected_set.intersection(actual_set))
            recall = matches / k
            acc = accums[k]
            acc[0] += recall
            acc[1] += recall * recall
            if recall < acc[2]:
                acc[2] = recall
            if recall > acc[3]:
                acc[3] = recall
            acc[4] += matches
            if matches == k:
                acc[5] += 1

    query_count = ground_truth.rows
    results = []
    for k in sorted(top_ks):
        recall_sum, sq_sum, min_r, max_r, total_matches, perfect = accums[k]
        mean_recall = recall_sum / query_count
        variance = max(0.0, sq_sum / query_count - mean_recall ** 2)
        results.append({
            "top_k": k,
            "query_count": query_count,
            "total_matches": int(total_matches),
            "total_possible_matches": query_count * k,
            "mean_recall_at_k": mean_recall,
            "min_recall_at_k": min_r,
            "max_recall_at_k": max_r,
            "stddev_recall_at_k": math.sqrt(variance),
            "perfect_query_count": int(perfect),
            "perfect_query_fraction": perfect / query_count,
        })
    return results


def write_recall_csv(path: Path, summaries: list[dict[str, int | float]]) -> None:
    output = path.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0]) if summaries else []
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    ground_truth = load_npy_matrix(args.ground_truth)
    query_id_matrices = [load_npy_matrix(path) for path in args.query_ids]
    summaries = compute_recall_multi(ground_truth, query_id_matrices, sorted(set(args.top_k)))
    write_recall_csv(args.output, summaries)
    for s in summaries:
        print(
            f"wrote {args.output.expanduser().resolve()} "
            f"(recall@{s['top_k']}={s['mean_recall_at_k']:.6f}, "
            f"queries={s['query_count']})"
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
