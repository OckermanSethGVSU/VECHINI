#!/usr/bin/env python3
"""Inspect Qdrant workload input files."""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print metadata for .npy files used by Qdrant workloads."
    )
    parser.add_argument("path", type=Path, help="Path to a .npy file")
    parser.add_argument(
        "--field",
        choices=("rows", "shape", "dtype"),
        default="rows",
        help="Metadata field to print. Defaults to rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    arr = np.load(args.path.expanduser(), mmap_mode="r")

    if args.field == "rows":
        print(arr.shape[0])
    elif args.field == "shape":
        print(",".join(str(dim) for dim in arr.shape))
    else:
        print(arr.dtype)


if __name__ == "__main__":
    main()
