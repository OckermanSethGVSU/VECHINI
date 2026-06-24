#!/usr/bin/env python3
"""Run efficient local Qdrant recall sweeps from a TOML configuration."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import hashlib
import itertools
import json
import math
import os
import random
import shlex
import shutil
import struct
import subprocess
import sys
import threading
import time
import tomllib
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
QDRANT_DIR = REPO_ROOT / "qdrant"
CONFIGURE_COLLECTION = QDRANT_DIR / "scripts" / "configure_collection.py"
BUILD_INDEX = QDRANT_DIR / "scripts" / "build_index.py"
UPDATE_QUANTIZATION = QDRANT_DIR / "scripts" / "update_quantization.py"
COMPUTE_RECALL = REPO_ROOT / "utils" / "compute_recall.py"

RESULT_FIELDS = [
    "run_key",
    "status",
    "error",
    "dataset",
    "data_file",
    "query_file",
    "ground_truth_file",
    "corpus_size",
    "query_count",
    "vector_dim",
    "data_size_bytes",
    "distance_metric",
    "qdrant_image",
    "number_of_segments",
    "segment_size_kb",
    "actual_segments",
    "quantization_variant",
    "quantization",
    "quantization_always_ram",
    "quantization_scalar_quantile",
    "quantization_binary_encoding",
    "quantization_product_compression",
    "quantization_turbo_bits",
    "hnsw_m",
    "ef_construct",
    "ef_search",
    "top_k",
    "physical_top_k",
    "top_k_execution_mode",
    "insert_time_s",
    "quantization_time_s",
    "index_time_s",
    "query_time_s",
    "mean_recall_at_k",
    "min_recall_at_k",
    "max_recall_at_k",
    "stddev_recall_at_k",
    "perfect_query_count",
    "perfect_query_fraction",
    "result_dir",
]

SWEEP_PARAM_FIELDS = [
    "SWEEP_DATASET",
    "ACTIVE_TASK",
    "QDRANT_SWEEP_IMAGE",
    "COLLECTION_NAME",
    "VECTOR_DIM",
    "DISTANCE_METRIC",
    "INSERT_CORPUS_SIZE",
    "QUERY_CORPUS_SIZE",
    "DEFAULT_SEGMENT_NUMBER",
    "MAX_SEGMENT_SIZE",
    "QUANTIZATION_VARIANT",
    "QUANTIZATION_TYPE",
    "QUANTIZATION_ALWAYS_RAM",
    "QUANTIZATION_SCALAR_QUANTILE",
    "QUANTIZATION_BINARY_ENCODING",
    "QUANTIZATION_PRODUCT_COMPRESSION",
    "QUANTIZATION_TURBO_BITS",
    "HNSW_M",
    "HNSW_EF_CONSTRUCTION",
    "HNSW_EF_SEARCH",
    "TOP_K",
    "SWEEP_TOP_K_VALUES",
    "TOP_K_EXECUTION_MODE",
]

METRIC_MAP = {
    "cosine": "COSINE",
    "dot": "IP",
    "ip": "IP",
    "inner_product": "IP",
    "innerproduct": "IP",
    "euclidean": "L2",
    "euclid": "L2",
    "l2": "L2",
}


@dataclass(frozen=True)
class NpyMetadata:
    rows: int
    columns: int
    dtype: str
    item_size: int


@dataclass(frozen=True)
class Dataset:
    name: str
    directory: Path
    data: Path
    queries: Path
    ground_truth: Path
    distance_metric: str
    data_meta: NpyMetadata
    query_meta: NpyMetadata
    ground_truth_meta: NpyMetadata


@dataclass(frozen=True)
class QuantizationVariant:
    name: str
    quantization_type: str
    always_ram: bool = True
    scalar_quantile: str = ""
    binary_encoding: str = "DEFAULT"
    product_compression: str = "X16"
    turbo_bits: str = "BITS4"

    def environment(self) -> dict[str, str]:
        return {
            "QUANTIZATION_TYPE": self.quantization_type,
            "QUANTIZATION_ALWAYS_RAM": "True" if self.always_ram else "False",
            "QUANTIZATION_SCALAR_QUANTILE": self.scalar_quantile,
            "QUANTIZATION_BINARY_ENCODING": self.binary_encoding,
            "QUANTIZATION_PRODUCT_COMPRESSION": self.product_compression,
            "QUANTIZATION_TURBO_BITS": self.turbo_bits,
        }


@dataclass(frozen=True)
class RunSettings:
    output_dir: Path
    results_csv: Path
    batch_client: Path
    image: str
    container_name: str
    host: str
    http_port: int
    grpc_port: int
    p2p_port: int
    insert_batch_size: int
    query_batch_size: int
    insert_clients: int
    query_clients: int
    streaming: bool
    keep_container: bool
    resume: bool
    rpc_timeout: str
    health_timeout_seconds: int
    parallel_top_k_workers: int


@dataclass(frozen=True)
class CollectionTarget:
    dataset: Dataset
    segments: int
    quantization: QuantizationVariant


@dataclass(frozen=True)
class InsertionTarget:
    """Identifies a unique inserted collection — keyed by dataset+segments only.

    Quantization is excluded because it can be changed live without reinserting.
    """
    dataset: Dataset
    segments: int


@dataclass(frozen=True)
class QueryPlan:
    """One physical Qdrant query for one logical top_k value."""
    ef_search: int
    physical_top_k: int
    logical_top_ks: tuple[int, ...]


@dataclass
class CollectionState:
    collection_name: str
    collection_dir: Path
    build_dir: Path
    base_env: dict[str, str]
    data_size_bytes: int
    segment_size_kb: int
    insert_time: float
    current_quantization: QuantizationVariant | None = None
    current_graph: tuple[int, int] | None = None
    quantization_time: float | str = "n/a"
    index_time: float = 0.0
    actual_segments: Any = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run local Qdrant parameter sweeps while reusing inserted data and "
            "HNSW builds across query-only ef_search values."
        )
    )
    parser.add_argument("config", type=Path, help="TOML sweep configuration")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan without starting Qdrant",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Execute or display at most this many query configurations",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore successful rows already present in the results CSV",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=None,
        help=(
            "Override [run].results_csv. Successful rows in this CSV are used "
            "for resume, and new local results are appended to the same file."
        ),
    )
    return parser.parse_args()


def require_positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def require_list(table: dict[str, Any], name: str) -> list[Any]:
    value = table.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"sweep.{name} must be a non-empty TOML array")
    return value


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def read_npy_metadata(path: Path) -> NpyMetadata:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError(f"{path}: not an NPY file")
        major, minor = handle.read(2)
        if major == 1:
            header_size = struct.unpack("<H", handle.read(2))[0]
            encoding = "latin1"
        elif major in {2, 3}:
            header_size = struct.unpack("<I", handle.read(4))[0]
            encoding = "utf-8" if major == 3 else "latin1"
        else:
            raise ValueError(f"{path}: unsupported NPY version {major}.{minor}")
        header = ast.literal_eval(handle.read(header_size).decode(encoding).strip())

    shape = header.get("shape")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError(f"{path}: expected a 2D NPY matrix, got {shape!r}")
    if header.get("fortran_order") is not False:
        raise ValueError(f"{path}: Fortran-order arrays are not supported")
    dtype = str(header.get("descr"))
    try:
        item_size = int(dtype[2:])
    except ValueError as exc:
        raise ValueError(f"{path}: unsupported NPY dtype {dtype!r}") from exc
    if item_size <= 0:
        raise ValueError(f"{path}: invalid NPY item size in dtype {dtype!r}")
    return NpyMetadata(int(shape[0]), int(shape[1]), dtype, item_size)


def read_run_config_env(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.is_file():
        return env
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parts = shlex.split(line)
        except ValueError:
            return {}
        if len(parts) != 1 or "=" not in parts[0]:
            return {}
        key, value = parts[0].split("=", 1)
        env[key] = value
    return env


def reusable_query_artifacts(
    query_dir: Path,
    physical_top_k: int,
    expected_rows: int | None = None,
    expected_env: dict[str, str] | None = None,
) -> bool:
    """Return True only if query artifacts exist, are wide enough, and match this run."""
    result_ids_path = query_dir / "query_result_ids.npy"
    times_csv = query_dir / "query_times.csv"
    if not result_ids_path.is_file() or not times_csv.is_file():
        return False
    if expected_env is not None:
        existing_env = read_run_config_env(query_dir / "run_config.env")
        if not existing_env:
            return False
        for key, value in expected_env.items():
            if existing_env.get(key) != str(value):
                return False
    try:
        result_ids = np.load(result_ids_path, mmap_mode="r")
    except Exception:
        return False
    if result_ids.ndim != 2:
        return False
    if expected_rows is not None and result_ids.shape[0] != expected_rows:
        return False
    if result_ids.shape[1] < physical_top_k:
        return False
    try:
        read_rank_zero_total(times_csv)
    except Exception:
        return False
    return True


def remove_query_artifacts(query_dir: Path) -> None:
    for filename in ("query_result_ids.npy", "query_times.csv", "recall.csv"):
        (query_dir / filename).unlink(missing_ok=True)


def _file_digest(path: Path) -> bytes:
    """MD5 digest of a file, read in 1 MiB chunks. Used only for equality checks."""
    h = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.digest()


def read_distance_metric(directory: Path) -> str:
    path = directory / "distance_metric.txt"
    if not path.is_file():
        raise ValueError(f"missing dataset metric file: {path}")
    raw = path.read_text(encoding="utf-8").strip().lower()
    try:
        return METRIC_MAP[raw]
    except KeyError as exc:
        raise ValueError(f"{path}: unsupported distance metric {raw!r}") from exc


def validate_ground_truth_ids(
    dataset_name: str,
    ground_truth: Path,
    corpus_size: int,
) -> None:
    ids = np.load(ground_truth, mmap_mode="r")
    if ids.size == 0:
        raise ValueError(f"dataset {dataset_name}: ground truth is empty")
    minimum = int(ids.min())
    maximum = int(ids.max())
    if minimum < 0 or maximum >= corpus_size:
        raise ValueError(
            f"dataset {dataset_name}: ground-truth IDs [{minimum}, {maximum}] "
            f"fall outside configured corpus [0, {corpus_size - 1}]; "
            "the data and ground-truth files do not describe the same corpus"
        )


def load_datasets(config: dict[str, Any], config_dir: Path) -> list[Dataset]:
    entries = config.get("datasets")
    if not isinstance(entries, list) or not entries:
        raise ValueError("config must contain at least one [[datasets]] entry")

    datasets: list[Dataset] = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"datasets[{index}] must be a TOML table")
        if entry.get("enabled", True) is False:
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"datasets[{index}].name is required")
        if name in names:
            raise ValueError(f"duplicate dataset name: {name}")
        names.add(name)

        directory = resolve_path(str(entry.get("directory", "")), config_dir)
        if not directory.is_dir():
            raise ValueError(f"dataset directory does not exist: {directory}")

        def dataset_file(key: str) -> Path:
            value = str(entry.get(key, "")).strip()
            if not value:
                raise ValueError(f"dataset {name}: {key} is required")
            path = resolve_path(value, directory)
            if not path.is_file():
                raise ValueError(f"dataset {name}: file does not exist: {path}")
            return path

        data = dataset_file("data")
        queries = dataset_file("queries")
        ground_truth = dataset_file("ground_truth")
        data_meta = read_npy_metadata(data)
        query_meta = read_npy_metadata(queries)
        ground_truth_meta = read_npy_metadata(ground_truth)

        if data_meta.columns != query_meta.columns:
            raise ValueError(
                f"dataset {name}: data dim {data_meta.columns} != query dim {query_meta.columns}"
            )
        if query_meta.rows != ground_truth_meta.rows:
            raise ValueError(
                f"dataset {name}: query rows {query_meta.rows} != "
                f"ground-truth rows {ground_truth_meta.rows}"
            )
        validate_ground_truth_ids(name, ground_truth, data_meta.rows)
        datasets.append(
            Dataset(
                name=name,
                directory=directory,
                data=data,
                queries=queries,
                ground_truth=ground_truth,
                distance_metric=read_distance_metric(directory),
                data_meta=data_meta,
                query_meta=query_meta,
                ground_truth_meta=ground_truth_meta,
            )
        )
    if not datasets:
        raise ValueError("no enabled dataset entries")
    return datasets


def normalize_qdrant_version(version: str) -> str:
    version = version.strip()
    if not version or version == "latest" or version.startswith("v"):
        return version or "latest"
    return f"v{version}"


def locate_batch_client(run: dict[str, Any], config_dir: Path) -> Path:
    configured = str(run.get("batch_client", "")).strip()
    candidates = []
    if configured:
        candidates.append(resolve_path(configured, config_dir))
    candidates.extend(
        [
            QDRANT_DIR / "clients" / "batch_client" / "target" / "release" / "batch_client",
            QDRANT_DIR / "clients" / "batch_client" / "target" / "debug" / "batch_client",
            QDRANT_DIR / "clients" / "batch_client" / "batch_client",
        ]
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return candidates[0].resolve()


def load_settings(
    config: dict[str, Any], config_dir: Path, no_resume: bool
) -> RunSettings:
    run = config.get("run", {})
    if not isinstance(run, dict):
        raise ValueError("[run] must be a TOML table")
    output_dir = resolve_path(str(run.get("output_dir", "qdrant_sweep")), config_dir)
    results_csv_value = str(run.get("results_csv", "results.csv"))
    results_csv = resolve_path(results_csv_value, output_dir)
    image = str(run.get("qdrant_image", "")).strip()
    if not image:
        version = normalize_qdrant_version(str(run.get("qdrant_version", "latest")))
        image = f"qdrant/qdrant:{version}"
    return RunSettings(
        output_dir=output_dir,
        results_csv=results_csv,
        batch_client=locate_batch_client(run, config_dir),
        image=image,
        container_name=str(run.get("container_name", "qdrant-local-sweep")),
        host=str(run.get("host", "127.0.0.1")),
        http_port=require_positive_int(run.get("http_port", 6333), "run.http_port"),
        grpc_port=require_positive_int(run.get("grpc_port", 6334), "run.grpc_port"),
        p2p_port=require_positive_int(run.get("p2p_port", 6335), "run.p2p_port"),
        insert_batch_size=require_positive_int(
            run.get("insert_batch_size", 512), "run.insert_batch_size"
        ),
        query_batch_size=require_positive_int(
            run.get("query_batch_size", 32), "run.query_batch_size"
        ),
        insert_clients=require_positive_int(
            run.get("insert_clients", 1), "run.insert_clients"
        ),
        query_clients=require_positive_int(
            run.get("query_clients", 1), "run.query_clients"
        ),
        streaming=bool(run.get("streaming", True)),
        keep_container=bool(run.get("keep_container", False)),
        resume=bool(run.get("resume", True)) and not no_resume,
        rpc_timeout=str(run.get("rpc_timeout", "")).strip(),
        health_timeout_seconds=require_positive_int(
            run.get("health_timeout_seconds", 900),
            "run.health_timeout_seconds",
        ),
        parallel_top_k_workers=require_positive_int(
            run.get("parallel_top_k_workers", 1),
            "run.parallel_top_k_workers",
        ),
    )


def load_sweep(config: dict[str, Any]) -> dict[str, list[Any]]:
    sweep = config.get("sweep")
    if not isinstance(sweep, dict):
        raise ValueError("[sweep] must be a TOML table")
    result = {
        "top_k": sweep.get("top_k", [config.get("run", {}).get("top_k", 10)]),
        "hnsw_m": require_list(sweep, "hnsw_m"),
        "ef_construct": require_list(sweep, "ef_construct"),
        "ef_search": require_list(sweep, "ef_search"),
        "number_of_segments": require_list(sweep, "number_of_segments"),
    }
    if not isinstance(result["top_k"], list) or not result["top_k"]:
        raise ValueError("sweep.top_k must be a non-empty TOML array")
    for key in ("top_k", "hnsw_m", "ef_construct", "ef_search", "number_of_segments"):
        result[key] = [require_positive_int(value, f"sweep.{key}") for value in result[key]]
        result[key] = list(dict.fromkeys(result[key]))
    return result


def effective_query_plans(sweep: dict[str, list[Any]]) -> list[QueryPlan]:
    """Build the physical query plan for a sweep.

    Reconstructs the logical pair set with Qdrant's effective efSearch rule:
    ef_search = max(requested_ef_search, top_k).  Each logical top_k gets its
    own physical query because Qdrant's search path can depend on top_k.
    """
    seen: set[tuple[int, int]] = set()
    plans: list[QueryPlan] = []
    for requested_ef_search in sweep["ef_search"]:
        for top_k in sweep["top_k"]:
            ef_search = max(requested_ef_search, top_k)
            key = (ef_search, top_k)
            if key in seen:
                continue
            seen.add(key)
            plans.append(
                QueryPlan(
                    ef_search=ef_search,
                    physical_top_k=top_k,
                    logical_top_ks=(top_k,),
                )
            )
    return sorted(plans, key=lambda plan: (plan.ef_search, plan.physical_top_k))


def graph_setting_order(sweep: dict[str, list[Any]]) -> list[tuple[int, int]]:
    settings = list(itertools.product(sweep["hnsw_m"], sweep["ef_construct"]))
    random.shuffle(settings)
    return settings


def prioritized_collection_targets(
    datasets: list[Dataset],
    segments_values: list[int],
    quantization_variants: list[QuantizationVariant],
) -> list[CollectionTarget]:
    targets = [
        CollectionTarget(dataset, segments, variant)
        for segments in segments_values
        for variant in quantization_variants
        for dataset in datasets
    ]
    # Use a flat shuffle so neither a segment count nor the unquantized case is
    # systematically favored when the scheduler switches datasets.
    random.shuffle(targets)
    return targets


def load_quantization_variants(config: dict[str, Any]) -> list[QuantizationVariant]:
    allowed_types = {"NONE", "SCALAR", "BINARY", "PRODUCT", "TURBO"}
    allowed_binary_encodings = {"DEFAULT", "TWO_BITS", "ONE_AND_HALF_BITS"}
    allowed_product_compressions = {"X4", "X8", "X16", "X32", "X64"}
    allowed_turbo_bits = {"BITS4", "BITS2", "BITS1_5", "BITS1"}

    entries = config.get("quantization_variants")
    if entries is None:
        sweep = config.get("sweep", {})
        legacy_types = require_list(sweep, "quantization")
        table = config.get("quantization", {})
        if not isinstance(table, dict):
            raise ValueError("[quantization] must be a TOML table")
        entries = [
            {
                "name": str(quantization_type).strip().lower(),
                "type": quantization_type,
                **table,
            }
            for quantization_type in legacy_types
        ]
    if not isinstance(entries, list) or not entries:
        raise ValueError("config must contain at least one [[quantization_variants]] entry")

    variants: list[QuantizationVariant] = []
    names: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"quantization_variants[{index}] must be a TOML table")
        if entry.get("enabled", True) is False:
            continue
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"quantization_variants[{index}].name is required")
        if name in names:
            raise ValueError(f"duplicate quantization variant name: {name}")
        names.add(name)

        quantization_type = str(entry.get("type", "")).strip().upper()
        if quantization_type not in allowed_types:
            raise ValueError(
                f"quantization variant {name}: unsupported type {quantization_type!r}"
            )
        scalar_quantile = entry.get("scalar_quantile", "")
        if scalar_quantile != "":
            scalar_quantile = str(float(scalar_quantile))
            if not 0 < float(scalar_quantile) <= 1:
                raise ValueError(
                    f"quantization variant {name}: scalar_quantile must be in (0, 1]"
                )
        binary_encoding = str(entry.get("binary_encoding", "DEFAULT")).upper()
        product_compression = str(
            entry.get("product_compression", "X16")
        ).upper()
        turbo_bits = str(entry.get("turbo_bits", "BITS4")).upper()
        if binary_encoding not in allowed_binary_encodings:
            raise ValueError(
                f"quantization variant {name}: unsupported binary_encoding"
            )
        if product_compression not in allowed_product_compressions:
            raise ValueError(
                f"quantization variant {name}: unsupported product_compression"
            )
        if turbo_bits not in allowed_turbo_bits:
            raise ValueError(f"quantization variant {name}: unsupported turbo_bits")
        variants.append(
            QuantizationVariant(
                name=name,
                quantization_type=quantization_type,
                always_ram=bool(entry.get("always_ram", True)),
                scalar_quantile=str(scalar_quantile),
                binary_encoding=binary_encoding,
                product_compression=product_compression,
                turbo_bits=turbo_bits,
            )
        )
    if not variants:
        raise ValueError("no enabled quantization variants")
    return variants


def command_env(base: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(base)
    for name in (
        "NO_PROXY",
        "no_proxy",
        "http_proxy",
        "https_proxy",
        "HTTP_PROXY",
        "HTTPS_PROXY",
    ):
        env[name] = ""
    return env


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=cwd,
            env=command_env(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {process.returncode}: "
            f"{' '.join(command)}; see {log_path}"
        )


def write_run_config(directory: Path, env: dict[str, str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    contents = [
        "# Generated by qdrant/utils/run_local_recall_sweep.py",
        "# This file contains the explicit environment used for this operation.",
    ]
    contents.extend(
        f"{name}={shlex.quote(str(value))}"
        for name, value in sorted(env.items())
    )
    (directory / "run_config.env").write_text(
        "\n".join(contents) + "\n",
        encoding="utf-8",
    )
    with (directory / "sweep_params.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=SWEEP_PARAM_FIELDS)
        writer.writeheader()
        writer.writerow({field: env.get(field, "") for field in SWEEP_PARAM_FIELDS})


def detect_container_runtime() -> str:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    raise RuntimeError("Docker or Podman is required")


def container_exists(runtime: str, name: str) -> bool:
    result = subprocess.run(
        [runtime, "ps", "-a", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout.splitlines()


def container_running(runtime: str, name: str) -> bool:
    result = subprocess.run(
        [runtime, "ps", "--format", "{{.Names}}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return name in result.stdout.splitlines()


def container_image(runtime: str, name: str) -> str:
    result = subprocess.run(
        [runtime, "inspect", "--format", "{{.Config.Image}}", name],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def start_container(runtime: str, settings: RunSettings) -> bool:
    data_dir = settings.output_dir / "qdrant_storage"
    config_dir = settings.output_dir / "qdrant_config"
    snapshots_dir = settings.output_dir / "qdrant_snapshots"
    for path in (data_dir, config_dir, snapshots_dir):
        path.mkdir(parents=True, exist_ok=True)

    exists = container_exists(runtime, settings.container_name)
    use_latest = settings.image.rsplit(":", 1)[-1] == "latest"
    if exists and (
        use_latest
        or container_image(runtime, settings.container_name) != settings.image
    ):
        subprocess.run(
            [runtime, "rm", "-f", settings.container_name],
            check=True,
        )
        exists = False

    created = False
    if exists:
        if not container_running(runtime, settings.container_name):
            subprocess.run(
                [runtime, "start", settings.container_name],
                check=True,
            )
    else:
        subprocess.run(
            [
                runtime,
                "run",
                "-d",
                "--name",
                settings.container_name,
                "-p",
                f"{settings.http_port}:6333",
                "-p",
                f"{settings.grpc_port}:6334",
                "-p",
                f"{settings.p2p_port}:6335",
                "-v",
                f"{data_dir}:/qdrant/storage",
                "-v",
                f"{config_dir}:/qdrant/config/local",
                "-v",
                f"{snapshots_dir}:/qdrant/snapshots",
                "--pull=always",
                settings.image,
            ],
            check=True,
        )
        created = True

    health_url = f"http://{settings.host}:{settings.http_port}/healthz"
    for _ in range(settings.health_timeout_seconds):
        try:
            with urllib.request.urlopen(health_url, timeout=2) as response:
                if response.status == 200:
                    return created
        except OSError:
            time.sleep(1)
    raise RuntimeError(
        f"Qdrant did not become healthy at {health_url} within "
        f"{settings.health_timeout_seconds} seconds"
    )


def stop_container(runtime: str, settings: RunSettings) -> None:
    if container_running(runtime, settings.container_name):
        subprocess.run([runtime, "stop", settings.container_name], check=True)


def query_collection_info(settings: RunSettings, collection_name: str) -> dict[str, Any]:
    url = (
        f"http://{settings.host}:{settings.http_port}/collections/"
        f"{collection_name}"
    )
    with urllib.request.urlopen(url, timeout=30) as response:
        payload = json.load(response)
    return payload["result"]


def read_rank_zero_total(path: Path) -> float:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("rank") == "0":
                return float(row["total_s"])
    raise ValueError(f"{path}: rank 0 total_s not found")


def read_recall_csv(path: Path) -> dict[int, dict[str, str]]:
    """Read a multi-row recall CSV produced by compute_recall.py, keyed by top_k."""
    with path.open(newline="", encoding="utf-8") as handle:
        return {int(row["top_k"]): row for row in csv.DictReader(handle)}


def read_single_csv_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one data row")
    return rows[0]


def write_recall_csv_local(path: Path, summaries: list[dict[str, Any]]) -> None:
    """Write a recall CSV in the same format as compute_recall.py produces."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(summaries[0]) if summaries else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summaries:
            writer.writerow(row)


def compute_recall_inline(
    gt_arr: np.ndarray,
    qr_path: Path,
    top_ks: list[int],
) -> list[dict[str, Any]]:
    """Compute recall@k without spawning a subprocess.

    Uses numpy mmap for I/O and searchsorted for intersection counting.
    Numerically equivalent to compute_recall.py's compute_recall_multi.
    """
    qr = np.load(qr_path, mmap_mode="r")
    n = gt_arr.shape[0]
    max_k = max(top_ks)
    if n != qr.shape[0]:
        raise ValueError(
            f"row-count mismatch: ground truth has {n} rows, "
            f"query results have {qr.shape[0]}"
        )
    if max_k > gt_arr.shape[1]:
        raise ValueError(
            f"top_k {max_k} exceeds ground-truth columns {gt_arr.shape[1]}"
        )
    if max_k > qr.shape[1]:
        raise ValueError(
            f"top_k {max_k} exceeds result columns {qr.shape[1]}"
        )

    # Validate ground-truth slice for max_k (matches compute_recall.py:259-266).
    gt_slice = gt_arr[:, :max_k]
    neg_rows = np.where((gt_slice < 0).any(axis=1))[0]
    if neg_rows.size:
        raise ValueError(
            f"negative ground-truth ID in row {int(neg_rows[0])}"
        )
    row_unique_counts = np.apply_along_axis(lambda r: np.unique(r).size, 1, gt_slice)
    dup_rows = np.where(row_unique_counts != max_k)[0]
    if dup_rows.size:
        raise ValueError(
            f"duplicate ground-truth ID in row {int(dup_rows[0])}"
        )

    results = []
    for k in sorted(top_ks):
        # Sort each ground-truth row so we can use searchsorted for O(k log k) lookup.
        gt_sorted = np.sort(gt_arr[:, :k], axis=1)  # (N, k)
        qr_k = qr[:, :k]                            # (N, k) view

        matches_arr = np.empty(n, dtype=np.int64)
        for i in range(n):
            # De-duplicate and remove sentinel negatives, matching the set-based logic
            # in compute_recall.py: actual_set = {pid for pid in row[:k] if pid >= 0}
            qr_valid = np.unique(qr_k[i][qr_k[i] >= 0])
            pos = np.searchsorted(gt_sorted[i], qr_valid)
            pos = np.minimum(pos, k - 1)
            matches_arr[i] = int(np.sum(gt_sorted[i, pos] == qr_valid))

        recall_per_query = matches_arr / k
        mean_r = float(recall_per_query.mean())
        sq_mean = float((recall_per_query ** 2).mean())
        variance = max(0.0, sq_mean - mean_r ** 2)

        results.append({
            "top_k": k,
            "query_count": n,
            "total_matches": int(matches_arr.sum()),
            "total_possible_matches": n * k,
            "mean_recall_at_k": mean_r,
            "min_recall_at_k": float(recall_per_query.min()),
            "max_recall_at_k": float(recall_per_query.max()),
            "stddev_recall_at_k": math.sqrt(variance),
            "perfect_query_count": int((matches_arr == k).sum()),
            "perfect_query_fraction": float((matches_arr == k).mean()),
        })
    return results


def safe_name(value: Any) -> str:
    return "".join(character if str(character).isalnum() else "_" for character in str(value))


def make_run_key(
    dataset: Dataset,
    segments: int,
    quantization: QuantizationVariant,
    hnsw_m: int,
    ef_construct: int,
    ef_search: int,
    top_k: int,
) -> str:
    return "__".join(
        map(
            safe_name,
            (
                dataset.name,
                dataset.distance_metric,
                f"seg{segments}",
                quantization.name,
                quantization.quantization_type,
                f"ram{quantization.always_ram}",
                f"sq{quantization.scalar_quantile or 'default'}",
                f"be{quantization.binary_encoding}",
                f"pc{quantization.product_compression}",
                f"tb{quantization.turbo_bits}",
                f"m{hnsw_m}",
                f"efc{ef_construct}",
                f"efs{ef_search}",
                f"k{top_k}",
                "individual_top_k",
            ),
        )
    )


def successful_result_rows(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "run_key" not in reader.fieldnames:
            raise ValueError(f"{path}: results CSV is missing the run_key column")
        return {
            row["run_key"]: row
            for row in reader
            if row.get("status") == "success" and row.get("run_key")
        }


def completed_run_keys(path: Path) -> set[str]:
    return set(successful_result_rows(path))


def successful_rows_by_dataset(
    path: Path,
    datasets: Iterable[Dataset],
) -> dict[str, int]:
    counts = {dataset.name: 0 for dataset in datasets}
    if not path.is_file():
        return counts
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return counts
        if "dataset" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise ValueError(
                f"{path}: results CSV is missing the dataset or status column"
            )
        for row in reader:
            dataset = row.get("dataset", "")
            if row.get("status") == "success" and dataset in counts:
                counts[dataset] += 1
    return counts


def successful_coverage_by_dataset(
    path: Path,
    datasets: Iterable[Dataset],
) -> dict[str, dict[str, Counter[str]]]:
    coverage = {
        dataset.name: {
            "number_of_segments": Counter(),
            "quantization_variant": Counter(),
            "hnsw_m": Counter(),
            "ef_construct": Counter(),
        }
        for dataset in datasets
    }
    if not path.is_file():
        return coverage
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return coverage
        required = {
            "dataset",
            "status",
            "number_of_segments",
            "quantization_variant",
            "hnsw_m",
            "ef_construct",
        }
        missing = sorted(required - set(reader.fieldnames))
        if missing:
            raise ValueError(
                f"{path}: results CSV is missing required coverage columns: "
                f"{', '.join(missing)}"
            )
        for row in reader:
            dataset = row.get("dataset", "")
            if row.get("status") != "success" or dataset not in coverage:
                continue
            for field in coverage[dataset]:
                coverage[dataset][field][str(row.get(field, ""))] += 1
    return coverage


def merge_successful_result_rows(
    destination: Path,
    rows: dict[str, dict[str, str]],
) -> int:
    existing_rows: list[dict[str, str]] = []
    existing_success: set[str] = set()
    if destination.is_file():
        with destination.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or "run_key" not in reader.fieldnames:
                raise ValueError(
                    f"{destination}: results CSV is missing the run_key column"
                )
            for row in reader:
                existing_rows.append(row)
                if row.get("status") == "success" and row.get("run_key"):
                    existing_success.add(row["run_key"])

    imported = [row for key, row in rows.items() if key not in existing_success]
    if not imported:
        return 0
    imported_keys = {row["run_key"] for row in imported}
    existing_rows = [
        row for row in existing_rows if row.get("run_key") not in imported_keys
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in existing_rows + imported:
            writer.writerow(normalize_result_row(row))
    os.replace(temporary, destination)
    return len(imported)


def normalize_result_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized = {field: row.get(field, "") for field in RESULT_FIELDS}
    if not normalized.get("physical_top_k"):
        normalized["physical_top_k"] = normalized.get("top_k", "")
    if not normalized.get("quantization_time_s"):
        normalized["quantization_time_s"] = "n/a"
    return normalized


def ensure_result_header(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames == RESULT_FIELDS:
            return
        if reader.fieldnames is None or "run_key" not in reader.fieldnames:
            raise ValueError(f"{path}: results CSV is missing the run_key column")
        rows = [normalize_result_row(row) for row in reader]

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_result_header(path)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(normalize_result_row(row))


def base_environment(
    settings: RunSettings,
    dataset: Dataset,
    collection_name: str,
    segments: int,
    segment_size_kb: int,
    quantization: QuantizationVariant,
    sweep_top_k: list[int],
) -> dict[str, str]:
    env = {
        "RUN_MODE": "local",
        "TASK": "QUERY",
        "SWEEP_DATASET": dataset.name,
        "QDRANT_SWEEP_IMAGE": settings.image,
        "QUANTIZATION_VARIANT": quantization.name,
        "COLLECTION_NAME": collection_name,
        "VECTOR_DIM": str(dataset.data_meta.columns),
        "DISTANCE_METRIC": dataset.distance_metric,
        "DEFAULT_SEGMENT_NUMBER": str(segments),
        "MAX_SEGMENT_SIZE": str(segment_size_kb),
        "REBALANCE_TOPOLOGY": "False",
        "INSERT_DATA_FILEPATH": str(dataset.data),
        "INSERT_CORPUS_SIZE": str(dataset.data_meta.rows),
        "INSERT_BATCH_SIZE": str(settings.insert_batch_size),
        "INSERT_CLIENTS_PER_WORKER": str(settings.insert_clients),
        "INSERT_BALANCE_STRATEGY": "NO_BALANCE",
        "INSERT_STREAMING": "True" if settings.streaming else "False",
        "QUERY_DATA_FILEPATH": str(dataset.queries),
        "QUERY_CORPUS_SIZE": str(dataset.query_meta.rows),
        "QUERY_BATCH_SIZE": str(settings.query_batch_size),
        "QUERY_CLIENTS_PER_WORKER": str(settings.query_clients),
        "QUERY_BALANCE_STRATEGY": "NO_BALANCE",
        "QUERY_STREAMING": "True" if settings.streaming else "False",
        "QUERY_DEBUG_RESULTS": "True",
        "TOP_K": "",
        "SWEEP_TOP_K_VALUES": ",".join(map(str, sweep_top_k)),
        "TOP_K_EXECUTION_MODE": "INDIVIDUAL",
        "N_WORKERS": "1",
        "QDRANT_HOST": settings.host,
        "QDRANT_REST_PORT": str(settings.http_port),
        "QDRANT_GRPC_PORT": str(settings.grpc_port),
        "QDRANT_URL": f"http://{settings.host}:{settings.grpc_port}",
        "RPC_TIMEOUT": settings.rpc_timeout,
        "CONFIGURE_COLLECTION_INITIAL_SLEEP_SECONDS": "0",
        "CONFIGURE_COLLECTION_READY_SLEEP_SECONDS": "0",
    }
    env.update(quantization.environment())
    return env


def write_registry(path: Path, settings: RunSettings) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.parent / "runtime_state").mkdir(exist_ok=True)
    path.write_text(
        f"0,{settings.host},{settings.p2p_port}\n",
        encoding="utf-8",
    )


def print_plan(
    datasets: list[Dataset],
    sweep: dict[str, list[Any]],
    quantization_variants: list[QuantizationVariant],
    settings: RunSettings,
    completed: set[str],
) -> int:
    query_plans = effective_query_plans(sweep)
    dataset_success_counts = (
        successful_rows_by_dataset(settings.results_csv, datasets)
        if settings.resume
        else {dataset.name: 0 for dataset in datasets}
    )
    per_graph_physical = len(query_plans)
    per_graph_logical = sum(len(p.logical_top_ks) for p in query_plans)
    configs_per_unit = (
        len(datasets)
        * len(sweep["number_of_segments"])
        * len(quantization_variants)
        * len(sweep["hnsw_m"])
        * len(sweep["ef_construct"])
    )
    total_logical = configs_per_unit * per_graph_logical
    skipped = 0
    if settings.resume:
        for dataset, segments, quantization, hnsw_m, ef_construct, plan in itertools.product(
            datasets,
            sweep["number_of_segments"],
            quantization_variants,
            sweep["hnsw_m"],
            sweep["ef_construct"],
            query_plans,
        ):
            for logical_top_k in plan.logical_top_ks:
                if make_run_key(
                    dataset,
                    segments,
                    quantization,
                    hnsw_m,
                    ef_construct,
                    plan.ef_search,
                    logical_top_k,
                ) in completed:
                    skipped += 1
    print(f"datasets: {len(datasets)}")
    for dataset in datasets:
        data_size_bytes = (
            dataset.data_meta.rows
            * dataset.data_meta.columns
            * dataset.data_meta.item_size
        )
        segment_sizes = ", ".join(
            f"{segments}:{math.ceil(data_size_bytes / segments / 1024)}KB"
            for segments in sweep["number_of_segments"]
        )
        print(
            f"  {dataset.name}: rows={dataset.data_meta.rows}, "
            f"queries={dataset.query_meta.rows}, dim={dataset.data_meta.columns}, "
            f"metric={dataset.distance_metric}, "
            f"successful_results={dataset_success_counts[dataset.name]}, "
            f"segment_sizes={segment_sizes}"
        )
    print(f"quantization variants: {len(quantization_variants)}")
    print(f"top_k values: {', '.join(map(str, sweep['top_k']))}")
    print(f"parallel top_k workers: {settings.parallel_top_k_workers}")
    print(
        f"query plans per (m, ef_construct): {per_graph_physical} physical, "
        f"{per_graph_logical} logical rows"
    )
    for plan in query_plans:
        print(
            f"  ef_search={plan.ef_search}: run top_k={plan.physical_top_k}"
        )
    for variant in quantization_variants:
        print(
            f"  {variant.name}: type={variant.quantization_type}, "
            f"always_ram={variant.always_ram}, "
            f"scalar_quantile={variant.scalar_quantile or 'default'}, "
            f"binary_encoding={variant.binary_encoding}, "
            f"product_compression={variant.product_compression}, "
            f"turbo_bits={variant.turbo_bits}"
        )
    insertions = len(datasets) * len(sweep["number_of_segments"])
    quant_updates = insertions * (len(quantization_variants) - 1)
    print(f"insertions (dataset × segments): {insertions}")
    print(
        f"quantization updates: {quant_updates} "
        f"(one target-order pass; graph settings run inside each quantization)"
    )
    print(
        "priority order: dataset/segment/quantization targets are flat-shuffled; "
        "targets prefer the least-complete datasets and undercovered "
        "segment/quantization values within each dataset; once counts are "
        "within twenty target batches, execution stays on the current "
        "dataset/segment when possible"
    )
    print(
        "graph setting order: randomized per run across (hnsw_m, ef_construct) "
        "pairs, then biased toward undercovered values within each dataset; "
        "each graph-setting unit is still preserved"
    )
    print(
        "index builds: "
        f"{configs_per_unit}"
    )
    print(f"Qdrant query executions: {configs_per_unit * per_graph_physical}")
    print(f"recall result configurations: {total_logical}")
    print(f"already completed: {skipped}")
    print(f"remaining: {total_logical - skipped}")
    print(f"results: {settings.results_csv}")
    print(f"image: {settings.image}")
    print(f"batch client: {settings.batch_client}")
    return total_logical - skipped


def execute_sweep(
    datasets: list[Dataset],
    sweep: dict[str, list[Any]],
    quantization_variants: list[QuantizationVariant],
    settings: RunSettings,
    limit: int | None,
) -> None:
    if not settings.batch_client.is_file():
        raise RuntimeError(
            f"Qdrant batch client not found: {settings.batch_client}; build it with "
            "(cd qdrant/clients && ./build.sh batch_client)"
        )

    settings.output_dir.mkdir(parents=True, exist_ok=True)
    completed = completed_run_keys(settings.results_csv) if settings.resume else set()
    dataset_success_counts = (
        successful_rows_by_dataset(settings.results_csv, datasets)
        if settings.resume
        else {dataset.name: 0 for dataset in datasets}
    )
    dataset_coverage = (
        successful_coverage_by_dataset(settings.results_csv, datasets)
        if settings.resume
        else {
            dataset.name: {
                "number_of_segments": Counter(),
                "quantization_variant": Counter(),
                "hnsw_m": Counter(),
                "ef_construct": Counter(),
            }
            for dataset in datasets
        }
    )
    targets = prioritized_collection_targets(
        datasets,
        sweep["number_of_segments"],
        quantization_variants,
    )
    target_order = {target: index for index, target in enumerate(targets)}
    runtime = detect_container_runtime()
    start_container(runtime, settings)
    executed = 0
    query_plans = effective_query_plans(sweep)
    per_target_rows = sum(len(plan.logical_top_ks) for plan in query_plans)
    dataset_balance_window = per_target_rows * 20
    # Keyed by (dataset, segments) — quantization is changed live, not by reinsertion.
    insertion_states: dict[InsertionTarget, CollectionState] = {}
    # Ground-truth arrays loaded once per dataset and reused across all plans.
    gt_cache: dict[str, np.ndarray] = {}
    output_lock = threading.Lock()
    gt_cache_lock = threading.Lock()

    def is_complete(run_key: str) -> bool:
        with output_lock:
            return run_key in completed

    def limit_reached() -> bool:
        with output_lock:
            return limit is not None and executed >= limit

    def get_ground_truth(dataset: Dataset) -> np.ndarray:
        with gt_cache_lock:
            if dataset.name not in gt_cache:
                gt_cache[dataset.name] = np.load(
                    dataset.ground_truth, mmap_mode="r"
                )
            return gt_cache[dataset.name]

    def emit_result(row: dict[str, Any], *, mark_completed: bool = False) -> bool:
        nonlocal executed
        with output_lock:
            if limit is not None and executed >= limit:
                return False
            append_result(settings.results_csv, row)
            if mark_completed and row.get("status") == "success":
                completed.add(str(row["run_key"]))
                dataset_name = str(row.get("dataset", ""))
                if dataset_name in dataset_success_counts:
                    dataset_success_counts[dataset_name] += 1
                if dataset_name in dataset_coverage:
                    coverage = dataset_coverage[dataset_name]
                    for field in coverage:
                        coverage[field][str(row.get(field, ""))] += 1
            executed += 1
            current = executed
        if row.get("status") == "success":
            suffix = " (saturated)" if row.get("_saturated") else ""
            print(
                f"[{current}] {row['run_key']}: "
                f"recall={row['mean_recall_at_k']}{suffix}"
            )
        else:
            print(f"[{current}] {row['run_key']}: failed: {row.get('error', '')}")
        return True

    try:
        # Collection targets are outermost for locality.  Once a
        # dataset/segment/quantization target is selected, run randomized graph
        # settings inside that target before reconsidering dataset balance.
        graph_settings = graph_setting_order(sweep)
        graph_order = {
            graph: index for index, graph in enumerate(graph_settings)
        }
        pending_targets = list(targets)
        previous_target: CollectionTarget | None = None

        def target_pending_rows(target: CollectionTarget) -> int:
            return sum(
                1
                for hnsw_m, ef_construct in graph_settings
                for plan in query_plans
                for lk in plan.logical_top_ks
                if make_run_key(
                    target.dataset,
                    target.segments,
                    target.quantization,
                    hnsw_m,
                    ef_construct,
                    plan.ef_search,
                    lk,
                )
                not in completed
            )

        def target_coverage_score(target: CollectionTarget) -> tuple[int, int, int]:
            coverage = dataset_coverage[target.dataset.name]
            segment_count = coverage["number_of_segments"][str(target.segments)]
            quantization_count = coverage["quantization_variant"][
                target.quantization.name
            ]
            return (
                segment_count + quantization_count,
                segment_count,
                quantization_count,
            )

        def target_selection_key(target: CollectionTarget) -> tuple[int, int, int, int]:
            return (*target_coverage_score(target), target_order[target])

        def graph_selection_key(
            dataset: Dataset,
            hnsw_m: int,
            ef_construct: int,
        ) -> tuple[int, int, int, int]:
            coverage = dataset_coverage[dataset.name]
            m_count = coverage["hnsw_m"][str(hnsw_m)]
            ef_construct_count = coverage["ef_construct"][str(ef_construct)]
            return (
                m_count + ef_construct_count,
                m_count,
                ef_construct_count,
                graph_order[(hnsw_m, ef_construct)],
            )

        def choose_next_target() -> CollectionTarget | None:
            pending_row_counts = {
                target: pending_rows
                for target in pending_targets
                if (pending_rows := target_pending_rows(target)) > 0
            }
            candidates = [
                target
                for target in pending_targets
                if target in pending_row_counts
            ]
            if not candidates:
                return None
            min_count = min(
                dataset_success_counts.get(target.dataset.name, 0)
                for target in candidates
            )
            eligible = [
                target
                for target in candidates
                if dataset_success_counts.get(target.dataset.name, 0)
                <= min_count + dataset_balance_window
            ]
            projected_eligible = [
                target
                for target in eligible
                if dataset_success_counts.get(target.dataset.name, 0)
                + pending_row_counts[target]
                <= min_count + dataset_balance_window
                or dataset_success_counts.get(target.dataset.name, 0) == min_count
            ]
            if projected_eligible:
                eligible = projected_eligible
            if previous_target is None:
                lowest_count_targets = [
                    target
                    for target in eligible
                    if dataset_success_counts.get(target.dataset.name, 0)
                    == min_count
                ]
                return min(
                    lowest_count_targets,
                    key=target_selection_key,
                )
            same_insertion = [
                target
                for target in eligible
                if target.dataset == previous_target.dataset
                and target.segments == previous_target.segments
            ]
            if same_insertion:
                return min(same_insertion, key=target_selection_key)
            same_dataset = [
                target
                for target in eligible
                if target.dataset == previous_target.dataset
            ]
            if same_dataset:
                return min(same_dataset, key=target_selection_key)
            return min(eligible, key=target_selection_key)

        while pending_targets:
            target = choose_next_target()
            if target is None:
                break
            pending_targets.remove(target)
            previous_target = target
            dataset = target.dataset
            segments = target.segments
            quantization = target.quantization
            initial_hnsw_m, initial_ef_construct = graph_settings[0]
            if limit is not None and executed >= limit:
                return

            insertion_key = InsertionTarget(dataset, segments)
            state = insertion_states.get(insertion_key)
            if state is None:
                data_size_bytes = (
                    dataset.data_meta.rows
                    * dataset.data_meta.columns
                    * dataset.data_meta.item_size
                )
                segment_size_kb = math.ceil(data_size_bytes / segments / 1024)
                # Collection name and dir are keyed by dataset+segments only;
                # quantization lives in the index subdirectory.
                collection_name = safe_name(
                    f"sweep_{dataset.name}_{segments}"
                )[:200]
                collection_dir = (
                    settings.output_dir
                    / "runs"
                    / safe_name(dataset.name)
                    / f"segments_{segments}"
                )
                build_dir = collection_dir / "build"
                build_dir.mkdir(parents=True, exist_ok=True)
                write_registry(build_dir / "ip_registry.txt", settings)
                base_env = base_environment(
                    settings,
                    dataset,
                    collection_name,
                    segments,
                    segment_size_kb,
                    quantization,
                    sweep["top_k"],
                )
                collection_env = dict(base_env)
                collection_env.update(
                    {
                        "ACTIVE_TASK": "INSERT",
                        "HNSW_M": str(initial_hnsw_m),
                        "HNSW_EF_CONSTRUCTION": str(initial_ef_construct),
                    }
                )
                write_run_config(build_dir, collection_env)
                run_command(
                    [sys.executable, str(CONFIGURE_COLLECTION)],
                    build_dir,
                    collection_env,
                    collection_dir / "configure.log",
                )
                ready_flag = build_dir / "ready.flag"
                if ready_flag.exists():
                    ready_flag.unlink()
                run_command(
                    [str(settings.batch_client)],
                    build_dir,
                    collection_env,
                    collection_dir / "insert.log",
                )
                state = CollectionState(
                    collection_name=collection_name,
                    collection_dir=collection_dir,
                    build_dir=build_dir,
                    base_env=base_env,
                    data_size_bytes=data_size_bytes,
                    segment_size_kb=segment_size_kb,
                    insert_time=read_rank_zero_total(
                        build_dir / "insert_times.csv"
                    ),
                    current_quantization=quantization,
                )
                insertion_states[insertion_key] = state

            # Quantization can be changed live without reinsertion.
            # When the target quantization differs from what the collection
            # currently has, update it in place and wait for GREEN.
            if state.current_quantization != quantization:
                quant_dir = (
                    state.collection_dir
                    / "quant_updates"
                    / safe_name(quantization.name)
                )
                quant_dir.mkdir(parents=True, exist_ok=True)
                write_registry(quant_dir / "ip_registry.txt", settings)
                quant_env = dict(state.base_env)
                quant_env.update({"QUANTIZATION_VARIANT": quantization.name})
                quant_env.update(quantization.environment())
                write_run_config(quant_dir, quant_env)
                run_command(
                    [sys.executable, str(UPDATE_QUANTIZATION)],
                    quant_dir,
                    quant_env,
                    quant_dir / "update_quantization.log",
                )
                quantization_time_path = quant_dir / "quantization_time.txt"
                state.quantization_time = (
                    float(quantization_time_path.read_text(encoding="utf-8"))
                    if quantization_time_path.is_file()
                    else "n/a"
                )
                state.base_env.update({"QUANTIZATION_VARIANT": quantization.name})
                state.base_env.update(quantization.environment())
                state.current_quantization = quantization
                state.current_graph = None
            else:
                state.quantization_time = "n/a"

            target_graph_settings = sorted(
                graph_settings,
                key=lambda graph: graph_selection_key(
                    dataset, graph[0], graph[1]
                ),
            )
            for hnsw_m, ef_construct in target_graph_settings:
                pending_plans = [
                    plan for plan in query_plans
                    if any(
                        make_run_key(
                            dataset, segments, quantization,
                            hnsw_m, ef_construct, plan.ef_search, lk,
                        )
                        not in completed
                        for lk in plan.logical_top_ks
                    )
                ]
                if not pending_plans:
                    continue
                if limit is not None and executed >= limit:
                    return

                graph = (hnsw_m, ef_construct)
                # Index dir includes quantization name so results stay separate.
                index_dir = (
                    state.collection_dir
                    / safe_name(quantization.name)
                    / f"m_{hnsw_m}"
                    / f"ef_construct_{ef_construct}"
                )
                index_dir.mkdir(parents=True, exist_ok=True)
                write_registry(index_dir / "ip_registry.txt", settings)
                index_env = dict(state.base_env)
                index_env.update(
                    {
                        "ACTIVE_TASK": "INDEX",
                        "HNSW_M": str(hnsw_m),
                        "HNSW_EF_CONSTRUCTION": str(ef_construct),
                    }
                )
                if state.current_graph != graph:
                    write_run_config(index_dir, index_env)
                    run_command(
                        [sys.executable, str(BUILD_INDEX)],
                        index_dir,
                        index_env,
                        index_dir / "index.log",
                    )
                    state.index_time = float(
                        (index_dir / "index_time.txt").read_text(encoding="utf-8")
                    )
                    collection_info = query_collection_info(
                        settings,
                        state.collection_name,
                    )
                    state.actual_segments = collection_info.get(
                        "segments_count", ""
                    )
                    state.current_graph = graph

                def run_plan_lane(lane_plans: list[QueryPlan]) -> None:
                    # Saturation is local to one top_k lane.  Each lane keeps
                    # ef_search increasing so saturation remains order-preserving.
                    # Require three comparable identical outputs before
                    # saturating later efSearch values.
                    saturation_streaks: dict[int, tuple[int, bytes, int]] = {}
                    last_recall_by_logical: dict[int, dict[int, dict[str, Any]]] = {}
                    saturated_physical: set[int] = set()

                    for plan in lane_plans:
                        if limit_reached():
                            return

                        query_dir = (
                            index_dir
                            / f"ef_search_{plan.ef_search}"
                            / f"top_k_{plan.physical_top_k}"
                        )
                        query_dir.mkdir(parents=True, exist_ok=True)
                        write_registry(query_dir / "ip_registry.txt", settings)
                        query_env = dict(index_env)
                        query_env.update(
                            {
                                "ACTIVE_TASK": "QUERY",
                                "HNSW_EF_SEARCH": str(plan.ef_search),
                                "TOP_K": str(plan.physical_top_k),
                            }
                        )
                        result_ids_path = query_dir / "query_result_ids.npy"
                        can_reuse_query_artifacts = reusable_query_artifacts(
                            query_dir,
                            plan.physical_top_k,
                            dataset.query_meta.rows,
                            query_env,
                        )
                        write_run_config(query_dir, query_env)

                        def _base_row(logical_top_k: int) -> dict[str, Any]:
                            return {
                                "run_key": make_run_key(
                                    dataset, segments, quantization,
                                    hnsw_m, ef_construct,
                                    plan.ef_search, logical_top_k,
                                ),
                                "status": "failed",
                                "dataset": dataset.name,
                                "data_file": dataset.data,
                                "query_file": dataset.queries,
                                "ground_truth_file": dataset.ground_truth,
                                "corpus_size": dataset.data_meta.rows,
                                "query_count": dataset.query_meta.rows,
                                "vector_dim": dataset.data_meta.columns,
                                "data_size_bytes": state.data_size_bytes,
                                "distance_metric": dataset.distance_metric,
                                "qdrant_image": settings.image,
                                "number_of_segments": segments,
                                "segment_size_kb": state.segment_size_kb,
                                "actual_segments": state.actual_segments,
                                "quantization_variant": quantization.name,
                                "quantization": quantization.quantization_type,
                                "quantization_always_ram": quantization.always_ram,
                                "quantization_scalar_quantile": quantization.scalar_quantile,
                                "quantization_binary_encoding": quantization.binary_encoding,
                                "quantization_product_compression": quantization.product_compression,
                                "quantization_turbo_bits": quantization.turbo_bits,
                                "hnsw_m": hnsw_m,
                                "ef_construct": ef_construct,
                                "ef_search": plan.ef_search,
                                "top_k": logical_top_k,
                                "physical_top_k": plan.physical_top_k,
                                "top_k_execution_mode": "PARALLEL_TOP_K"
                                if parallel_top_k_enabled
                                else "INDIVIDUAL",
                                "insert_time_s": state.insert_time,
                                "quantization_time_s": state.quantization_time,
                                "index_time_s": state.index_time,
                                "result_dir": query_dir,
                            }

                        if plan.physical_top_k in saturated_physical:
                            prev_recalls = last_recall_by_logical.get(
                                plan.physical_top_k, {}
                            )
                            for logical_top_k in plan.logical_top_ks:
                                run_key = make_run_key(
                                    dataset, segments, quantization,
                                    hnsw_m, ef_construct,
                                    plan.ef_search, logical_top_k,
                                )
                                if is_complete(run_key):
                                    continue
                                row = _base_row(logical_top_k)
                                recall_for_k = prev_recalls.get(logical_top_k)
                                if recall_for_k is None:
                                    row["error"] = (
                                        "saturated but no recall metrics cached "
                                        f"for top_k={logical_top_k}"
                                    )
                                else:
                                    row.update(recall_for_k)
                                    row["status"] = "success"
                                    row["query_time_s"] = 0.0
                                    row["_saturated"] = True
                                emit_result(
                                    row,
                                    mark_completed=row["status"] == "success",
                                )
                            continue

                        ran_query = False
                        if not can_reuse_query_artifacts:
                            remove_query_artifacts(query_dir)
                            try:
                                run_command(
                                    [str(settings.batch_client)],
                                    query_dir,
                                    query_env,
                                    query_dir / "query.log",
                                )
                                ran_query = True
                            except Exception as exc:
                                for logical_top_k in plan.logical_top_ks:
                                    run_key = make_run_key(
                                        dataset, segments, quantization,
                                        hnsw_m, ef_construct,
                                        plan.ef_search, logical_top_k,
                                    )
                                    if is_complete(run_key):
                                        continue
                                    row = _base_row(logical_top_k)
                                    row["error"] = str(exc)
                                    emit_result(row)
                                continue

                        try:
                            query_time_s = read_rank_zero_total(
                                query_dir / "query_times.csv"
                            )
                        except Exception as exc:
                            for logical_top_k in plan.logical_top_ks:
                                run_key = make_run_key(
                                    dataset, segments, quantization,
                                    hnsw_m, ef_construct,
                                    plan.ef_search, logical_top_k,
                                )
                                if is_complete(run_key):
                                    continue
                                row = _base_row(logical_top_k)
                                row["error"] = f"query_times.csv unreadable: {exc}"
                                emit_result(row)
                            continue

                        combined_recall_path = query_dir / "recall.csv"
                        existing_recall: dict[int, dict[str, Any]] = {}
                        recall_is_current = (
                            combined_recall_path.is_file()
                            and combined_recall_path.stat().st_mtime
                            >= result_ids_path.stat().st_mtime
                        )
                        if recall_is_current and not ran_query:
                            try:
                                existing_recall = read_recall_csv(combined_recall_path)
                            except Exception:
                                existing_recall = {}
                        needed = [
                            lk for lk in plan.logical_top_ks
                            if lk not in existing_recall
                        ]
                        if needed:
                            try:
                                summaries = compute_recall_inline(
                                    get_ground_truth(dataset),
                                    result_ids_path,
                                    list(plan.logical_top_ks),
                                )
                                write_recall_csv_local(combined_recall_path, summaries)
                                existing_recall = {s["top_k"]: s for s in summaries}
                            except Exception as exc:
                                for logical_top_k in plan.logical_top_ks:
                                    run_key = make_run_key(
                                        dataset, segments, quantization,
                                        hnsw_m, ef_construct,
                                        plan.ef_search, logical_top_k,
                                    )
                                    if is_complete(run_key):
                                        continue
                                    row = _base_row(logical_top_k)
                                    row["error"] = str(exc)
                                    emit_result(row)
                                continue

                        recall_metric_keys = (
                            "mean_recall_at_k",
                            "min_recall_at_k",
                            "max_recall_at_k",
                            "stddev_recall_at_k",
                            "perfect_query_count",
                            "perfect_query_fraction",
                        )
                        this_plan_recalls: dict[int, dict[str, Any]] = {
                            lk: {
                                key: value
                                for key, value in existing_recall[lk].items()
                                if key in recall_metric_keys
                            }
                            for lk in plan.logical_top_ks
                            if lk in existing_recall
                            and is_complete(
                                make_run_key(
                                    dataset, segments, quantization,
                                    hnsw_m, ef_construct,
                                    plan.ef_search, lk,
                                )
                            )
                        }
                        for logical_top_k in plan.logical_top_ks:
                            run_key = make_run_key(
                                dataset, segments, quantization,
                                hnsw_m, ef_construct,
                                plan.ef_search, logical_top_k,
                            )
                            if is_complete(run_key):
                                continue
                            row = _base_row(logical_top_k)
                            try:
                                row.update(existing_recall[logical_top_k])
                                row["query_time_s"] = (
                                    query_time_s
                                    if logical_top_k == plan.physical_top_k
                                    else 0.0
                                )
                                row["status"] = "success"
                                this_plan_recalls[logical_top_k] = {
                                    key: row[key]
                                    for key in recall_metric_keys
                                    if key in row
                                }
                            except Exception as exc:
                                row["error"] = str(exc)
                            emit_result(
                                row,
                                mark_completed=row["status"] == "success",
                            )

                        if result_ids_path.is_file():
                            current_hash = _file_digest(result_ids_path)
                            streak = saturation_streaks.get(plan.physical_top_k)
                            if streak is None:
                                saturation_streaks[plan.physical_top_k] = (
                                    plan.ef_search,
                                    current_hash,
                                    1,
                                )
                            else:
                                previous_ef_search, previous_hash, streak_count = streak
                                # Do not declare saturation from nearby effective
                                # efSearch values such as 10->16, 50->64, or
                                # 100->128.  Require at least a 2x step and
                                # three identical outputs before saturation.
                                comparable_step = (
                                    plan.ef_search >= 2 * previous_ef_search
                                )
                                if comparable_step and previous_hash == current_hash:
                                    streak_count += 1
                                    if streak_count >= 3:
                                        saturated_physical.add(plan.physical_top_k)
                                else:
                                    streak_count = 1
                                saturation_streaks[plan.physical_top_k] = (
                                    plan.ef_search,
                                    current_hash,
                                    streak_count,
                                )
                        if this_plan_recalls:
                            last_recall_by_logical.setdefault(
                                plan.physical_top_k, {}
                            ).update(this_plan_recalls)

                lanes: list[list[QueryPlan]] = []
                for top_k in sorted({plan.physical_top_k for plan in pending_plans}):
                    lane = [
                        plan for plan in pending_plans
                        if plan.physical_top_k == top_k
                    ]
                    lanes.append(sorted(lane, key=lambda p: p.ef_search))

                parallel_top_k_enabled = (
                    settings.parallel_top_k_workers > 1
                    and len(lanes) > 1
                    and limit is None
                )
                if parallel_top_k_enabled:
                    max_workers = min(settings.parallel_top_k_workers, len(lanes))
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=max_workers
                    ) as executor:
                        futures = [
                            executor.submit(run_plan_lane, lane)
                            for lane in lanes
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            future.result()
                else:
                    for lane in lanes:
                        run_plan_lane(lane)
    finally:
        if not settings.keep_container:
            stop_container(runtime, settings)


def main() -> int:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)

    settings = load_settings(config, config_path.parent, args.no_resume)
    if args.results_csv is not None:
        settings = replace(
            settings,
            results_csv=args.results_csv.expanduser().resolve(),
        )
    datasets = load_datasets(config, config_path.parent)
    sweep = load_sweep(config)
    quantization_variants = load_quantization_variants(config)
    prioritized_collection_targets(
        datasets,
        sweep["number_of_segments"],
        quantization_variants,
    )
    if max(sweep["top_k"]) > min(
        dataset.ground_truth_meta.columns for dataset in datasets
    ):
        raise ValueError("sweep.top_k exceeds at least one ground-truth matrix width")
    completed = completed_run_keys(settings.results_csv) if settings.resume else set()
    print_plan(datasets, sweep, quantization_variants, settings, completed)
    if args.dry_run:
        return 0
    execute_sweep(datasets, sweep, quantization_variants, settings, args.limit)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
