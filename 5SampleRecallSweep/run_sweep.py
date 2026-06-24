#!/home/seth-ockerman/Documents/basicEnv/bin/python3
"""
5-cluster remote Qdrant recall sweep.

Vectors are pre-inserted; this script patches HNSW config and quantization
(combined into one update_collection call per triple), waits for the index to
rebuild, queries all 5 clusters in parallel, and computes recall against the
per-sample UUID-based ground truth.

Sweep parameters (from sweep.md):
  M:           16, 32, 64, 128
  efConstruct: 128, 256, 512, 1024
  efSearch:    1000, 2000, 3000, 4000, 5000
  quantization: turbo_bits4, turbo_bits2, turbo_bits1_5, turbo_bits1,
                binary_1bit, binary_1_5_bit, binary_2bit
  top_k:       1000

Rebuild minimisation: each (M, efConstruct, quantization) triple triggers
exactly ONE update_collection call.  All 5 clusters are updated and queried
in parallel via ThreadPoolExecutor.

Usage:
  python3 run_sweep.py [--env ENV] [--output-dir DIR]
                       [--collection NAME] [--batch-size N]
                       [--no-resume] [--dry-run]
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import itertools
import math
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import QueryRequest, SearchParams, QuantizationSearchParams

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SWEEP_DIR = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = SWEEP_DIR / "env"
DEFAULT_OUTPUT_DIR = SWEEP_DIR / "results"
QUERY_EMBEDDINGS_FILE = SWEEP_DIR / "fineweb_query_embeddings_5000.embedding.npy"
GROUND_TRUTH_DIR = SWEEP_DIR / "groundTruth"

# ---------------------------------------------------------------------------
# Sweep constants (from sweep.md)
# ---------------------------------------------------------------------------
TOP_K = 1000
HNSW_M_VALUES = [16, 32, 64, 128]
EF_CONSTRUCT_VALUES = [128, 256, 512, 1024]
EF_SEARCH_VALUES = [1000, 2000, 3000, 4000, 5000]
OVERSAMPLING_VALUES = [1.0, 2.0, 3.0, 4.0, 5.0]
N_QUERIES = 1000


def unique_query_combos(
    ef_search_values: list[int],
    oversampling_values: list[float],
    top_k: int,
) -> list[tuple[int, float]]:
    """
    Return the (efSearch, oversampling) pairs that produce distinct results.

    Qdrant's effective HNSW exploration ef = max(efSearch, top_k * oversampling).
    For a given oversampling, all efSearch values below the floor (top_k * oversampling)
    produce identical results — only the smallest one is kept as a representative.
    """
    result = []
    for os in oversampling_values:
        floor = top_k * os
        seen_below = False
        for efs in sorted(ef_search_values):
            if efs < floor:
                if not seen_below:
                    result.append((efs, os))
                    seen_below = True
            else:
                result.append((efs, os))
    return result


QUERY_COMBOS: list[tuple[int, float]] = unique_query_combos(EF_SEARCH_VALUES, OVERSAMPLING_VALUES, TOP_K)

# ---------------------------------------------------------------------------
# Result CSV schema
# ---------------------------------------------------------------------------
RESULT_FIELDS = [
    "run_key",
    "status",
    "error",
    "cluster_idx",
    "sample",
    "cluster_url",
    "collection_name",
    "query_count",
    "vector_dim",
    "distance_metric",
    "quantization_variant",
    "quantization_type",
    "quantization_always_ram",
    "quantization_turbo_bits",
    "quantization_binary_encoding",
    "hnsw_m",
    "ef_construct",
    "ef_search",
    "oversampling",
    "top_k",
    "update_time_s",
    "query_time_s",
    "mean_recall_at_k",
    "min_recall_at_k",
    "max_recall_at_k",
    "stddev_recall_at_k",
    "perfect_query_count",
    "perfect_query_fraction",
]


# ---------------------------------------------------------------------------
# Quantization variants
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class QuantizationVariant:
    name: str
    quantization_type: str   # "TURBO" | "BINARY"
    always_ram: bool
    turbo_bits: str          # "BITS4" | "BITS2" | "BITS1_5" | "BITS1" (TURBO only)
    binary_encoding: str     # "DEFAULT" | "ONE_AND_HALF_BITS" | "TWO_BITS" (BINARY only)

    def build_config(self) -> models.QuantizationConfig:
        if self.quantization_type == "TURBO":
            return models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=getattr(models.TurboQuantBitSize, self.turbo_bits),
                    always_ram=self.always_ram,
                )
            )
        if self.quantization_type == "BINARY":
            encoding = (
                None
                if self.binary_encoding == "DEFAULT"
                else getattr(models.BinaryQuantizationEncoding, self.binary_encoding)
            )
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    encoding=encoding,
                    always_ram=self.always_ram,
                )
            )
        raise ValueError(f"Unknown quantization type: {self.quantization_type!r}")


QUANTIZATION_VARIANTS: list[QuantizationVariant] = [
    # ordered most quantized (cheapest) → least quantized (most expensive)
    QuantizationVariant("turbo_bits1",    "TURBO",  True, "BITS1",  "DEFAULT"),
    QuantizationVariant("binary_1bit",    "BINARY", True, "BITS4",  "DEFAULT"),
    QuantizationVariant("binary_1_5_bit", "BINARY", True, "BITS4",  "ONE_AND_HALF_BITS"),
    QuantizationVariant("turbo_bits1_5",  "TURBO",  True, "BITS1_5","DEFAULT"),
    QuantizationVariant("binary_2bit",    "BINARY", True, "BITS4",  "TWO_BITS"),
    QuantizationVariant("turbo_bits2",    "TURBO",  True, "BITS2",  "DEFAULT"),
    QuantizationVariant("turbo_bits4",    "TURBO",  True, "BITS4",  "DEFAULT"),
]


# ---------------------------------------------------------------------------
# Cluster config
# ---------------------------------------------------------------------------
@dataclass
class ClusterConfig:
    idx: int       # 1–5
    url: str       # e.g. https://....cloud.qdrant.io:6334
    api_key: str
    sample: str    # "sample1" – "sample5"

    @property
    def ground_truth_path(self) -> Path:
        return GROUND_TRUTH_DIR / f"{self.sample}.hit_ids.npy"


# ---------------------------------------------------------------------------
# Env / cluster loading
# ---------------------------------------------------------------------------
def load_env_file(path: Path) -> dict[str, str]:
    """Parse `export KEY=VALUE` lines from a shell env file."""
    result: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:]
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    return result


def load_clusters(env_path: Path) -> list[ClusterConfig]:
    env = load_env_file(env_path)
    clusters = []
    for i in range(1, 6):
        url = env.get(f"QDRANT_URL_{i}", "")
        key = env.get(f"QDRANT_API_KEY_{i}", "")
        if not url:
            raise ValueError(f"Missing QDRANT_URL_{i} in {env_path}")
        if not key:
            raise ValueError(f"Missing QDRANT_API_KEY_{i} in {env_path}")
        clusters.append(ClusterConfig(idx=i, url=url, api_key=key, sample=f"sample{i}"))
    return clusters


# ---------------------------------------------------------------------------
# Qdrant client helpers
# ---------------------------------------------------------------------------
def build_client(cluster: ClusterConfig, timeout: int = 7200) -> QdrantClient:
    """Connect to a remote Qdrant cloud cluster over gRPC+TLS."""
    parsed = urlparse(cluster.url)
    host = parsed.hostname
    grpc_port = parsed.port or 6334
    rest_port = grpc_port - 1   # 6333

    return QdrantClient(
        host=host,
        port=rest_port,
        grpc_port=grpc_port,
        api_key=cluster.api_key,
        prefer_grpc=True,
        https=True,
        timeout=timeout,
        grpc_options={"grpc.enable_http_proxy": 0},
    )


def discover_collection(client: QdrantClient, hint: str | None = None) -> str:
    """Return the collection name, optionally verifying `hint` exists."""
    collections = client.get_collections().collections
    if not collections:
        raise RuntimeError("No collections found on cluster")
    if hint:
        if hint not in {c.name for c in collections}:
            names = [c.name for c in collections]
            raise RuntimeError(f"Collection {hint!r} not found; available: {names}")
        return hint
    if len(collections) > 1:
        names = [c.name for c in collections]
        raise RuntimeError(
            f"Multiple collections found: {names}; use --collection to specify one"
        )
    return collections[0].name


def get_distance_metric(client: QdrantClient, collection_name: str) -> str:
    """Extract the distance metric string from collection info."""
    try:
        info = client.get_collection(collection_name)
        cfg = info.config
        params = getattr(cfg, "params", None)
        if params is None:
            return "unknown"
        vectors = getattr(params, "vectors", None)
        if vectors is None:
            return "unknown"
        dist = getattr(vectors, "distance", None)
        if dist is None:
            return "unknown"
        return str(dist).split(".")[-1].upper()
    except Exception:
        return "unknown"


def optimizer_status_ok(value) -> bool:
    if isinstance(value, dict):
        return value.get("ok") is not None or value.get("status") == "ok"
    return str(value).split(".")[-1].strip().lower() == "ok"


def collection_is_ready(info: models.CollectionInfo) -> bool:
    return (
        info.status == models.CollectionStatus.GREEN
        and optimizer_status_ok(getattr(info, "optimizer_status", None))
    )


def wait_for_ready(
    client: QdrantClient,
    collection_name: str,
    timeout: int = 7200,
    poll_interval: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if collection_is_ready(client.get_collection(collection_name)):
                return
        except Exception:
            pass
        time.sleep(poll_interval)
    raise TimeoutError(
        f"Collection {collection_name!r} did not reach GREEN within {timeout}s"
    )


# ---------------------------------------------------------------------------
# Index update
# ---------------------------------------------------------------------------
def snapshot_collection_state(info: models.CollectionInfo) -> dict[str, Any]:
    """
    Extract key fields from CollectionInfo for audit logging.
    Captures actual applied config + indexing completeness.
    """
    def _str(v: Any) -> str:
        return str(v) if v is not None else None

    snap: dict[str, Any] = {
        "status": _str(info.status),
        "optimizer_status": _str(getattr(info, "optimizer_status", None)),
        "points_count": getattr(info, "points_count", None),
        "vectors_count": getattr(info, "vectors_count", None),
        "indexed_vectors_count": getattr(info, "indexed_vectors_count", None),
    }

    cfg = getattr(info, "config", None)

    # Collection-level HNSW config (what was actually applied)
    hnsw = getattr(cfg, "hnsw_config", None)
    if hnsw is not None:
        snap["hnsw_m"] = getattr(hnsw, "m", None)
        snap["hnsw_ef_construct"] = getattr(hnsw, "ef_construct", None)
        snap["hnsw_full_scan_threshold"] = getattr(hnsw, "full_scan_threshold", None)

    # Quantization config
    quant_cfg = getattr(cfg, "quantization_config", None)
    snap["quantization_config"] = _str(quant_cfg)

    # Per-vector config for the named "dense" vector
    params = getattr(cfg, "params", None)
    vectors = getattr(params, "vectors", None)
    if isinstance(vectors, dict) and "dense" in vectors:
        dense = vectors["dense"]
        snap["dense_size"] = getattr(dense, "size", None)
        snap["dense_distance"] = _str(getattr(dense, "distance", None))
        snap["dense_hnsw_config"] = _str(getattr(dense, "hnsw_config", None))
        snap["dense_quantization_config"] = _str(getattr(dense, "quantization_config", None))

    return snap


def update_and_wait(
    client: QdrantClient,
    collection_name: str,
    m: int,
    ef_construct: int,
    quant: QuantizationVariant,
) -> tuple[float, dict[str, Any]]:
    """
    Apply a combined HNSW + quantization update, then wait for GREEN.
    Returns (elapsed_seconds, collection_state_snapshot).
    The snapshot captures the actual applied config for audit logging.
    """
    t0 = time.monotonic()
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(m=m, ef_construct=ef_construct),
        quantization_config=quant.build_config(),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),
    )
    wait_for_ready(client, collection_name)
    elapsed = time.monotonic() - t0
    info = client.get_collection(collection_name)
    snap = snapshot_collection_state(info)
    snap["collection_info_raw"] = info.model_dump(mode="json")
    return elapsed, snap


# ---------------------------------------------------------------------------
# Query execution — all ef_search values in a single pass
# ---------------------------------------------------------------------------
def run_queries_all_ef(
    client: QdrantClient,
    collection_name: str,
    query_embeddings: np.ndarray,    # (N, dim) float32
    query_combos: list[tuple[int, float]],  # (ef_search, oversampling) pairs to run
    top_k: int,
    batch_size: int = 32,
) -> tuple[dict[tuple[int, float], list[list[str]]], float]:
    """
    Run all queries for each (ef_search, oversampling) pair in one pass.
    For each batch of rows, sends len(batch) × len(query_combos) requests
    in a single query_batch_points call.

    Returns ({(ef_search, oversampling): result_ids_list}, total_query_time_s).
    """
    n = len(query_embeddings)
    n_combos = len(query_combos)
    result_ids: dict[tuple[int, float], list[list[str]]] = {
        combo: [[] for _ in range(n)] for combo in query_combos
    }

    t0 = time.monotonic()
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = query_embeddings[start:end]

        # interleaved: row0/combo0, row0/combo1, ..., row1/combo0, row1/combo1, ...
        requests = [
            QueryRequest(
                query=row.tolist(),
                limit=top_k,
                using="dense",
                with_payload=False,
                with_vector=False,
                params=SearchParams(
                    hnsw_ef=efs,
                    exact=False,
                    quantization=QuantizationSearchParams(
                        ignore=False,
                        rescore=True,
                        oversampling=os,
                    ),
                ),
            )
            for row in batch
            for efs, os in query_combos
        ]

        responses = client.query_batch_points(collection_name=collection_name, requests=requests)

        for i, row_idx in enumerate(range(start, end)):
            for j, combo in enumerate(query_combos):
                resp = responses[i * n_combos + j]
                result_ids[combo][row_idx] = [str(pt.id) for pt in resp.points[:top_k]]

    return result_ids, time.monotonic() - t0


# ---------------------------------------------------------------------------
# Recall computation — string UUID edition
# ---------------------------------------------------------------------------
def compute_recall(
    result_ids: list[list[str]],
    gt_hit_ids: np.ndarray,   # (N, K) string UUIDs
    k: int,
) -> tuple[dict[str, float], np.ndarray]:
    """
    Compute recall@k. Returns (summary_stats, per_query_recall).
    per_query_recall is shape (N,) float64 — save to disk for auditing.
    """
    n = len(result_ids)
    per_query = np.empty(n, dtype=np.float64)
    for i in range(n):
        gt_set = set(gt_hit_ids[i, :k])
        gt_set.discard("")
        result_set = set(result_ids[i][:k])
        result_set.discard("")
        per_query[i] = len(gt_set & result_set) / k if gt_set else 1.0

    perfect = int((per_query == 1.0).sum())
    mean = float(per_query.mean())
    sq_mean = float((per_query ** 2).mean())
    variance = max(0.0, sq_mean - mean ** 2)
    stats = {
        "mean_recall_at_k": mean,
        "min_recall_at_k": float(per_query.min()),
        "max_recall_at_k": float(per_query.max()),
        "stddev_recall_at_k": math.sqrt(variance),
        "perfect_query_count": perfect,
        "perfect_query_fraction": perfect / n,
    }
    return stats, per_query


# ---------------------------------------------------------------------------
# Result CSV helpers
# ---------------------------------------------------------------------------
def safe_name(value: Any) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value))


def make_run_key(
    sample: str,
    quant: QuantizationVariant,
    m: int,
    ef_construct: int,
    ef_search: int,
    oversampling: float,
    top_k: int,
) -> str:
    return "__".join(map(safe_name, (
        sample,
        quant.name,
        f"m{m}",
        f"efc{ef_construct}",
        f"efs{ef_search}",
        f"os{oversampling}",
        f"k{top_k}",
    )))


def successful_run_keys(results_csv: Path) -> set[str]:
    if not results_csv.is_file():
        return set()
    with results_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "run_key" not in reader.fieldnames:
            return set()
        return {
            row["run_key"]
            for row in reader
            if row.get("status") == "success" and row.get("run_key")
        }


def ensure_csv_header(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        return
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames == RESULT_FIELDS:
            return
        rows = list(reader)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})
    os.replace(tmp, path)


def append_result(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_header(path)
    exists = path.is_file() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in RESULT_FIELDS})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remote 5-cluster Qdrant recall sweep",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env", type=Path, default=DEFAULT_ENV_FILE,
                   help="Shell env file with QDRANT_URL_N and QDRANT_API_KEY_N")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory where results.csv is written")
    p.add_argument("--collection", type=str, default=None,
                   help="Collection name (auto-discovered from cluster 1 if omitted)")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Number of query vectors per query_batch_points request")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing results and re-run everything")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the execution plan and exit")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run 1 query per cluster with current collection config, no index changes, no CSV writes")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    print(f"Loading cluster configs from {args.env}")
    clusters = load_clusters(args.env)

    print("Building clients…")
    clients: dict[int, QdrantClient] = {c.idx: build_client(c) for c in clusters}

    # Collection name
    collection_name = discover_collection(clients[1], args.collection)
    print(f"Collection: {collection_name!r}")
    for c in clusters[1:]:
        discover_collection(clients[c.idx], collection_name)

    # Distance metric (read once from cluster 1)
    distance_metric = get_distance_metric(clients[1], collection_name)
    print(f"Distance metric: {distance_metric}")

    # Query embeddings
    print(f"Loading queries from {QUERY_EMBEDDINGS_FILE.name}…")
    queries = np.load(QUERY_EMBEDDINGS_FILE).astype(np.float32)[:N_QUERIES]
    n_queries, dim = queries.shape
    print(f"  {n_queries} queries (capped at N_QUERIES={N_QUERIES}), dim={dim}")

    # Ground truth per cluster
    print("Loading ground truth…")
    gt_arrays: dict[int, np.ndarray] = {}
    for c in clusters:
        gt = np.load(c.ground_truth_path, mmap_mode="r")
        gt_arrays[c.idx] = gt
        print(f"  {c.sample}: {gt.shape} {gt.dtype}")

    # Full sweep combos
    combos = list(itertools.product(HNSW_M_VALUES, EF_CONSTRUCT_VALUES, QUANTIZATION_VARIANTS))
    runs_per_cluster = len(combos) * len(QUERY_COMBOS)

    # Per-cluster CSV paths and completed sets
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cluster_csvs: dict[int, Path] = {
        c.idx: args.output_dir / f"results_{c.sample}.csv" for c in clusters
    }
    cluster_completed: dict[int, set[str]] = {
        c.idx: set() if args.no_resume else successful_run_keys(cluster_csvs[c.idx])
        for c in clusters
    }

    print(f"\nSweep plan:")
    print(
        f"  {len(HNSW_M_VALUES)} M × {len(EF_CONSTRUCT_VALUES)} efConstruct × "
        f"{len(QUANTIZATION_VARIANTS)} quant = {len(combos)} triples × "
        f"{len(QUERY_COMBOS)} unique (efSearch, oversampling) combos "
        f"(deduplicated from {len(EF_SEARCH_VALUES) * len(OVERSAMPLING_VALUES)}) = "
        f"{runs_per_cluster} runs/cluster"
    )
    for c in clusters:
        done = len(cluster_completed[c.idx])
        print(f"  {c.sample}: {done} done, {runs_per_cluster - done} remaining → {cluster_csvs[c.idx].name}")

    if args.dry_run:
        print("\n[dry-run] Exiting.")
        return

    if args.smoke_test:
        import json
        smoke_dir = args.output_dir / "smoke_test"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[smoke-test] 1 query per cluster, current config only, writing to {smoke_dir}/")

        # Use a fixed dummy combo for the run_key so the output filenames are realistic
        smoke_quant = QUANTIZATION_VARIANTS[0]
        smoke_m, smoke_efc, smoke_efs, smoke_os = HNSW_M_VALUES[0], EF_CONSTRUCT_VALUES[0], EF_SEARCH_VALUES[0], OVERSAMPLING_VALUES[0]

        for c in clusters:
            # Snapshot current collection state (no update)
            info = clients[c.idx].get_collection(collection_name)
            col_snap = snapshot_collection_state(info)
            col_snap["collection_info_raw"] = info.model_dump(mode="json")

            # Write build log
            build_log_path = smoke_dir / f"build_log_{c.sample}.jsonl"
            with build_log_path.open("w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "event": "smoke_test_snapshot",
                    "sample": c.sample,
                    "note": "current collection state, no index update performed",
                    "collection_state": col_snap,
                }) + "\n")

            # Run 1 query
            resp = clients[c.idx].query_batch_points(collection_name, [
                QueryRequest(
                    query=queries[0].tolist(),
                    limit=TOP_K,
                    using="dense",
                    with_payload=False,
                    with_vector=False,
                    params=SearchParams(
                        hnsw_ef=smoke_efs,
                        exact=False,
                        quantization=QuantizationSearchParams(
                            ignore=False,
                            rescore=True,
                            oversampling=smoke_os,
                        ),
                    ),
                )
            ])
            rids = [str(pt.id) for pt in resp[0].points]
            gt = gt_arrays[c.idx]
            stats, per_query = compute_recall([rids], gt[:1], TOP_K)

            # Write per-query recall npy
            recall_dir = smoke_dir / "per_query_recall" / c.sample
            recall_dir.mkdir(parents=True, exist_ok=True)
            run_key = make_run_key(c.sample, smoke_quant, smoke_m, smoke_efc, smoke_efs, smoke_os, TOP_K)
            npy_path = recall_dir / f"{run_key}.npy"
            np.save(npy_path, per_query)

            # Write CSV row
            csv_path = smoke_dir / f"results_{c.sample}.csv"
            row = {
                **{
                    "run_key": run_key,
                    "cluster_idx": c.idx,
                    "sample": c.sample,
                    "cluster_url": c.url,
                    "collection_name": collection_name,
                    "query_count": 1,
                    "vector_dim": dim,
                    "distance_metric": distance_metric,
                    "quantization_variant": smoke_quant.name,
                    "quantization_type": smoke_quant.quantization_type,
                    "quantization_always_ram": str(smoke_quant.always_ram),
                    "quantization_turbo_bits": smoke_quant.turbo_bits,
                    "quantization_binary_encoding": smoke_quant.binary_encoding,
                    "hnsw_m": smoke_m,
                    "ef_construct": smoke_efc,
                    "ef_search": smoke_efs,
                    "oversampling": smoke_os,
                    "top_k": TOP_K,
                    "update_time_s": "",
                    "status": "success",
                    "error": "",
                    "query_time_s": "",
                },
                **stats,
            }
            append_result(csv_path, row)

            print(
                f"  cluster {c.idx} ({c.sample}): "
                f"{len(rids)} results, recall={stats['mean_recall_at_k']:.4f}, "
                f"top-3={rids[:3]}"
            )

        print(f"[smoke-test] Files written to {smoke_dir}/")
        answer = input("\nDelete smoke test files? [y/N]: ").strip().lower()
        if answer == "y":
            import shutil
            shutil.rmtree(smoke_dir)
            print(f"Deleted {smoke_dir}/")
        return

    print_lock = threading.Lock()

    def base_row(
        c: ClusterConfig,
        quant: QuantizationVariant,
        m: int,
        ef_construct: int,
        ef_search: int,
        oversampling: float,
        run_key: str,
        update_time: float | str,
    ) -> dict[str, Any]:
        return {
            "run_key": run_key,
            "cluster_idx": c.idx,
            "sample": c.sample,
            "cluster_url": c.url,
            "collection_name": collection_name,
            "query_count": n_queries,
            "vector_dim": dim,
            "distance_metric": distance_metric,
            "quantization_variant": quant.name,
            "quantization_type": quant.quantization_type,
            "quantization_always_ram": str(quant.always_ram),
            "quantization_turbo_bits": quant.turbo_bits,
            "quantization_binary_encoding": quant.binary_encoding,
            "hnsw_m": m,
            "ef_construct": ef_construct,
            "ef_search": ef_search,
            "oversampling": oversampling,
            "top_k": TOP_K,
            "update_time_s": update_time,
        }

    # -----------------------------------------------------------------------
    # Main sweep: one independent worker thread per cluster.
    # Each worker manages its own CSV and completed set — no shared state.
    # -----------------------------------------------------------------------
    def cluster_worker(c: ClusterConfig) -> None:
        client = clients[c.idx]
        gt = gt_arrays[c.idx]
        csv_path = cluster_csvs[c.idx]
        done: set[str] = set(cluster_completed[c.idx])

        # Per-cluster output dirs
        recall_dir = args.output_dir / "per_query_recall" / c.sample
        recall_dir.mkdir(parents=True, exist_ok=True)
        build_log_path = args.output_dir / f"build_log_{c.sample}.jsonl"

        def log_build(entry: dict[str, Any]) -> None:
            import json
            with build_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

        def emit(row: dict[str, Any], per_query: np.ndarray | None = None, *, mark_done: bool = False) -> None:
            append_result(csv_path, row)
            if mark_done and row.get("status") == "success":
                done.add(str(row["run_key"]))
                if per_query is not None:
                    np.save(recall_dir / f"{row['run_key']}.npy", per_query)
            key = row.get("run_key", "")
            with print_lock:
                if row.get("status") == "success":
                    r = row.get("mean_recall_at_k", "?")
                    r_str = f"{r:.4f}" if isinstance(r, float) else str(r)
                    print(f"  [c{c.idx}] {key}: recall={r_str}")
                else:
                    print(f"  [c{c.idx}] {key}: FAILED – {row.get('error', '')}")

        for m, ef_construct, quant in combos:
            needs = [
                (efs, os) for efs, os in QUERY_COMBOS
                if make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K) not in done
            ]
            if not needs:
                continue

            # --- index update ---
            # Called only when there are pending queries for this combo.
            # If we crashed after the index was built but before all queries
            # finished, this call is instant — Qdrant sees no config change.
            try:
                update_time, col_snap = update_and_wait(client, collection_name, m, ef_construct, quant)
                log_build({
                    "event": "update_ok", "sample": c.sample,
                    "m": m, "ef_construct": ef_construct, "quant": quant.name,
                    "update_time_s": update_time,
                    "collection_state": col_snap,
                })
                with print_lock:
                    indexed = col_snap.get("indexed_vectors_count")
                    total = col_snap.get("points_count")
                    print(f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: ready in {update_time:.1f}s  indexed={indexed}/{total}")
            except Exception as exc:
                log_build({
                    "event": "update_failed", "sample": c.sample,
                    "m": m, "ef_construct": ef_construct, "quant": quant.name,
                    "error": str(exc),
                })
                with print_lock:
                    print(f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: update FAILED – {exc}")
                for efs, os in needs:
                    run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                    emit({
                        **base_row(c, quant, m, ef_construct, efs, os, run_key, ""),
                        "status": "failed",
                        "error": f"index update failed: {exc}",
                    })
                continue

            # --- query all (ef_search, oversampling) combos in a single pass ---
            try:
                all_rids, total_qtime = run_queries_all_ef(
                    client, collection_name, queries, needs, TOP_K, args.batch_size
                )
            except Exception as exc:
                for efs, os in needs:
                    run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                    emit({
                        **base_row(c, quant, m, ef_construct, efs, os, run_key, update_time),  # type: ignore[possibly-unbound]
                        "status": "failed", "error": f"query failed: {exc}",
                    })
                continue

            with print_lock:
                print(f"  [c{c.idx}] M={m} efc={ef_construct} {quant.name}: {len(needs)} combos queried in {total_qtime:.1f}s")

            # sanity check: warn if any query returned fewer than top_k results
            for efs, os in needs:
                short = sum(1 for r in all_rids[(efs, os)] if len(r) < TOP_K)
                if short:
                    with print_lock:
                        print(f"  [c{c.idx}] WARNING: {short} queries returned <{TOP_K} results at efs={efs} os={os}")

            qtime_per_combo = total_qtime / len(needs)
            for efs, os in needs:
                run_key = make_run_key(c.sample, quant, m, ef_construct, efs, os, TOP_K)
                brow = base_row(c, quant, m, ef_construct, efs, os, run_key, update_time)
                try:
                    recall_stats, per_query = compute_recall(all_rids[(efs, os)], gt, TOP_K)
                    emit(
                        {**brow, "status": "success", "query_time_s": qtime_per_combo, **recall_stats},
                        per_query=per_query,
                        mark_done=True,
                    )
                except Exception as exc:
                    emit({**brow, "status": "failed", "error": f"recall failed: {exc}"})

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(clusters)) as pool:
        futures = [pool.submit(cluster_worker, c) for c in clusters]
        for fut in concurrent.futures.as_completed(futures):
            fut.result()

    print(f"\nDone.  Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
