#!/usr/bin/env python3
"""Create a minimal Weaviate collection for UUID + self-provided vectors.

This repo's `ip_registry.txt` stores both HTTP and gRPC ports in the format:
rank,node,ip,http,grpc,gossip,data,raft,raft_internal
"""

import argparse
import csv
import os
import pprint
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Set

import numpy as np
import weaviate
import weaviate.classes as wvc

COLLECTION_NAME = os.getenv("COLLECTION_NAME")


def getenv_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


HNSW_M = getenv_int("HNSW_M", 16)
HNSW_EF_CONSTRUCTION = getenv_int("HNSW_EF_CONSTRUCTION", 100)
HNSW_EF_SEARCH = getenv_int("HNSW_EF_SEARCH", 64)
SHARD_COUNT = getenv_int("SHARD_COUNT", 0)
SHARD_TRIGGER_SLACK = getenv_int("SHARD_TRIGGER_SLACK", 5000)
INDEX_TYPE = os.getenv("INDEX_TYPE", "DYNAMIC").strip().upper()


def distance_metric():
    metric = os.getenv("DISTANCE_METRIC", "COSINE").strip().upper()
    if metric in {"IP", "DOT"}:
        return wvc.config.VectorDistances.DOT
    if metric in {"L2", "L2-SQUARED", "L2_SQUARED"}:
        return wvc.config.VectorDistances.L2_SQUARED
    return wvc.config.VectorDistances.COSINE


def npy_row_count(path: Path) -> int:
    arr = np.load(path, mmap_mode="r")
    if arr.ndim != 2:
        raise ValueError("expected 2D npy array in {0}, got shape {1!r}".format(path, arr.shape))
    return int(arr.shape[0])


def least_populated_shard_size(total_objects: int, shard_count: int) -> int:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive, got {0}".format(shard_count))
    return total_objects // shard_count


def resolve_dynamic_threshold(shard_count: int) -> int:
    explicit = os.getenv("HNSW_DYNAMIC_THRESHOLD")
    if explicit is not None and explicit.strip() != "":
        return int(explicit)

    task = os.getenv("TASK", "").strip().upper()

    def adjust_threshold(total_objects: int) -> int:
        if task == "INSERT":
            threshold = total_objects * 2
            print(
                "Auto-derived HNSW_DYNAMIC_THRESHOLD={0} from total_objects={1} "
                "for TASK=INSERT to isolate insert performance.".format(
                    threshold,
                    total_objects,
                ),
                flush=True,
            )
            return threshold

        if task in ["INDEX", "QUERY", "MIXED"]:
            least_populated = least_populated_shard_size(total_objects, shard_count)
            if shard_count == 1 and task in ["INDEX", "QUERY"]:
                threshold = max(0, total_objects - 1)
                print(
                    "Auto-derived HNSW_DYNAMIC_THRESHOLD={0} from total_objects={1} "
                    "for single-node TASK={2}.".format(
                        threshold,
                        total_objects,
                        task,
                    ),
                    flush=True,
                )
            else:
                slack = SHARD_TRIGGER_SLACK if shard_count > 1 else 0
                threshold = max(0, least_populated - slack)
                print(
                    "Auto-derived HNSW_DYNAMIC_THRESHOLD={0} from total_objects={1}, "
                    "shard_count={2}, least_populated_shard={3}, shard_trigger_slack={4} "
                    "for TASK={5}.".format(
                        threshold,
                        total_objects,
                        shard_count,
                        least_populated,
                        slack,
                        task,
                    ),
                    flush=True,
                )
            return threshold

        return total_objects

    corpus_size = os.getenv("INSERT_CORPUS_SIZE")
    if corpus_size is not None and corpus_size.strip() != "":
        return adjust_threshold(int(corpus_size))

    data_path = os.getenv("INSERT_DATA_FILEPATH")
    if data_path is not None and data_path.strip() != "":
        return adjust_threshold(npy_row_count(Path(data_path)))

    raise ValueError(
        "HNSW_DYNAMIC_THRESHOLD resolution requires HNSW_DYNAMIC_THRESHOLD, "
        "INSERT_CORPUS_SIZE, or INSERT_DATA_FILEPATH."
    )


def resolve_shard_count() -> int:
    if SHARD_COUNT != 0:
        return SHARD_COUNT

    worker_count = os.getenv("N_WORKERS")
    if worker_count is None or worker_count.strip() == "":
        raise ValueError(
            "SHARD_COUNT=0 requires N_WORKERS to be set so shard count can be derived."
        )

    return int(worker_count)


@dataclass(frozen=True)
class NodeEntry:
    rank: int
    node_name: str
    ip: str
    http_port: int
    grpc_port: int


def parse_registry(registry_path: Path) -> List[NodeEntry]:
    if not registry_path.exists():
        raise FileNotFoundError("Registry file not found: {0}".format(registry_path))

    entries = []
    seen_ranks = set()  # type: Set[int]

    with registry_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, row in enumerate(reader, start=1):
            if not row or all(not item.strip() for item in row):
                continue

            if len(row) < 5:
                raise ValueError(
                    "{0}:{1}: expected at least 5 columns, got {2}".format(
                        registry_path, line_number, len(row)
                    )
                )

            try:
                rank = int(row[0].strip())
                node_name = row[1].strip()
                ip = row[2].strip()
                http_port = int(row[3].strip())
                grpc_port = int(row[4].strip())
            except ValueError as exc:
                raise ValueError(
                    "{0}:{1}: failed to parse rank/http/grpc port from {2!r}".format(
                        registry_path, line_number, row
                    )
                ) from exc

            if rank in seen_ranks:
                continue

            seen_ranks.add(rank)
            entries.append(
                NodeEntry(
                    rank=rank,
                    node_name=node_name,
                    ip=ip,
                    http_port=http_port,
                    grpc_port=grpc_port,
                )
            )

    entries.sort(key=lambda entry: entry.rank)
    return entries


def connect_from_registry(registry_path: Path, rank: int):
    entries = parse_registry(registry_path)
    if not entries:
        raise ValueError("No registry entries found in {0}".format(registry_path))

    matches = [entry for entry in entries if entry.rank == rank]
    if not matches:
        raise ValueError("Rank {0} not found in {1}".format(rank, registry_path))

    node = matches[0]
    client = weaviate.connect_to_custom(
        http_host=node.ip,
        http_port=node.http_port,
        http_secure=False,
        grpc_host=node.ip,
        grpc_port=node.grpc_port,
        grpc_secure=False,
    )
    return client, node


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a minimal Weaviate collection with UUID + vector only"
    )
    parser.add_argument(
        "--registry",
        default="ip_registry.txt",
        help="Path to the registry file (default: ip_registry.txt)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Registry rank to connect to (default: 0)",
    )
    args = parser.parse_args()

    registry_path = Path(args.registry)
    client = None

    try:
        shard_count = resolve_shard_count()
        client, node = connect_from_registry(registry_path, args.rank)

        if not client.is_ready():
            print(
                "ERROR: Weaviate node at http://{0}:{1} is not ready".format(
                    node.ip, node.http_port
                ),
                file=sys.stderr,
            )
            return 1

        exists = client.collections.exists(COLLECTION_NAME)
        if exists:
            client.collections.delete(COLLECTION_NAME)
            exists = False
            print("Deleted existing collection {0}".format(COLLECTION_NAME), flush=True)

        if exists:
            print("Collection {0} already exists".format(COLLECTION_NAME), flush=True)
            collection = client.collections.use(COLLECTION_NAME)
            collection_config = collection.config.get()
            print("Collection status: exists=True" ,flush=True)
            print("Collection definition:", flush=True)
            print(collection_config, flush=True)
            return 0

        vector_index_config = wvc.config.Configure.VectorIndex.hnsw(
            distance_metric=distance_metric(),
            ef=HNSW_EF_SEARCH,
            ef_construction=HNSW_EF_CONSTRUCTION,
            max_connections=HNSW_M,
        )
        if INDEX_TYPE == "DYNAMIC":
            hnsw_dynamic_threshold = resolve_dynamic_threshold(shard_count)
            vector_index_config = wvc.config.Configure.VectorIndex.dynamic(
                distance_metric=distance_metric(),
                threshold=hnsw_dynamic_threshold,
                hnsw=vector_index_config,
                flat=wvc.config.Configure.VectorIndex.flat(
                    distance_metric=distance_metric(),
                ),
            )
        elif INDEX_TYPE != "HNSW":
            raise ValueError(
                "INDEX_TYPE must be DYNAMIC or HNSW, got {0!r}".format(INDEX_TYPE)
            )

        client.collections.create(
            name=COLLECTION_NAME,
            description="Minimal collection storing only UUIDs and self-provided vectors",
            vector_config=wvc.config.Configure.Vectors.self_provided(
                vector_index_config=vector_index_config
            ),
            sharding_config=(
                wvc.config.Configure.sharding(
                    desired_count=shard_count
                )
            ),
            properties=[
                wvc.config.Property(
                    name="doc_id",
                    data_type=wvc.config.DataType.INT,
                ),
            ],
        )

        print(
            "Created collection {0} on http://{1}:{2} using grpc port {3}".format(
                COLLECTION_NAME, node.ip, node.http_port, node.grpc_port
            ), flush=True
        )
        print("Shard count: {0}".format(shard_count), flush=True)
        print("Schema: no user properties; objects store UUID plus supplied vector")
        collection = client.collections.use(COLLECTION_NAME)
        collection_config = collection.config.get()
        print("Collection status: exists={0}".format(client.collections.exists(COLLECTION_NAME)))
        print("Collection definition:")
        print(collection_config, flush=True)
        return 0

    except Exception as exc:
        print("ERROR: {0}".format(exc), file=sys.stderr)
        return 1
    finally:
        if client is not None:
            client.close()


if __name__ == "__main__":
    raise SystemExit(main())
