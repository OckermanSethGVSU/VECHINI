import os, time, subprocess, glob
from urllib.request import pathname2url
import csv, json
import numpy as np
from qdrant_client import QdrantClient, AsyncQdrantClient, models
from qdrant_client.models import SearchRequest, QueryRequest

collection_name = os.environ["COLLECTION_NAME"].strip()

# ---------- Helpers ----------
def file_line_count(path):
    """Check if a file exists, and if so, return its line count."""
    if not os.path.isfile(path):
        print(f"File not found: {path}")
        return 0

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        count = sum(1 for _ in f)
    return count - 1

def load_ip_from_file(filepath):
    with open(filepath, "r") as f:
        rank, ip, port = f.readline().strip().split(",")
    return ip, int(port)


def load_all_ips(filepath):
    ips = []
    with open(filepath, "r") as f:
        for line in f:
            rank, ip, port = line.strip().split(",")
            ips.append((int(rank), ip, int(port)))
    return ips

def get_line(path, line_no):
    with open(path) as f:
        for i, line in enumerate(f, start=0):
            if i == line_no:
                return line.strip()


def mark_perf_event(filename):
    perf_dir = "runtime_state"
    os.makedirs(perf_dir, exist_ok=True)
    open(os.path.join(perf_dir, filename), "a").close()

def is_mixed_task():
    return os.getenv("TASK", "").strip().lower() == "mixed"

def should_mark_index_workflow():
    return os.getenv("TASK", "").strip().lower() in {"index", "mixed"}


def env_int(name, default):
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value.strip())

# ---------- Main ----------
def main():
    
    base_ip, file_port = load_ip_from_file("ip_registry.txt")
    # derive REST/gRPC explicitly
    rest_port = file_port - 2      # e.g., if file has 6335 -> rest=6333
    grpc_port = rest_port + 1      # -> grpc=6334

    collection_name = os.environ["COLLECTION_NAME"].strip()
    hnsw_m = env_int("HNSW_M", 16)
    ef_construction = env_int("HNSW_EF_CONSTRUCTION", 100)

    # Sync client for admin/index ops
    client = QdrantClient(
        host=base_ip,
        port=rest_port,
        grpc_port=grpc_port,
        prefer_grpc=True,
        timeout=6000,
        grpc_options={"grpc.enable_http_proxy": 0},
    )
    
    
    # wait for changes to take effect
    while True:
        info = client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            break
        time.sleep(0.25)

    info = client.get_collection(collection_name)
    print("*********************** Un-Indexed Collection Info ***********************", flush=True)
    print(info, flush=True)
    print("**************************************************************************", flush=True)
    # re-enable graph and time rebuild
    mark_workflow = should_mark_index_workflow()
    if mark_workflow:
        mark_perf_event("workflow_start.txt")
        time.sleep(5)
    
    t1 = time.time()
    client.update_collection(
        collection_name=collection_name,
        hnsw_config=models.HnswConfigDiff(m=hnsw_m, ef_construct=ef_construction),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1),

    )
    while True:
        info = client.get_collection(collection_name)
        if info.status == models.CollectionStatus.GREEN:
            t2 = time.time()
            if mark_workflow:
                mark_perf_event("workflow_stop.txt")
            break
        time.sleep(0.01)

    if is_mixed_task():
        # Restore the regular indexing threshold after the forced rebuild finishes.
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=models.OptimizersConfigDiff(indexing_threshold=10_000),
        )
        while True:
            info = client.get_collection(collection_name)
            if info.status == models.CollectionStatus.GREEN:
                break
            time.sleep(0.01)

    print("", flush=True)
    print("*********************** Indexed Collection Info ***********************", flush=True)
    print(info, flush=True)
    print("***********************************************************************", flush=True)
    with open(f"index_time.txt", "w") as f:
        f.write(str(t2 - t1))
 

if __name__ == "__main__":
    main()
