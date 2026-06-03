import os
import json
import time
import requests
import numpy as np
from pymilvus import MilvusClient
from pymilvus import utility

"""
IMPORTANT NOTE: if the collection was initialized with a FLAT index and this script
drops that index to rebuild HNSW or GPU_CAGRA on the same vector field, Milvus 2.6.6
hits a bug that massively degrades query performance. As of March 27th, 2026, that
bug is not resolved. This workaround is still required for our indexing experiments.
"""


RUNTIME_STATE_DIR = os.getenv("RUNTIME_STATE_DIR", "./runtime_state")


def runtime_state_path(name: str) -> str:
    return os.path.join(RUNTIME_STATE_DIR, name)


def create_index_with_fallback_poll(
    client,
    collection_name,
    index_params,
    timeout_s=18000,
    poll_interval_s=0.5,
):
    """
    Try synchronous index creation first.
    If sync=True fails, fall back to polling index state until finished.

    Returns the final index description/info when finished.
    Raises on timeout or terminal failure.
    """

    try:
        resp = client.create_index(collection_name, index_params, sync=True)
        return 
    except Exception as e:
        sync_error = e
        print(f"sync create_index failed, falling back to manual polling", flush=True)

    # Fallback: poll until the index state says finished
    last_print = 0
    start = time.time()
    while True:
        elapsed = time.time() - start
        if elapsed > timeout_s:
            raise TimeoutError(
                f"Timed out after {timeout_s}s waiting for index build. "
                f"Original sync error: {sync_error}"
            )

        try:
            info = client.describe_index(collection_name, "vector")

            state = info["state"]
            indexed_rows = info["indexed_rows"]
            total_rows = info["total_rows"]
            pending_index_rows = info["pending_index_rows"]

            now = time.time()
            if now - last_print >= 600:  # 10 minutes
                print(
                    f"index state={state} indexed_rows={indexed_rows}/{total_rows} "
                    f"pending_index_rows={pending_index_rows}",
                    flush=True,
                )
                last_print = now

            state_str = state.strip().lower()

            if state_str == "finished":
                return

            if state_str == "failed":
                raise RuntimeError(f"Index build entered failure state: {info}")

        except Exception as e:
            # Transient polling failures can happen; keep going until timeout
            print(f"transient polling error: {e}", flush=True)
            time.sleep(3)

        time.sleep(poll_interval_s)



def wait_for_milvus(host, port):
    print(f"Waiting for Milvus at {host}:{port}...")
    for _ in range(60):
        try:
            r = requests.get(f"http://{host}:{port}/healthz", timeout=2)
            if r.status_code == 200:
                print("Milvus is good.")
                return
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError("Milvus did not respond in time")

def read_ip_from_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def resolve_target_index_rows() -> int:
    explicit = (os.getenv("INSERT_CORPUS_SIZE") or "").strip()
    if explicit:
        return int(explicit)

    path = (os.getenv("INSERT_DATA_FILEPATH") or "").strip()
    if not path:
        raise RuntimeError("INSERT_CORPUS_SIZE or INSERT_DATA_FILEPATH is required for indexing")

    arr = np.load(os.path.expanduser(path), mmap_mode="r")
    if arr.ndim != 2:
        raise RuntimeError(f"expected 2D npy input at {path}, got shape {arr.shape}")
    return int(arr.shape[0])


MILVUS_HOST = read_ip_from_file(runtime_state_path("worker.ip"))
MILVUS_GRPC_PORT = int(os.getenv("MILVUS_GRPC_PORT", "20001"))
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN", "root:Milvus")
collection_name = os.getenv("COLLECTION_NAME").strip()
FLUSH_BEFORE_INDEX = os.getenv("FLUSH_BEFORE_INDEX", "TRUE").strip().lower() == "true"
INIT_FLAT_INDEX = os.getenv("INIT_FLAT_INDEX", "TRUE").strip().lower() == "true"
distance_metric = os.environ["DISTANCE_METRIC"].strip().lower()
GPU_INDEX = os.environ["GPU_INDEX"].strip().lower() == "true"


client = MilvusClient(f"http://{MILVUS_HOST}:{MILVUS_GRPC_PORT}", token=MILVUS_TOKEN)

if INIT_FLAT_INDEX:
    print(
        "Collection was initialized with a FLAT index; dropping it before building the target index. "
        "WARNING: this sequence triggers a Milvus bug (as of 03/27/26 and v.2.6.6) that massively degrades query performance.",
        flush=True,
    )
    print(f"Indexes for {collection_name}: ", client.list_indexes(collection_name), flush=True)
    print(client.describe_index(collection_name, "vector"), flush=True)

    client.release_collection(collection_name=collection_name)
    client.drop_index(collection_name, "vector")

    while True:
        idxs = client.list_indexes(collection_name=collection_name)
        if not idxs:
            break
        time.sleep(0.5)

    print(f"Indexes for {collection_name}: ", client.list_indexes(collection_name), flush=True)
    time.sleep(60)
else:
    print(
        "Collection was initialized without a FLAT index; building the target index directly.",
        flush=True,
    )



index_params = client.prepare_index_params()
match distance_metric:
    case "dot" | "ip" | "innerproduct":
        metric = "IP"
    case "cosine":
        metric = "COSINE"
    case "euclidan" | "l2":
        metric = "L2"
    case _:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

if GPU_INDEX:
    index_params.add_index(
    field_name="vector",      
    index_type="GPU_CAGRA",        
    metric_type=metric,
    params={
        "intermediate_graph_degree": 64,
        "graph_degree": 16,
    },
    mmap_enabled=False,
    )
else: 
    index_params.add_index(
        field_name="vector",      # your vector_field_name
        index_type="HNSW",        # <- FLAT index
        metric_type=metric,     # or "L2", "IP", etc.
        params={
            "M":16,
            "efConstruction": 100
            },                # FLAT doesn't need extra params
        mmap_enabled=False,
    )

if FLUSH_BEFORE_INDEX:
    client.flush(collection_name)
    print("Data flushed before index build", flush=True)
else:
    print("Skipping flush before index build", flush=True)

t1 = time.time()
resp = create_index_with_fallback_poll(client, collection_name, index_params)
t2 = time.time()

INDEX_PENDING_STABLE_SECONDS = float(os.getenv("INDEX_PENDING_STABLE_SECONDS", "60"))
TARGET_INDEX_ROWS = resolve_target_index_rows()

timeout_duration = 180

wait_start = time.time()
target_count_reached_at = None
stabilized_at = None
last_print = 0
pending_zero_since = None

while True:
    try:
        index_status = client.describe_index(collection_name, 'vector')
        indexed_rows = int(index_status['indexed_rows'])
        total_rows = int(index_status['total_rows'])
        pending_index_rows = int(index_status['pending_index_rows'])

        stats = client.get_collection_stats(collection_name)
        collection_current_count = int(stats["row_count"])
    except Exception as e:
        if time.time() - wait_start > 60 * timeout_duration:
            raise TimeoutError(f"Timed out while polling Milvus state: {e}", flush=True)
        print(f"[warn] polling failed: {e}", flush=True)
        time.sleep(10)
        continue

    now = time.time()
    if now - last_print > 60:
        print(
            f"[wait] rows={collection_current_count} indexed_rows={indexed_rows}/{total_rows} "
            f"target_rows={TARGET_INDEX_ROWS} pending_index_rows={pending_index_rows}",
            flush=True,
        )
        last_print = now

    if FLUSH_BEFORE_INDEX:
        if target_count_reached_at is None:
            if indexed_rows == TARGET_INDEX_ROWS:
                target_count_reached_at = now
                print(f"[wait] target indexed row count reached: {TARGET_INDEX_ROWS}", flush=True)
            else:
                if time.time() - wait_start > (60 * timeout_duration):
                    raise TimeoutError(
                        f"indexed_rows did not reach target {TARGET_INDEX_ROWS} within 10 minutes"
                    )
                time.sleep(1)
                continue
    else:
        if target_count_reached_at is None:
            target_count_reached_at = t2

    if pending_index_rows == 0:
        if pending_zero_since is None:
            pending_zero_since = now
            print(
                f"[wait] pending_index_rows reached 0; starting stabilization window of "
                f"{INDEX_PENDING_STABLE_SECONDS} seconds",
                flush=True,
            )
        elif now - pending_zero_since >= INDEX_PENDING_STABLE_SECONDS:
            stabilized_at = pending_zero_since
            print("[wait] pending_index_rows remained 0 through stabilization window", flush=True)
            break
    else:
        if pending_zero_since is not None:
            print("[wait] pending_index_rows became non-zero again; restarting stabilization polling", flush=True)
        pending_zero_since = None

    if time.time() - wait_start > (60 * timeout_duration):
        raise TimeoutError(
            f"pending_index_rows did not stay at 0 for {INDEX_PENDING_STABLE_SECONDS} seconds within 30 minutes"
        )

    time.sleep(1)

initial_index_time = t2 - t1
if FLUSH_BEFORE_INDEX:
    target_count_time = target_count_reached_at - t2
else:
    target_count_time = initial_index_time
stabilization_time = stabilized_at - target_count_reached_at

with open("index_time.txt", "w", encoding="utf-8") as f:
    f.write("inital_index,target_count,stablization\n")
    f.write(f"{initial_index_time},{target_count_time},{stabilization_time}\n")



load_start = time.time()
client.load_collection(collection_name)

while True:
    state = client.get_load_state(collection_name)
    if str(state['state']) == "Loaded":
        break
    time.sleep(0.5)
load_end = time.time()


with open(f"collection_time.txt", "w") as f:
        f.write(str(load_end- load_start))


while True:
    res = client.get_load_state(collection_name=collection_name)
    if "Loaded" in str(res.get("state", "")):
        break
    time.sleep(5)


res = client.describe_collection(collection_name=collection_name)
index_status = client.describe_index(collection_name, 'vector')
segments = client.list_loaded_segments(collection_name=collection_name)

print("**************Collection State After Indexing***************",flush=True)
print(res,flush=True)
print("************************************************************",flush=True)
print("**************Index State After Indexing********************",flush=True)
print(index_status, flush=True)
print("************************************************************",flush=True)
print("**************Segment State*********************",flush=True)
print("Number of Segments:", len(segments), "Segment Info: ", segments, flush=True)
print("************************************************************",flush=True)
