from qdrant_client import QdrantClient, models
from collections import defaultdict
from qdrant_client.models import SearchRequest
import time
import json
import os
from pathlib import Path
from qdrant_client.http.models import (
    ReplicateShard,
    ReplicateShardOperation,
    DropReplicaOperation,
    Replica
)


def is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value.strip())


def env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    return int(value.strip())


def build_quantization_config():
    quantization_type = os.getenv("QUANTIZATION_TYPE", "NONE").strip().upper()
    always_ram = is_truthy(os.getenv("QUANTIZATION_ALWAYS_RAM"))

    match quantization_type:
        case "NONE" | "":
            return None
        case "SCALAR":
            quantile_raw = os.getenv("QUANTIZATION_SCALAR_QUANTILE", "").strip()
            quantile = float(quantile_raw) if quantile_raw else None
            if quantile is not None and not 0 < quantile <= 1:
                raise ValueError("QUANTIZATION_SCALAR_QUANTILE must be in the range (0, 1]")
            return models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=quantile,
                    always_ram=always_ram,
                ),
            )
        case "BINARY":
            encoding_name = os.getenv(
                "QUANTIZATION_BINARY_ENCODING", "DEFAULT"
            ).strip().upper()
            encoding = None
            if encoding_name != "DEFAULT":
                encoding = getattr(models.BinaryQuantizationEncoding, encoding_name)
            return models.BinaryQuantization(
                binary=models.BinaryQuantizationConfig(
                    encoding=encoding,
                    always_ram=always_ram,
                ),
            )
        case "PRODUCT":
            compression_name = os.getenv(
                "QUANTIZATION_PRODUCT_COMPRESSION", "X16"
            ).strip().upper()
            return models.ProductQuantization(
                product=models.ProductQuantizationConfig(
                    compression=getattr(models.CompressionRatio, compression_name),
                    always_ram=always_ram,
                ),
            )
        case "TURBO":
            bits_name = os.getenv("QUANTIZATION_TURBO_BITS", "BITS4").strip().upper()
            return models.TurboQuantization(
                turbo=models.TurboQuantQuantizationConfig(
                    bits=getattr(models.TurboQuantBitSize, bits_name),
                    always_ram=always_ram,
                ),
            )
        case _:
            raise ValueError(f"Unknown quantization type: {quantization_type}")


def wait_for_shard_transfer(client, collection_name, shard_id, timeout=60, poll_interval=2):
    """Waits until the specified shard is no longer in a transfer state."""
    start = time.time()
    while time.time() - start < timeout:
        cluster_info = client.http.distributed_api.collection_cluster_info(collection_name).result
        active = [t for t in cluster_info.shard_transfers if t.shard_id == shard_id]
        if not active:
            return True
        # print(f"⏳ Waiting on shard {shard_id} transfer...")
        time.sleep(poll_interval)
    raise TimeoutError(f"Timed out waiting for shard {shard_id} transfer to finish")


def printShardStatus(nodes, collection_name):
    node_shard_map = {}
    for host, port in nodes:
        client = QdrantClient(
            host=host,
            port=port,
            grpc_port=port  + 1,
            prefer_grpc=True,
            timeout=600,
            grpc_options={"grpc.enable_http_proxy": 0},
        )
        info = client.http.distributed_api.collection_cluster_info(collection_name).result
    
        # print(info)
        local_shard_ids = [shard.shard_id for shard in info.local_shards]
        node_shard_map[f"{host}:{port}"] = local_shard_ids

    # Print the result
    for node, shards in node_shard_map.items():
        print(f"{node}: {shards}",flush=True)



def map_shard_ids_to_keys(
    topo,
    collection_name: str,
) -> dict:
    
    shard_to_key_map = {}
    for addr in topology:
        host, port = addr.split(":")
        port = int(port)
        client = QdrantClient(
            host=host,
            port=port,
            grpc_port=port  + 1,
            prefer_grpc=False,
            timeout=600,
            grpc_options={"grpc.enable_http_proxy": 0},
        )
        cluster_info = client.http.distributed_api.collection_cluster_info(collection_name=collection_name).result
        # print(cluster_info)
        if cluster_info.local_shards:
            # print(addr)
            for shard in cluster_info.local_shards:
                shard_to_key_map[shard.shard_id] = shard.shard_key

    
        

    return shard_to_key_map

def load_topology(file_path, use_localhost=True):
    nodes = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # skip empty/comment lines
            parts = line.split(",")
            if len(parts) != 3:
                raise ValueError(f"Invalid line: {line}")
            _, ip, port = parts
            host = "localhost" if use_localhost else ip
            nodes.append((host, int(port) - 2))
    return nodes

nodes = load_topology("ip_registry.txt", use_localhost=False)
run_mode = os.getenv("RUN_MODE", "PBS").strip().lower()
rebalance_topology = is_truthy(os.getenv("REBALANCE_TOPOLOGY"))
collection_name = os.environ["COLLECTION_NAME"].strip()
initial_sleep_seconds = env_int("CONFIGURE_COLLECTION_INITIAL_SLEEP_SECONDS", 2)
ready_sleep_seconds = env_int("CONFIGURE_COLLECTION_READY_SLEEP_SECONDS", 10)



topology = {}
total_shards = len(nodes)
# put one shard per node
for i in range(len(nodes)):
    topology[f"{nodes[i][0]}:{nodes[i][1]}"] = [(i % total_shards)]

vector_dim = int(os.environ["VECTOR_DIM"])
hnsw_m = env_int("HNSW_M", 16)
ef_construction = env_int("HNSW_EF_CONSTRUCTION", 100)
max_segment_size = env_optional_int("MAX_SEGMENT_SIZE")
default_segment_number = env_optional_int("DEFAULT_SEGMENT_NUMBER")
distance_metric = os.environ["DISTANCE_METRIC"].strip().lower()
quantization_config = build_quantization_config()

match distance_metric:
    case "dot" | "ip" | "innerproduct":
        metric = models.Distance.DOT
    case "cosine":
        metric = models.Distance.COSINE
    case "euclidean" | "euclid" | "euclidan" | "l2":
        metric = models.Distance.EUCLID
    case _:
        raise ValueError(f"Unknown distance metric: {distance_metric}")

while True:
    try:
        optimizers_kwargs = {"indexing_threshold": 0}
        if max_segment_size is not None:
            optimizers_kwargs["max_segment_size"] = max_segment_size
        if default_segment_number is not None:
            optimizers_kwargs["default_segment_number"] = default_segment_number

        client = QdrantClient(
                host=nodes[0][0],
                port=nodes[0][1],
                grpc_port=nodes[0][1] + 1,
                prefer_grpc=True,
                timeout=600,
                grpc_options={"grpc.enable_http_proxy": 0},
            )
        collection_kwargs = {
            "collection_name": collection_name,
            "vectors_config": models.VectorParams(size=vector_dim, distance=metric),
            "shard_number": len(nodes),
            "hnsw_config": models.HnswConfigDiff(
                m=hnsw_m, ef_construct=ef_construction
            ),
            "optimizers_config": models.OptimizersConfigDiff(**optimizers_kwargs),
            "replication_factor": 1,
        }
        if quantization_config is not None:
            collection_kwargs["quantization_config"] = quantization_config

        client.recreate_collection(
            **collection_kwargs,
        )
        
        break
    except Exception as e:
        # Optional: log the error
        print(f"Error while recreating collection: {e}", flush=True)

        # Sleep for 30 seconds on error
        # time.sleep(30)
        exit()

print(f"Created Collection: {collection_name}",flush=True)
if run_mode == "local" or not rebalance_topology:
    time.sleep(initial_sleep_seconds)
    info = client.get_collection(collection_name)
    print("*********************** Initial Collection Info ***********************", flush=True)
    print(info, flush=True)
    print("***********************************************************************", flush=True)
    time.sleep(ready_sleep_seconds)
    if not rebalance_topology:
        Path("ready.flag").touch()
    exit()

print("Target topo: ", topology, flush=True)



time.sleep(10)
# id_key_map = map_shard_ids_to_keys(nodes,collection_name)
# with open("id_key_map.json", 'w') as f:
#     json.dump(id_key_map, f, indent=4)

print("Current cluster status:", flush=True)
printShardStatus(nodes, collection_name)

# exit()
# Generate mapping between ip:port and peer-ID
peer_ids = {}
reverse_peer_ids = {}
for addr in topology:
    host, port = addr.split(":")
    port = int(port)
    client = QdrantClient(
        host=host,
        port=port,
        grpc_port=port  + 1,
        prefer_grpc=True,
        timeout=600,
        grpc_options={"grpc.enable_http_proxy": 0},
    )
    peer_id = client.http.distributed_api.cluster_status().result.peer_id

    peer_ids[addr] = peer_id
    reverse_peer_ids[peer_id] = addr



# Determine where each shard is located 
shard_locations = defaultdict(set) 
for addr in topology:
    host, port = addr.split(":")
    port = int(port)
    client = QdrantClient(
        host=host,
        port=port,
        grpc_port=port  + 1,
        prefer_grpc=True,
        timeout=600,
        grpc_options={"grpc.enable_http_proxy": 0},
    )
    cluster_info = client.http.distributed_api.collection_cluster_info(collection_name).result

    for shard in cluster_info.local_shards:
        shard_locations[shard.shard_id].add(peer_ids[addr])


# Replicate shards to desired location
for addr, desired_shards in topology.items():
    host, port = addr.split(":")
    port = int(port)
    client = QdrantClient(
        host=host,
        port=port,
        grpc_port=port  + 1,
        prefer_grpc=True,
        timeout=600,
        grpc_options={"grpc.enable_http_proxy": 0},
    )
    this_peer = peer_ids[addr]
    
    for shard_id in desired_shards:
        if this_peer not in shard_locations.get(shard_id, set()):
            # get peer with target shard 
            
            from_peer_id = next(iter(shard_locations[shard_id]))
            
            # print(f"[REPLICATE] Shard {shard_id} → {addr} from -> {reverse_peer_ids[from_peer_id]}")
            
            # Create replica operation and execute
            operation = ReplicateShardOperation(
                replicate_shard=ReplicateShard(
                    shard_id=int(shard_id),
                    from_peer_id=int(from_peer_id),
                    to_peer_id=int(this_peer)
                )
            )

            client.http.distributed_api.update_collection_cluster(
                collection_name=collection_name,
                cluster_operations=operation
            )

            wait_for_shard_transfer(client, collection_name, int(shard_id))
            shard_locations[shard_id].add(this_peer)

print("\nStatus after replication stage: ",flush=True)
printShardStatus(nodes, collection_name)

for addr, desired_shards in topology.items():
    host, port = addr.split(":")
    port = int(port)
    client = QdrantClient(
        host=host,
        port=port,
        grpc_port=port  + 1,
        prefer_grpc=True,
        timeout=600,
        grpc_options={"grpc.enable_http_proxy": 0},
    )
    this_peer = peer_ids[addr]

    # Get currently hosted shards
    cluster_info = client.http.distributed_api.collection_cluster_info(collection_name).result
    local_shard_ids = {shard.shard_id for shard in cluster_info.local_shards}

    for shard_id in local_shard_ids:
        if shard_id not in desired_shards:
            # print(f"[REMOVE] Shard {shard_id} from {addr}")
           
            try:
                # Create drop replica operation and execute
                drop_op = DropReplicaOperation(
                    drop_replica=Replica(
                        shard_id=shard_id,
                        peer_id=this_peer
                    )
                )

                client.http.distributed_api.update_collection_cluster(
                    collection_name=collection_name,
                    cluster_operations=drop_op
                )
            except Exception as e:
                print(f"⚠️ Failed to remove shard {shard_id} from {addr}: {e}", flush=True)

print("\nStatus after removal (final) stage:", flush=True)
printShardStatus(nodes, collection_name)

Path("ready.flag").touch()
