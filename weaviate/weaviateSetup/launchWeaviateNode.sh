#!/bin/bash

# Usage:
#   ./launchWeaviateNode.sh <storage_medium> <useperf> <total>

STORAGE_MEDIUM=${1:?Usage: $0 <storage_medium> <useperf> <total>}
USEPERF=${2:?Usage: $0 <storage_medium> <useperf> <total>}
TOTAL=${3:-${PMI_SIZE:-${OMPI_COMM_WORLD_SIZE:-${MV2_COMM_WORLD_SIZE:-${PMIX_SIZE:-}}}}}
RANK="${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-${MV2_COMM_WORLD_RANK:-${SLURM_PROCID:-}}}}}"

if [[ -z "${RANK}" ]]; then
    echo "Error: MPI rank not found in environment" >&2
    exit 1
fi

if [[ -z "${TOTAL}" ]]; then
    echo "Error: total ranks not provided and MPI size not found" >&2
    exit 1
fi

RANK=$((RANK))
TOTAL=$((TOTAL))
NODE_NAME="node${RANK}"
mkdir -p ./runtime_state

# ------------------------------------------------------------
# Discover LOCAL IP only
# ------------------------------------------------------------
python3 mapping.py --rank "$RANK"

IP_JSON="interfaces${RANK}.json"
if [[ ! -f "$IP_JSON" ]]; then
    echo "Error: expected IP mapping file '$IP_JSON' not found" >&2
    exit 1
fi

IP_ADDR=$(jq -r '.hsn0.ipv4[0] // empty' "$IP_JSON")
if [[ -z "${IP_ADDR}" ]]; then
    echo "Error: could not read .hsn0.ipv4[0] from '$IP_JSON'" >&2
    cat "$IP_JSON" >&2
    exit 1
fi

# ------------------------------------------------------------
# Per-rank ports
# ------------------------------------------------------------
HTTP_PORT=$((8080 + RANK * 100))
GO_PROFILING_PORT=$((HTTP_PORT + 1))
GRPC_PORT=$((HTTP_PORT + 2))
CLUSTER_GOSSIP_BIND_PORT=$((HTTP_PORT + 3))
CLUSTER_DATA_BIND_PORT=$((HTTP_PORT + 4))
RAFT_PORT=$((HTTP_PORT + 5))
RAFT_INTERNAL_RPC_PORT=$((HTTP_PORT + 6))

RAFT_JOIN=$(
    for r in $(seq 0 $((TOTAL - 1))); do
        printf "node%d:%d," "$r" "$((8085 + r * 100))"
    done
)
RAFT_JOIN=${RAFT_JOIN%,}

# ------------------------------------------------------------
# Shared registry
# ------------------------------------------------------------
OUTPUT_FILE="ip_registry.txt"
touch "$OUTPUT_FILE"

if [[ $RANK -eq 0 ]]; then
    : > "$OUTPUT_FILE"
fi

sleep 1


# rank,node,ip,http,grpc,gossip,data,raft,raft_internal
echo "${RANK},${NODE_NAME},${IP_ADDR},${HTTP_PORT},${GRPC_PORT},${CLUSTER_GOSSIP_BIND_PORT},${CLUSTER_DATA_BIND_PORT},${RAFT_PORT},${RAFT_INTERNAL_RPC_PORT}" >> "$OUTPUT_FILE"

while true; do
    REG_COUNT=$(awk -F, 'NF >= 9 {seen[$1]=1} END {print length(seen)}' "$OUTPUT_FILE")
    if [[ "${REG_COUNT}" -ge "${TOTAL}" ]]; then
        break
    fi
    sleep 0.1
done

BOOTSTRAP_NODE_NAME=$(awk -F, '$1 == 0 {print $2; exit}' "$OUTPUT_FILE")
BOOTSTRAP_IP=$(awk -F, '$1 == 0 {print $3; exit}' "$OUTPUT_FILE")
BOOTSTRAP_HTTP_PORT=$(awk -F, '$1 == 0 {print $4; exit}' "$OUTPUT_FILE")
BOOTSTRAP_GRPC_PORT=$(awk -F, '$1 == 0 {print $5; exit}' "$OUTPUT_FILE")
BOOTSTRAP_GOSSIP_PORT=$(awk -F, '$1 == 0 {print $6; exit}' "$OUTPUT_FILE")
BOOTSTRAP_DATA_PORT=$(awk -F, '$1 == 0 {print $7; exit}' "$OUTPUT_FILE")
BOOTSTRAP_RAFT_PORT=$(awk -F, '$1 == 0 {print $8; exit}' "$OUTPUT_FILE")

if [[ -z "${BOOTSTRAP_NODE_NAME}" || -z "${BOOTSTRAP_IP}" || -z "${BOOTSTRAP_GOSSIP_PORT}" || -z "${BOOTSTRAP_RAFT_PORT}" ]]; then
    echo "Error: failed to read rank 0 bootstrap info from '$OUTPUT_FILE'" >&2
    cat "$OUTPUT_FILE" >&2
    exit 1
fi

# ------------------------------------------------------------
# Storage selection
# ------------------------------------------------------------
APPTAINER_ARGS=()

if [[ "$STORAGE_MEDIUM" == "memory" ]]; then
    TARGET_BASE="/dev/shm/WeaviateDir"
    (( RANK == 0 )) && echo "Using memory for persistence"

elif [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
    DAOS_POOL="radix-io"
    DAOS_CONT="vectorDBTesting"
    TARGET_BASE="/tmp/${DAOS_POOL}/${DAOS_CONT}/${myDIR:-default}/WeaviateDir"
    (( RANK == 0 )) && echo "Using DAOS for persistence at ${TARGET_BASE}"
    APPTAINER_ARGS+=(
        --bind "/home/treewalker/daos_lib64:/opt/daos/lib64:ro"
        --env LD_LIBRARY_PATH=/opt/daos/lib64
    )

elif [[ "$STORAGE_MEDIUM" == "lustre" ]]; then
    TARGET_BASE="./WeaviateDir"
    (( RANK == 0 )) && echo "Using lustre for persistence"

elif [[ "$STORAGE_MEDIUM" == "SSD" ]]; then
    TARGET_BASE="/local/scratch/WeaviateDir"
    (( RANK == 0 )) && echo "Using SSD for persistence"

else
    echo "Error: unknown STORAGE_MEDIUM '$STORAGE_MEDIUM'" >&2
    exit 1
fi

rm -rf "${TARGET_BASE}/node${RANK}"
mkdir -p "${TARGET_BASE}/node${RANK}"

WEAVIATE_LOCAL_SIF="${WEAVIATE_LOCAL_SIF:-./weaviate.sif}"
WEAVIATE_IMAGE_URI="${WEAVIATE_IMAGE_URI:-docker://semitechnologies/weaviate:1.36.0}"

if [[ -f "$WEAVIATE_LOCAL_SIF" ]]; then
    IMAGE="$WEAVIATE_LOCAL_SIF"
else
    IMAGE="$WEAVIATE_IMAGE_URI"
fi

COMMON_ARGS=(
    --fakeroot
    --writable-tmpfs
    -B "${TARGET_BASE}/node${RANK}:/var/lib/weaviate"

    --env AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
    --env PERSISTENCE_DATA_PATH=/var/lib/weaviate
    --env ASYNC_INDEXING="${ASYNC_INDEXING}"
    --env DISABLE_LAZY_LOAD_SHARDS="${DISABLE_LAZY_LOAD_SHARDS}"
    --env HNSW_STARTUP_WAIT_FOR_VECTOR_CACHE="${HNSW_STARTUP_WAIT_FOR_VECTOR_CACHE}"

    --env GO_PROFILING_PORT="${GO_PROFILING_PORT}"
    --env GRPC_PORT="${GRPC_PORT}"

    # This mode is required whenever multiple Weaviate nodes share a host.
    # It makes Weaviate honor the explicit name:raftPort mapping from RAFT_JOIN.
    --env CLUSTER_IN_LOCALHOST=true
    --env CLUSTER_HOSTNAME="${NODE_NAME}"
    --env CLUSTER_ADVERTISE_ADDR="${IP_ADDR}"
    --env CLUSTER_GOSSIP_BIND_PORT="${CLUSTER_GOSSIP_BIND_PORT}"
    --env CLUSTER_DATA_BIND_PORT="${CLUSTER_DATA_BIND_PORT}"

    --env RAFT_PORT="${RAFT_PORT}"
    --env RAFT_INTERNAL_RPC_PORT="${RAFT_INTERNAL_RPC_PORT}"
    --env RAFT_JOIN="${RAFT_JOIN}"
    --env RAFT_BOOTSTRAP_TIMEOUT=180
    --env NO_PROXY="" 
    --env no_proxy="" 
    --env http_proxy="" 
    --env https_proxy="" 
    --env HTTP_PROXY="" 
    --env HTTPS_PROXY=""
)

if [[ -n "${CORES:-}" ]]; then
    COMMON_ARGS+=(
        --env GOMAXPROCS="${CORES}"
    )
fi

if [[ -n "${GRPC_MAX_MESSAGE_SIZE:-}" ]]; then
    COMMON_ARGS+=(
        --env GRPC_MAX_MESSAGE_SIZE="${GRPC_MAX_MESSAGE_SIZE}"
    )
fi

WEAVIATE_CMD=(
    weaviate
    --host ${IP_ADDR}
    --port "${HTTP_PORT}"
    --scheme http
    --raft-port "${RAFT_PORT}"
    --raft-internal-rpc-port "${RAFT_INTERNAL_RPC_PORT}"
)

READY_TIMEOUT_SEC="${WEAVIATE_READY_TIMEOUT_SEC:-30}"
LAUNCH_RETRIES="${WEAVIATE_LAUNCH_RETRIES:-3}"
RETRY_BACKOFF_SEC="${WEAVIATE_RETRY_BACKOFF_SEC:-5}"

wait_for_ready() {
    local host="$1"
    local port="$2"
    local timeout_sec="$3"
    local waited=0

    until NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        curl -sf "http://${host}:${port}/v1/.well-known/ready" >/dev/null; do
        if (( waited >= timeout_sec )); then
            return 1
        fi
        sleep 1
        ((waited += 1))
    done

    return 0
}

cleanup_failed_launch() {
    local pid="$1"

    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    fi

    rm -rf "${TARGET_BASE}/node${RANK}"
    mkdir -p "${TARGET_BASE}/node${RANK}"
}

sleep_for_follower_stagger() {
    if [[ $RANK -le 0 ]]; then
        return 0
    fi

    local stagger_sec="0.5"
    if [[ $RANK -gt 1 ]]; then
        stagger_sec="$(awk -v rank="$RANK" 'BEGIN { printf "%.1f", rank / 2 }')"
    fi
    echo "Follower rank ${RANK} staggering launch by ${stagger_sec}s before joining"
    sleep "$stagger_sec"
}

launch_weaviate_with_retries() {
    local mode="$1"
    local attempt

    for attempt in $(seq 1 "${LAUNCH_RETRIES}"); do
        if [[ "$mode" == "bootstrap" ]]; then
            apptainer exec \
                "${COMMON_ARGS[@]}" \
                "${APPTAINER_ARGS[@]}" \
                --env RAFT_BOOTSTRAP_EXPECT=1 \
                "${IMAGE}" \
                "${WEAVIATE_CMD[@]}" \
                > "rank${RANK}.out" 2>&1 &
        else
            apptainer exec \
                "${COMMON_ARGS[@]}" \
                "${APPTAINER_ARGS[@]}" \
                --env CLUSTER_JOIN="${BOOTSTRAP_IP}:${BOOTSTRAP_GOSSIP_PORT}" \
                "${IMAGE}" \
                "${WEAVIATE_CMD[@]}" \
                > "rank${RANK}.out" 2>&1 &
        fi
        LAUNCH_PID=$!

        if wait_for_ready "${IP_ADDR}" "${HTTP_PORT}" "${READY_TIMEOUT_SEC}"; then
            echo "Rank ${RANK} became ready on attempt ${attempt}/${LAUNCH_RETRIES}" >&2
            return 0
        fi

        echo "Rank ${RANK} failed to become ready within ${READY_TIMEOUT_SEC}s on attempt ${attempt}/${LAUNCH_RETRIES}" >&2
        cleanup_failed_launch "$LAUNCH_PID"

        if (( attempt < LAUNCH_RETRIES )); then
            sleep "${RETRY_BACKOFF_SEC}"
        fi
    done

    echo "Rank ${RANK} exhausted ${LAUNCH_RETRIES} launch attempts without reaching readiness" >&2
    return 1
}

if [[ $RANK -eq 0 ]]; then
    echo "Launching founding node ${NODE_NAME}"

    launch_weaviate_with_retries bootstrap || exit 1
else
    until bash -c "exec 3<>/dev/tcp/${BOOTSTRAP_IP}/${BOOTSTRAP_GOSSIP_PORT}" 2>/dev/null; do
        sleep 0.2
    done

    until bash -c "exec 3<>/dev/tcp/${BOOTSTRAP_IP}/${BOOTSTRAP_RAFT_PORT}" 2>/dev/null; do
        sleep 0.2
    done

    sleep_for_follower_stagger

    echo "Launching follower node ${NODE_NAME} via ${BOOTSTRAP_NODE_NAME}@${BOOTSTRAP_IP}:${BOOTSTRAP_GOSSIP_PORT}"

    launch_weaviate_with_retries follower || exit 1
fi

touch "./runtime_state/weaviate_running${RANK}.txt"

wait "${LAUNCH_PID}"
