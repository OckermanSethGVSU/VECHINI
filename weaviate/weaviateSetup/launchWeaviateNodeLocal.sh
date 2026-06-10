#!/usr/bin/env bash
set -euo pipefail

STORAGE_MEDIUM=${1:?Usage: $0 <storage_medium> <total>}
TOTAL=${2:-1}
RANK=0
NODE_NAME="local-node${RANK}"

mkdir -p ./runtime_state

if command -v docker >/dev/null 2>&1; then
    CONTAINER_RUNTIME="docker"
elif command -v podman >/dev/null 2>&1; then
    CONTAINER_RUNTIME="podman"
else
    echo "Neither docker nor podman is installed." >&2
    exit 1
fi

WEAVIATE_LOCAL_NAME="${WEAVIATE_LOCAL_NAME:-weaviate-local}"
WEAVIATE_LOCAL_IMAGE="${WEAVIATE_LOCAL_IMAGE:-semitechnologies/weaviate:1.36.0}"
WEAVIATE_LOCAL_HOST="${WEAVIATE_LOCAL_HOST:-127.0.0.1}"
HTTP_PORT="${WEAVIATE_LOCAL_HTTP_PORT:-8080}"
GO_PROFILING_PORT="${WEAVIATE_LOCAL_PROFILING_PORT:-$((HTTP_PORT + 1))}"
GRPC_PORT="${WEAVIATE_LOCAL_GRPC_PORT:-50051}"
CLUSTER_GOSSIP_BIND_PORT="${WEAVIATE_LOCAL_GOSSIP_PORT:-7100}"
CLUSTER_DATA_BIND_PORT="${WEAVIATE_LOCAL_DATA_PORT:-7101}"
RAFT_PORT="${WEAVIATE_LOCAL_RAFT_PORT:-7102}"
RAFT_INTERNAL_RPC_PORT="${WEAVIATE_LOCAL_RAFT_INTERNAL_PORT:-7103}"
DATA_DIR="${WEAVIATE_LOCAL_DATA_DIR:-$PWD/.local/weaviate/data}"
MODULES_DIR="${WEAVIATE_LOCAL_MODULES_DIR:-$PWD/.local/weaviate/modules}"

mkdir -p "$DATA_DIR" "$MODULES_DIR"

if "$CONTAINER_RUNTIME" ps --format '{{.Names}}' | grep -Fxq "$WEAVIATE_LOCAL_NAME"; then
    echo "Stopping existing Weaviate container '$WEAVIATE_LOCAL_NAME'..."
    "$CONTAINER_RUNTIME" stop "$WEAVIATE_LOCAL_NAME" >/dev/null
fi

if "$CONTAINER_RUNTIME" ps -a --format '{{.Names}}' | grep -Fxq "$WEAVIATE_LOCAL_NAME"; then
    echo "Removing existing Weaviate container '$WEAVIATE_LOCAL_NAME' to refresh env..."
    "$CONTAINER_RUNTIME" rm "$WEAVIATE_LOCAL_NAME" >/dev/null
fi

echo "Launching Weaviate container '$WEAVIATE_LOCAL_NAME' from '$WEAVIATE_LOCAL_IMAGE'..."
"$CONTAINER_RUNTIME" run -d \
    --name "$WEAVIATE_LOCAL_NAME" \
    -p "${HTTP_PORT}:8080" \
    -p "${GO_PROFILING_PORT}:${GO_PROFILING_PORT}" \
    -p "${GRPC_PORT}:50051" \
    -p "${CLUSTER_GOSSIP_BIND_PORT}:7100" \
    -p "${CLUSTER_DATA_BIND_PORT}:7101" \
    -p "${RAFT_PORT}:7102" \
    -p "${RAFT_INTERNAL_RPC_PORT}:7103" \
    -v "${DATA_DIR}:/var/lib/weaviate" \
    -v "${MODULES_DIR}:/tmp/weaviate" \
    -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
    -e PERSISTENCE_DATA_PATH=/var/lib/weaviate \
    -e ASYNC_INDEXING="${ASYNC_INDEXING:-true}" \
    -e DISABLE_LAZY_LOAD_SHARDS="${DISABLE_LAZY_LOAD_SHARDS}" \
    -e HNSW_STARTUP_WAIT_FOR_VECTOR_CACHE="${HNSW_STARTUP_WAIT_FOR_VECTOR_CACHE}" \
    -e DEFAULT_VECTORIZER_MODULE=none \
    -e ENABLE_MODULES= \
    -e CLUSTER_HOSTNAME="${NODE_NAME}" \
    -e RAFT_BOOTSTRAP_EXPECT=1 \
    -e QUERY_DEFAULTS_LIMIT=20 \
    -e GRPC_PORT=50051 \
    -e GO_PROFILING_PORT="${GO_PROFILING_PORT}" \
    "$WEAVIATE_LOCAL_IMAGE" >/dev/null

cat > ip_registry.txt <<EOF
${RANK},${NODE_NAME},${WEAVIATE_LOCAL_HOST},${HTTP_PORT},${GRPC_PORT},${CLUSTER_GOSSIP_BIND_PORT},${CLUSTER_DATA_BIND_PORT},${RAFT_PORT},${RAFT_INTERNAL_RPC_PORT}
EOF

until NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    curl -sf "http://${WEAVIATE_LOCAL_HOST}:${HTTP_PORT}/v1/.well-known/ready" >/dev/null; do
    sleep 1
done

touch "./runtime_state/weaviate_running${RANK}.txt"
