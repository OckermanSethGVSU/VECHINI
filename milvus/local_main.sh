#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$(pwd)}"

if [[ -f "$RUN_DIR/run_config.env" ]]; then
    set -a
    source "$RUN_DIR/run_config.env"
    set +a
fi

export_default_var() {
    local name="$1"
    local default_value="$2"

    printf -v "$name" '%s' "${!name:-$default_value}"
    export "$name"
}

PYTHON_ENV_VARS=(
    NO_PROXY=""
    no_proxy=""
    http_proxy=""
    https_proxy=""
    HTTP_PROXY=""
    HTTPS_PROXY=""
)

BASE_DIR="${BASE_DIR:-$(dirname "$RUN_DIR")}"
export_default_var myDIR "$(basename "$RUN_DIR")"
export_default_var RESULT_PATH "$RUN_DIR"
export_default_var RUNTIME_STATE_DIR "$RUN_DIR/runtime_state"
export_default_var MILVUS_HOST "127.0.0.1"

if [[ -z "${MINIO_MODE:-}" ]]; then
    if [[ "${MODE^^}" == "DISTRIBUTED" ]]; then
        export MINIO_MODE="single"
    else
        export MINIO_MODE="off"
    fi
fi

if [[ "${MODE^^}" == "DISTRIBUTED" ]]; then
    DEFAULT_MILVUS_GRPC_PORT="20001"
    DEFAULT_MILVUS_HEALTH_PORT="30001"
else
    DEFAULT_MILVUS_GRPC_PORT="19530"
    DEFAULT_MILVUS_HEALTH_PORT="9091"
fi

export MILVUS_GRPC_PORT="${MILVUS_GRPC_PORT:-$DEFAULT_MILVUS_GRPC_PORT}"
export MILVUS_HEALTH_PORT="${MILVUS_HEALTH_PORT:-$DEFAULT_MILVUS_HEALTH_PORT}"

export_default_var CONTAINER_NAME "${MILVUS_LOCAL_NAME:-milvus-standalone}"
export_default_var IMAGE "${MILVUS_LOCAL_IMAGE:-milvusdb/milvus:v2.6.12}"
export_default_var ETCD_IMAGE "${MILVUS_ETCD_IMAGE:-quay.io/coreos/etcd:v3.5.18}"
export_default_var ETCD_PORT "${MILVUS_ETCD_PORT:-2379}"
export_default_var VOLUMES_DIR "${MILVUS_LOCAL_VOLUME_DIR:-$RUN_DIR/volumes/milvus}"
export_default_var MINIO_CONTAINER_NAME "${MINIO_LOCAL_NAME:-milvus-minio}"
export_default_var MINIO_IMAGE "${MINIO_LOCAL_IMAGE:-minio/minio:RELEASE.2025-02-28T09-55-16Z}"
export_default_var MINIO_API_PORT "${MINIO_API_PORT:-9000}"
export_default_var MINIO_CONSOLE_PORT "${MINIO_CONSOLE_PORT:-9001}"
export_default_var MINIO_HOST "${MINIO_HOST:-127.0.0.1}"
export_default_var MINIO_INTERNAL_HOST "${MINIO_INTERNAL_HOST:-$MINIO_CONTAINER_NAME}"
export_default_var MINIO_BUCKET_NAME "${MINIO_BUCKET_NAME:-a-bucket}"
export_default_var MINIO_ACCESS_KEY_ID "${MINIO_ACCESS_KEY_ID:-minioadmin}"
export_default_var MINIO_SECRET_ACCESS_KEY "${MINIO_SECRET_ACCESS_KEY:-minioadmin}"
export_default_var MINIO_NETWORK_NAME "${MINIO_NETWORK_NAME:-milvus-local-net}"
export_default_var MINIO_VOLUMES_DIR "${MINIO_LOCAL_VOLUME_DIR:-$RUN_DIR/volumes/minio}"
export_default_var LOCAL_CLUSTER_PREFIX "${MILVUS_LOCAL_CLUSTER_PREFIX:-milvus-local}"
export_default_var CONFIG_DIR "$RUN_DIR/configs"
export_default_var LOCAL_SHARED_STORAGE_PATH "${LOCAL_SHARED_STORAGE_PATH:-$RUN_DIR/volumes/localfs/shared}"
export_default_var EMBED_ETCD_FILE "$RUN_DIR/embedEtcd.yaml"
export_default_var USER_CONFIG_FILE "$RUN_DIR/user.yaml"
export_default_var STANDARD_BINARY_PATH "${STANDARD_BINARY_PATH:-}"
export_default_var MIXED_BINARY_PATH "${MIXED_BINARY_PATH:-}"

mkdir -p "$RUN_DIR" "$RUNTIME_STATE_DIR" "$VOLUMES_DIR" "$MINIO_VOLUMES_DIR" "$CONFIG_DIR" "$LOCAL_SHARED_STORAGE_PATH"

distributed_registry_path() {
    local component="$1"

    case "$component" in
        etcd)
            printf '%s\n' "$RUN_DIR/etcdFiles/etcd_registry.txt"
            ;;
        minio)
            printf '%s\n' "$RUN_DIR/minioFiles/minio_registry.txt"
            ;;
        COORDINATOR|STREAMING|QUERY|DATA|PROXY)
            printf '%s\n' "$RUN_DIR/$component/${component}_registry.txt"
            ;;
        *)
            echo "Unknown distributed registry component: $component" >&2
            return 1
            ;;
    esac
}

if [[ "${MODE^^}" == "DISTRIBUTED" ]]; then
    export PROXY_REGISTRY_PATH="${PROXY_REGISTRY_PATH:-$(distributed_registry_path PROXY)}"
else
    export PROXY_REGISTRY_PATH="${PROXY_REGISTRY_PATH:-$RUNTIME_STATE_DIR/PROXY_registry.txt}"
fi

pick_binary() {
    local override="$1"
    shift

    if [[ -n "$override" ]]; then
        printf '%s\n' "$override"
        return 0
    fi

    local candidate
    for candidate in "$@"; do
        if [[ -x "$candidate" ]]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done

    printf '%s\n' "$1"
}

STANDARD_BINARY_PATH="$(pick_binary \
    "$STANDARD_BINARY_PATH" \
    "$ROOT_DIR/clients/batch_client/batch_client" \
    "$ROOT_DIR/batch_client")"

MIXED_BINARY_PATH="$(pick_binary \
    "$MIXED_BINARY_PATH" \
    "$ROOT_DIR/clients/mixed/mixed" \
    "$ROOT_DIR/mixed")"

resolve_mixed_insert_start_id() {
    if [[ "$TASK" != "MIXED" || -n "${INSERT_START_ID:-}" ]]; then
        return 0
    fi

    if [[ -n "${RESTORE_DIR:-}" ]]; then
        export INSERT_START_ID="${EXPECTED_CORPUS_SIZE:?EXPECTED_CORPUS_SIZE is required when RESTORE_DIR is set}"
    elif [[ -n "${INSERT_CORPUS_SIZE:-}" ]]; then
        export INSERT_START_ID="$INSERT_CORPUS_SIZE"
    elif [[ -n "${INSERT_DATA_FILEPATH:-}" ]]; then
        if ! export INSERT_START_ID="$(env "${PYTHON_ENV_VARS[@]}" python3 ./npy_inspect.py "$INSERT_DATA_FILEPATH")"; then
            echo "Error: failed to derive INSERT_START_ID from INSERT_DATA_FILEPATH using npy_inspect.py." >&2
            exit 1
        fi
    else
        echo "Error: TASK=MIXED requires INSERT_START_ID, INSERT_CORPUS_SIZE, RESTORE_DIR, or INSERT_DATA_FILEPATH." >&2
        exit 1
    fi
}

ensure_runtime_tools() {
    if command -v docker >/dev/null 2>&1; then
        CONTAINER_RUNTIME="docker"
    elif command -v podman >/dev/null 2>&1; then
        CONTAINER_RUNTIME="podman"
    else
        echo "Neither docker nor podman is installed." >&2
        exit 1
    fi

    command -v curl >/dev/null 2>&1 || { echo "curl is required." >&2; exit 1; }
    command -v python3 >/dev/null 2>&1 || { echo "python3 is required." >&2; exit 1; }

    if [[ "$TASK" == "MIXED" ]]; then
        if [[ ! -x "$MIXED_BINARY_PATH" ]]; then
            echo "Missing mixed binary at $MIXED_BINARY_PATH" >&2
            exit 1
        fi
    else
        if [[ ! -x "$STANDARD_BINARY_PATH" ]]; then
            echo "Missing batch_client binary at $STANDARD_BINARY_PATH" >&2
            exit 1
        fi
    fi
}

write_local_configs() {
    if [[ -d "$EMBED_ETCD_FILE" ]]; then
        rm -rf "$EMBED_ETCD_FILE"
    fi
    if [[ -d "$USER_CONFIG_FILE" ]]; then
        rm -rf "$USER_CONFIG_FILE"
    fi

    cat > "$EMBED_ETCD_FILE" <<'EOF'
listen-client-urls: http://0.0.0.0:2379
advertise-client-urls: http://0.0.0.0:2379
quota-backend-bytes: 4294967296
auto-compaction-mode: revision
auto-compaction-retention: '1000'
EOF

    cat > "$USER_CONFIG_FILE" <<'EOF'
# Extra config to override default milvus.yaml
EOF
}

write_registry_files() {
    printf '%s\n' "$MILVUS_HOST" > "$RUNTIME_STATE_DIR/worker.ip"
    printf '0,%s,%s,%s\n' "$MILVUS_HOST" "$MILVUS_GRPC_PORT" "$MILVUS_HEALTH_PORT" > "$RUNTIME_STATE_DIR/PROXY_registry.txt"

    if [[ "$MINIO_MODE" == "single" ]]; then
        printf '0,%s,%s\n' "$MINIO_HOST" "$MINIO_API_PORT" > "$RUNTIME_STATE_DIR/minio_registry.txt"
    else
        rm -f "$RUNTIME_STATE_DIR/minio_registry.txt"
    fi
}

wait_for_milvus() {
    echo "Waiting for Milvus at ${MILVUS_HOST}:${MILVUS_HEALTH_PORT}..."
    for _ in {1..180}; do
        if env "${PYTHON_ENV_VARS[@]}" curl -fsS "http://${MILVUS_HOST}:${MILVUS_HEALTH_PORT}/healthz" >/dev/null 2>&1; then
            echo "Milvus is ready."
            return 0
        fi
        sleep 1
    done

    echo "Milvus did not become healthy within 180 seconds." >&2
    echo "Inspect logs with: ${CONTAINER_RUNTIME} logs ${CONTAINER_NAME}" >&2
    exit 1
}

ensure_container_network() {
    if ! "$CONTAINER_RUNTIME" network inspect "$MINIO_NETWORK_NAME" >/dev/null 2>&1; then
        "$CONTAINER_RUNTIME" network create "$MINIO_NETWORK_NAME" >/dev/null
    fi
}

container_exists() {
    local name="$1"
    "$CONTAINER_RUNTIME" ps -a --format '{{.Names}}' | grep -Fxq "$name"
}

container_running() {
    local name="$1"
    "$CONTAINER_RUNTIME" ps --format '{{.Names}}' | grep -Fxq "$name"
}

container_mount_source() {
    local name="$1"
    local destination="$2"

    "$CONTAINER_RUNTIME" inspect "$name" \
        --format "{{range .Mounts}}{{if eq .Destination \"$destination\"}}{{println .Source}}{{end}}{{end}}" |
        head -n 1
}

remove_container_if_present() {
    local name="$1"

    if container_running "$name"; then
        "$CONTAINER_RUNTIME" stop "$name" >/dev/null
    fi

    if container_exists "$name"; then
        "$CONTAINER_RUNTIME" rm "$name" >/dev/null
    fi
}

wait_for_minio() {
    echo "Waiting for MinIO at ${MINIO_HOST}:${MINIO_API_PORT}..."
    for _ in {1..120}; do
        if env "${PYTHON_ENV_VARS[@]}" curl -fsS "http://${MINIO_HOST}:${MINIO_API_PORT}/minio/health/live" >/dev/null 2>&1; then
            echo "MinIO is ready."
            return 0
        fi
        sleep 1
    done

    echo "MinIO did not become healthy within 120 seconds." >&2
    echo "Inspect logs with: ${CONTAINER_RUNTIME} logs ${MINIO_CONTAINER_NAME}" >&2
    exit 1
}

wait_for_http_ok() {
    local url="$1"
    local label="$2"
    local attempts="${3:-180}"

    echo "Waiting for ${label} at ${url}..."
    for ((attempt=1; attempt<=attempts; attempt++)); do
        if env "${PYTHON_ENV_VARS[@]}" curl -fsS "$url" >/dev/null 2>&1; then
            echo "${label} is ready."
            return 0
        fi
        sleep 1
    done

    echo "${label} did not become ready in time: ${url}" >&2
    exit 1
}

distributed_etcd_instances() {
    if [[ "${ETCD_MODE,,}" == "replicated" ]]; then
        printf '3\n'
    elif [[ "${ETCD_MODE,,}" == "single" ]]; then
        printf '1\n'
    else
        echo "Local distributed mode requires ETCD_MODE='single' or 'replicated' (got '${ETCD_MODE}')." >&2
        exit 1
    fi
}

distributed_minio_instances() {
    if [[ "${MINIO_MODE,,}" == "off" ]]; then
        printf '0\n'
    elif [[ "${MINIO_MODE,,}" == "stripped" ]]; then
        printf '4\n'
    elif [[ "${MINIO_MODE,,}" == "single" ]]; then
        printf '1\n'
    else
        echo "Local distributed mode requires MINIO_MODE='off', 'single', or 'stripped' (got '${MINIO_MODE}')." >&2
        exit 1
    fi
}

role_service_base_port() {
    case "$1" in
        COORDINATOR) printf '20000\n' ;;
        PROXY) printf '20001\n' ;;
        QUERY) printf '20004\n' ;;
        DATA) printf '20006\n' ;;
        STREAMING) printf '20007\n' ;;
        *)
            echo "Unknown distributed role '$1'." >&2
            exit 1
            ;;
    esac
}

role_metrics_base_port() {
    case "$1" in
        COORDINATOR) printf '30000\n' ;;
        PROXY) printf '30001\n' ;;
        QUERY) printf '30004\n' ;;
        DATA) printf '30006\n' ;;
        STREAMING) printf '30007\n' ;;
        *)
            echo "Unknown distributed role '$1'." >&2
            exit 1
            ;;
    esac
}

role_service_port() {
    local role="$1"
    local rank="$2"
    printf '%s\n' $(( $(role_service_base_port "$role") + (8 * rank) ))
}

role_metrics_port() {
    local role="$1"
    local rank="$2"
    printf '%s\n' $(( $(role_metrics_base_port "$role") + (8 * rank) ))
}

component_container_name() {
    local kind="$1"
    local rank="$2"
    printf '%s-%s-%s\n' "$LOCAL_CLUSTER_PREFIX" "${kind,,}" "$rank"
}

prepare_distributed_support_files() {
    local replace_source=""
    local config_source=""

    if [[ -f "$RUN_DIR/replace_unified.py" ]]; then
        replace_source="$RUN_DIR/replace_unified.py"
    elif [[ -f "$ROOT_DIR/scripts/replace_unified.py" ]]; then
        replace_source="$ROOT_DIR/scripts/replace_unified.py"
    fi

    if [[ -z "$replace_source" ]]; then
        echo "Distributed local mode requires replace_unified.py in the run directory or repo checkout." >&2
        exit 1
    fi

    if [[ "$replace_source" != "$RUN_DIR/replace_unified.py" ]]; then
        cp "$replace_source" "$RUN_DIR/replace_unified.py"
    fi

    if [[ -f "$CONFIG_DIR/unified_milvus.yaml" ]]; then
        return 0
    fi

    if [[ -f "$ROOT_DIR/runtime/configs/unified_milvus.yaml" ]]; then
        config_source="$ROOT_DIR/runtime/configs/unified_milvus.yaml"
    elif [[ -f "$RUN_DIR/unified_milvus.yaml" ]]; then
        config_source="$RUN_DIR/unified_milvus.yaml"
    fi

    if [[ -z "$config_source" ]]; then
        echo "Distributed local mode requires configs/unified_milvus.yaml in the run directory or repo checkout." >&2
        exit 1
    fi

    cp "$config_source" "$CONFIG_DIR/unified_milvus.yaml"
}

write_distributed_registry_files() {
    local etcd_instances
    local minio_instances
    local role role_count rank service_port metrics_port
    local etcd_registry minio_registry role_registry

    etcd_instances="$(distributed_etcd_instances)"
    minio_instances="$(distributed_minio_instances)"
    etcd_registry="$(distributed_registry_path etcd)"
    minio_registry="$(distributed_registry_path minio)"

    printf '%s\n' "$MILVUS_HOST" > "$RUNTIME_STATE_DIR/worker.ip"
    mkdir -p "$RUN_DIR/etcdFiles" "$RUN_DIR/minioFiles" \
        "$RUN_DIR/COORDINATOR" "$RUN_DIR/STREAMING" "$RUN_DIR/QUERY" "$RUN_DIR/DATA" "$RUN_DIR/PROXY"
    : > "$etcd_registry"
    : > "$minio_registry"
    : > "$(distributed_registry_path COORDINATOR)"
    : > "$(distributed_registry_path STREAMING)"
    : > "$(distributed_registry_path QUERY)"
    : > "$(distributed_registry_path DATA)"
    : > "$(distributed_registry_path PROXY)"

    for ((rank=0; rank<etcd_instances; rank++)); do
        printf '%s,%s,%s,%s\n' \
            "$rank" \
            "$MILVUS_HOST" \
            "$((2379 + (100 * rank)))" \
            "$((2380 + (100 * rank)))" >> "$etcd_registry"
    done

    for ((rank=0; rank<minio_instances; rank++)); do
        printf '%s,%s,%s\n' \
            "$rank" \
            "$MILVUS_HOST" \
            "$((9000 + (100 * rank)))" >> "$minio_registry"
    done

    for role in COORDINATOR STREAMING QUERY DATA PROXY; do
        role_registry="$(distributed_registry_path "$role")"
        case "$role" in
            COORDINATOR) role_count="$COORDINATOR_NODES" ;;
            STREAMING) role_count="$STREAMING_NODES" ;;
            QUERY) role_count="$QUERY_NODES" ;;
            DATA) role_count="$DATA_NODES" ;;
            PROXY) role_count="$NUM_PROXIES" ;;
        esac

        for ((rank=0; rank<role_count; rank++)); do
            service_port="$(role_service_port "$role" "$rank")"
            metrics_port="$(role_metrics_port "$role" "$rank")"
            printf '%s,%s,%s,%s\n' \
                "$rank" "$MILVUS_HOST" "$service_port" "$metrics_port" >> "$role_registry"
        done
    done
}

generate_distributed_configs() {
    local role role_count rank

    prepare_distributed_support_files
    write_distributed_registry_files

    (
        cd "$RUN_DIR"
        env LOCAL_SHARED_STORAGE_PATH="$LOCAL_SHARED_STORAGE_PATH" "${PYTHON_ENV_VARS[@]}" python3 ./replace_unified.py --mode distributed
        for role in COORDINATOR STREAMING QUERY DATA PROXY; do
            case "$role" in
                COORDINATOR) role_count="$COORDINATOR_NODES" ;;
                STREAMING) role_count="$STREAMING_NODES" ;;
                QUERY) role_count="$QUERY_NODES" ;;
                DATA) role_count="$DATA_NODES" ;;
                PROXY) role_count="$NUM_PROXIES" ;;
            esac

            for ((rank=0; rank<role_count; rank++)); do
                env LOCAL_SHARED_STORAGE_PATH="$LOCAL_SHARED_STORAGE_PATH" "${PYTHON_ENV_VARS[@]}" python3 ./replace_unified.py --mode "$role" --rank "$rank"
            done
        done
    )
}

launch_local_distributed_etcd() {
    local etcd_instances
    local rank client_port peer_port initial_cluster state data_dir name
    local cluster_parts=()

    etcd_instances="$(distributed_etcd_instances)"
    for ((rank=0; rank<etcd_instances; rank++)); do
        cluster_parts+=("etcd-${rank}=http://${MILVUS_HOST}:$((2380 + (100 * rank)))")
    done
    initial_cluster="$(IFS=,; printf '%s' "${cluster_parts[*]}")"
    state="new"
    if [[ -n "$RESTORE_DIR" ]]; then
        state="existing"
    fi

    for ((rank=0; rank<etcd_instances; rank++)); do
        name="$(component_container_name "etcd" "$rank")"
        client_port="$((2379 + (100 * rank)))"
        peer_port="$((2380 + (100 * rank)))"
        data_dir="$RUN_DIR/volumes/etcd/${rank}"
        mkdir -p "$data_dir"
        remove_container_if_present "$name"
        "$CONTAINER_RUNTIME" run -d \
            --name "$name" \
            --network host \
            -e ETCD_AUTO_COMPACTION_MODE=revision \
            -e ETCD_AUTO_COMPACTION_RETENTION=1000 \
            -e ETCD_QUOTA_BACKEND_BYTES=4294967296 \
            -v "${data_dir}:/etcd" \
            "$ETCD_IMAGE" \
            etcd \
            --name "etcd-${rank}" \
            --advertise-client-urls "http://${MILVUS_HOST}:${client_port}" \
            --listen-client-urls "http://0.0.0.0:${client_port}" \
            --initial-advertise-peer-urls "http://${MILVUS_HOST}:${peer_port}" \
            --listen-peer-urls "http://0.0.0.0:${peer_port}" \
            --initial-cluster "$initial_cluster" \
            --initial-cluster-state "$state" \
            --data-dir /etcd >/dev/null
    done

    for ((rank=0; rank<etcd_instances; rank++)); do
        wait_for_http_ok "http://${MILVUS_HOST}:$((2379 + (100 * rank)))/health" "etcd-${rank}" 120
    done
}

launch_local_distributed_minio() {
    local minio_instances
    local rank name api_port console_port data_dir
    local endpoints=()

    minio_instances="$(distributed_minio_instances)"
    if (( minio_instances == 0 )); then
        return 0
    fi

    if (( minio_instances == 1 )); then
        name="$(component_container_name "minio" "0")"
        api_port="9000"
        console_port="9001"
        data_dir="$RUN_DIR/volumes/minio/0"
        mkdir -p "$data_dir"
        remove_container_if_present "$name"
        "$CONTAINER_RUNTIME" run -d \
            --name "$name" \
            --network host \
            -e MINIO_ROOT_USER="$MINIO_ACCESS_KEY_ID" \
            -e MINIO_ROOT_PASSWORD="$MINIO_SECRET_ACCESS_KEY" \
            -v "${data_dir}:/data0" \
            "$MINIO_IMAGE" \
            minio server /data0 \
            --address "${MILVUS_HOST}:${api_port}" \
            --console-address "${MILVUS_HOST}:${console_port}" >/dev/null
        wait_for_http_ok "http://${MILVUS_HOST}:${api_port}/minio/health/live" "minio-0" 180
        return 0
    fi

    for ((rank=0; rank<minio_instances; rank++)); do
        endpoints+=("http://${MILVUS_HOST}:$((9000 + (100 * rank)))/data${rank}")
    done

    for ((rank=0; rank<minio_instances; rank++)); do
        name="$(component_container_name "minio" "$rank")"
        api_port="$((9000 + (100 * rank)))"
        console_port="$((9001 + (100 * rank)))"
        data_dir="$RUN_DIR/volumes/minio/${rank}"
        mkdir -p "$data_dir"
        remove_container_if_present "$name"
        "$CONTAINER_RUNTIME" run -d \
            --name "$name" \
            --network host \
            -e MINIO_ROOT_USER="$MINIO_ACCESS_KEY_ID" \
            -e MINIO_ROOT_PASSWORD="$MINIO_SECRET_ACCESS_KEY" \
            -v "${data_dir}:/data${rank}" \
            "$MINIO_IMAGE" \
            minio server "${endpoints[@]}" \
            --address "${MILVUS_HOST}:${api_port}" \
            --console-address "${MILVUS_HOST}:${console_port}" >/dev/null
    done

    for ((rank=0; rank<minio_instances; rank++)); do
        wait_for_http_ok "http://${MILVUS_HOST}:$((9000 + (100 * rank)))/minio/health/live" "minio-${rank}" 180
    done
}

launch_local_distributed_role() {
    local role="$1"
    local count="$2"
    local command="$3"
    local rank name config_file data_dir metrics_port storage_type
    local -a shared_storage_args=()

    if [[ "${MINIO_MODE,,}" == "off" ]]; then
        storage_type="local"
        mkdir -p "$LOCAL_SHARED_STORAGE_PATH"
        shared_storage_args=(
            -v "${LOCAL_SHARED_STORAGE_PATH}:${LOCAL_SHARED_STORAGE_PATH}"
        )
    else
        storage_type="remote"
    fi

    for ((rank=0; rank<count; rank++)); do
        name="$(component_container_name "$role" "$rank")"
        config_file="$RUN_DIR/configs/${role}${rank}.yaml"
        data_dir="$RUN_DIR/volumes/${role,,}/${rank}"
        metrics_port="$(role_metrics_port "$role" "$rank")"
        mkdir -p "$data_dir"

        if [[ ! -f "$config_file" ]]; then
            echo "Missing generated config for ${role}${rank}: ${config_file}" >&2
            exit 1
        fi

        remove_container_if_present "$name"
        "$CONTAINER_RUNTIME" run -d \
            --name "$name" \
            --security-opt seccomp:unconfined \
            --network host \
            -e MILVUSCONF=/milvus/configs/ \
            -e DEPLOY_MODE=DISTRIBUTED \
            -e COMMON_STORAGETYPE="$storage_type" \
            -e METRICS_PORT="$metrics_port" \
            -e LOCAL_SHARED_STORAGE_PATH="$LOCAL_SHARED_STORAGE_PATH" \
            -v "${data_dir}:/var/lib/milvus" \
            -v "${config_file}:/milvus/configs/milvus.yaml:ro" \
            "${shared_storage_args[@]}" \
            "$IMAGE" milvus run "$command" >/dev/null
    done
}

wait_for_distributed_role_health() {
    local role="$1"
    local count="$2"
    local rank metrics_port

    for ((rank=0; rank<count; rank++)); do
        metrics_port="$(role_metrics_port "$role" "$rank")"
        wait_for_http_ok "http://${MILVUS_HOST}:${metrics_port}/healthz" "${role,,}-${rank}" 300
    done
}

start_local_distributed_cluster() {
    generate_distributed_configs
    launch_local_distributed_etcd
    launch_local_distributed_minio
    launch_local_distributed_role "COORDINATOR" "$COORDINATOR_NODES" "mixcoord"
    launch_local_distributed_role "STREAMING" "$STREAMING_NODES" "streamingnode"
    launch_local_distributed_role "QUERY" "$QUERY_NODES" "querynode"
    launch_local_distributed_role "DATA" "$DATA_NODES" "datanode"
    launch_local_distributed_role "PROXY" "$NUM_PROXIES" "proxy"
    wait_for_distributed_role_health "COORDINATOR" "$COORDINATOR_NODES"
    wait_for_distributed_role_health "STREAMING" "$STREAMING_NODES"
    wait_for_distributed_role_health "QUERY" "$QUERY_NODES"
    wait_for_distributed_role_health "DATA" "$DATA_NODES"
    wait_for_distributed_role_health "PROXY" "$NUM_PROXIES"
}

start_local_minio() {
    case "$MINIO_MODE" in
        off)
            return 0
            ;;
        single)
            ;;
        *)
            echo "Local mode only supports MINIO_MODE='off' or 'single' (got '${MINIO_MODE}')." >&2
            exit 1
            ;;
    esac

    ensure_container_network

    if container_exists "$MINIO_CONTAINER_NAME"; then
        local minio_port_bindings
        local minio_mount_source
        minio_port_bindings="$("$CONTAINER_RUNTIME" inspect "$MINIO_CONTAINER_NAME" --format '{{json .HostConfig.PortBindings}}')"
        minio_mount_source="$(container_mount_source "$MINIO_CONTAINER_NAME" "/data")"

        if [[ "$minio_port_bindings" != *"9000/tcp"* ]]; then
            echo "Recreating MinIO container '$MINIO_CONTAINER_NAME' to publish host ports."
            remove_container_if_present "$MINIO_CONTAINER_NAME"
        elif [[ "$minio_mount_source" != "$MINIO_VOLUMES_DIR" ]]; then
            echo "Recreating MinIO container '$MINIO_CONTAINER_NAME' to refresh its data bind mount."
            remove_container_if_present "$MINIO_CONTAINER_NAME"
        elif [[ -d "$minio_mount_source" && ! -w "$minio_mount_source" ]]; then
            echo "Recreating MinIO container '$MINIO_CONTAINER_NAME' because its data directory is not writable."
            remove_container_if_present "$MINIO_CONTAINER_NAME"
        fi
    fi

    if container_running "$MINIO_CONTAINER_NAME"; then
        echo "MinIO container '$MINIO_CONTAINER_NAME' is already running."
    else
        if container_exists "$MINIO_CONTAINER_NAME"; then
            echo "Starting existing MinIO container '$MINIO_CONTAINER_NAME'..."
            "$CONTAINER_RUNTIME" start "$MINIO_CONTAINER_NAME" >/dev/null
        else
            echo "Launching MinIO container '$MINIO_CONTAINER_NAME' from image '$MINIO_IMAGE'..."
            "$CONTAINER_RUNTIME" run -d \
                --name "$MINIO_CONTAINER_NAME" \
                --network "$MINIO_NETWORK_NAME" \
                -e MINIO_ROOT_USER="$MINIO_ACCESS_KEY_ID" \
                -e MINIO_ROOT_PASSWORD="$MINIO_SECRET_ACCESS_KEY" \
                -v "${MINIO_VOLUMES_DIR}:/data" \
                -p "${MINIO_API_PORT}:9000" \
                -p "${MINIO_CONSOLE_PORT}:9001" \
                --health-cmd="curl -f http://localhost:9000/minio/health/live" \
                --health-interval=15s \
                --health-start-period=10s \
                --health-timeout=5s \
                --health-retries=5 \
                "$MINIO_IMAGE" server /data --console-address ":9001" >/dev/null
        fi
    fi

    wait_for_minio
}

start_local_milvus() {
    local storage_type="local"
    local -a minio_env_args=()

    write_local_configs
    ensure_container_network

    case "$MINIO_MODE" in
        off)
            ;;
        single)
            storage_type="remote"
            minio_env_args=(
                -e MINIO_ADDRESS="${MINIO_INTERNAL_HOST}:9000"
                -e MINIO_ACCESS_KEY_ID="${MINIO_ACCESS_KEY_ID}"
                -e MINIO_SECRET_ACCESS_KEY="${MINIO_SECRET_ACCESS_KEY}"
                -e MINIO_BUCKET_NAME="${MINIO_BUCKET_NAME}"
            )
            ;;
        *)
            echo "Local mode only supports MINIO_MODE='off' or 'single' (got '${MINIO_MODE}')." >&2
            exit 1
            ;;
    esac

    if container_exists "$CONTAINER_NAME"; then
        local current_network
        local current_env
        local embed_mount_source
        local user_mount_source
        current_network="$("$CONTAINER_RUNTIME" inspect "$CONTAINER_NAME" --format '{{.HostConfig.NetworkMode}}')"
        current_env="$("$CONTAINER_RUNTIME" inspect "$CONTAINER_NAME" --format '{{join .Config.Env "\n"}}')"
        embed_mount_source="$(container_mount_source "$CONTAINER_NAME" "/milvus/configs/embedEtcd.yaml")"
        user_mount_source="$(container_mount_source "$CONTAINER_NAME" "/milvus/configs/user.yaml")"

        if [[ "$current_network" != "$MINIO_NETWORK_NAME" ]]; then
            echo "Recreating Milvus container '$CONTAINER_NAME' to attach it to network '$MINIO_NETWORK_NAME'."
            remove_container_if_present "$CONTAINER_NAME"
        elif [[ "$storage_type" == "remote" && "$current_env" != *"COMMON_STORAGETYPE=remote"* ]]; then
            echo "Recreating Milvus container '$CONTAINER_NAME' to enable remote object storage."
            remove_container_if_present "$CONTAINER_NAME"
        elif [[ "$storage_type" == "local" && "$current_env" != *"COMMON_STORAGETYPE=local"* ]]; then
            echo "Recreating Milvus container '$CONTAINER_NAME' to restore local object storage."
            remove_container_if_present "$CONTAINER_NAME"
        elif [[ "$embed_mount_source" != "$EMBED_ETCD_FILE" || "$user_mount_source" != "$USER_CONFIG_FILE" ]]; then
            echo "Recreating Milvus container '$CONTAINER_NAME' to refresh config bind mounts."
            remove_container_if_present "$CONTAINER_NAME"
        elif [[ -d "$embed_mount_source" || -d "$user_mount_source" ]]; then
            echo "Recreating Milvus container '$CONTAINER_NAME' because a config bind source is a directory."
            remove_container_if_present "$CONTAINER_NAME"
        fi
    fi

    if container_running "$CONTAINER_NAME"; then
        echo "Milvus container '$CONTAINER_NAME' is already running."
    else
        if container_exists "$CONTAINER_NAME"; then
            echo "Starting existing Milvus container '$CONTAINER_NAME'..."
            "$CONTAINER_RUNTIME" start "$CONTAINER_NAME" >/dev/null
        else
            echo "Launching Milvus container '$CONTAINER_NAME' from image '$IMAGE'..."
            "$CONTAINER_RUNTIME" run -d \
                --name "$CONTAINER_NAME" \
                --security-opt seccomp:unconfined \
                --network "$MINIO_NETWORK_NAME" \
                -e ETCD_USE_EMBED=true \
                -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
                -e ETCD_CONFIG_PATH=/milvus/configs/embedEtcd.yaml \
                -e COMMON_STORAGETYPE="${storage_type}" \
                -e DEPLOY_MODE=STANDALONE \
                -v "${VOLUMES_DIR}:/var/lib/milvus" \
                -v "${EMBED_ETCD_FILE}:/milvus/configs/embedEtcd.yaml" \
                -v "${USER_CONFIG_FILE}:/milvus/configs/user.yaml" \
                -p "${MILVUS_GRPC_PORT}:19530" \
                -p "${MILVUS_HEALTH_PORT}:9091" \
                -p "${ETCD_PORT}:2379" \
                --health-cmd="curl -f http://localhost:9091/healthz" \
                --health-interval=30s \
                --health-start-period=90s \
                --health-timeout=20s \
                --health-retries=3 \
                "${minio_env_args[@]}" \
                "$IMAGE" milvus run standalone >/dev/null
        fi
    fi

    write_registry_files
    wait_for_milvus
}

run_setup_collection() {
    env "${PYTHON_ENV_VARS[@]}" python3 ./setup_collection.py
}

resolve_mixed_insert_start_id

normalize_insert_method() {
    local method="${INSERT_METHOD:-traditional}"
    method="${method,,}"
    case "$method" in
        traditional|standard|direct)
            printf 'traditional\n'
            ;;
        bulk|bulk_upload|bulk-upload|import)
            printf 'bulk\n'
            ;;
        *)
            echo "Unsupported INSERT_METHOD='$INSERT_METHOD'. Valid options: traditional, bulk" >&2
            exit 1
            ;;
    esac
}

normalize_bulk_upload_transport() {
    local transport="${BULK_UPLOAD_TRANSPORT:-writer}"
    transport="${transport,,}"
    case "$transport" in
        writer|remote_writer|remote-writer)
            printf 'writer\n'
            ;;
        mc|mc_cp|mc-cp|minio_mc|minio-mc)
            printf 'mc\n'
            ;;
        *)
            echo "Unsupported BULK_UPLOAD_TRANSPORT='$BULK_UPLOAD_TRANSPORT'. Valid options: writer, mc" >&2
            exit 1
            ;;
    esac
}

run_insert() {
	export ACTIVE_TASK="INSERT"
	: "${INSERT_BALANCE_STRATEGY:?INSERT_BALANCE_STRATEGY is required}"
	: "${INSERT_CLIENTS_PER_PROXY:?INSERT_CLIENTS_PER_PROXY is required}"
	: "${INSERT_DATA_FILEPATH:?INSERT_DATA_FILEPATH is required}"
	: "${INSERT_BATCH_SIZE:?INSERT_BATCH_SIZE is required}"

	env GOGC="${LOCAL_INSERT_GOGC:-25}" "${PYTHON_ENV_VARS[@]}" "$STANDARD_BINARY_PATH"
}

run_bulk_upload() {
    export ACTIVE_TASK="IMPORT"
    : "${INSERT_BATCH_SIZE:?INSERT_BATCH_SIZE is required}"
    : "${IMPORT_PROCESSES:?IMPORT_PROCESSES is required}"
    COLLECTION_NAME="${COLLECTION_NAME:-standalone}"
    VECTOR_FIELD="${VECTOR_FIELD:-vector}"
    ID_FIELD="${ID_FIELD:-id}"
    export COLLECTION_NAME VECTOR_FIELD ID_FIELD
    export MINIO_ENDPOINT="${MINIO_ENDPOINT:-${MINIO_HOST}:${MINIO_API_PORT}}"
    local bulk_transport
    local bulk_script
    local bulk_transport_args=()
    local bulk_request_args=()

    if [[ -n "${BULK_IMPORT_LOAD_REQUEST:-}" ]]; then
        bulk_request_args+=(--load-import-request "$BULK_IMPORT_LOAD_REQUEST")
    else
        : "${INSERT_DATA_FILEPATH:?INSERT_DATA_FILEPATH is required}"
        bulk_request_args+=(--input "$INSERT_DATA_FILEPATH")
    fi

    if [[ -n "${BULK_IMPORT_REQUEST_PATH:-}" ]]; then
        bulk_request_args+=(--import-request-path "$BULK_IMPORT_REQUEST_PATH")
    fi

    if [[ "${BULK_IMPORT_PREPARE_ONLY:-}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
        bulk_request_args+=(--prepare-only)
    fi

    if [[ "${MINIO_MODE}" != "single" ]]; then
        echo "TASK=IMPORT requires MINIO_MODE=single in local mode." >&2
        exit 1
    fi

    bulk_transport="$(normalize_bulk_upload_transport)"
    if [[ "$bulk_transport" == "mc" ]]; then
        bulk_script="./bulk_upload_import_mc.py"
    else
        bulk_script="./bulk_upload_import.py"
        bulk_transport_args+=(--writer-mode remote)
    fi

    local bulk_args=(
        --processes "$IMPORT_PROCESSES"
        --collection "$COLLECTION_NAME"
        --vector-field "$VECTOR_FIELD"
        --id-field "$ID_FIELD"
        --vector-dim "$VECTOR_DIM"
        --batch-rows "$INSERT_BATCH_SIZE"
        "${bulk_transport_args[@]}"
        "${bulk_request_args[@]}"
    )
    if [[ -n "${INSERT_CORPUS_SIZE:-}" ]]; then
        bulk_args+=(--corpus-size "$INSERT_CORPUS_SIZE")
    fi

    env "${PYTHON_ENV_VARS[@]}" python3 "$bulk_script" "${bulk_args[@]}"
}

run_insert_for_task() {
    local insert_method
    insert_method="$(normalize_insert_method)"

    if [[ "$insert_method" == "bulk" ]]; then
        run_bulk_upload
    else
        run_insert
    fi
}

run_index() {
    export ACTIVE_TASK="INDEX"
    touch "$RUNTIME_STATE_DIR/workflow_start.txt"
    env "${PYTHON_ENV_VARS[@]}" python3 ./index.py
}

run_query() {
    export ACTIVE_TASK="QUERY"
    : "${QUERY_BALANCE_STRATEGY:?QUERY_BALANCE_STRATEGY is required}"
    : "${QUERY_CLIENTS_PER_PROXY:?QUERY_CLIENTS_PER_PROXY is required}"
    : "${QUERY_DATA_FILEPATH:?QUERY_DATA_FILEPATH is required}"
    : "${QUERY_BATCH_SIZE:?QUERY_BATCH_SIZE is required}"

    env "${PYTHON_ENV_VARS[@]}" "$STANDARD_BINARY_PATH"
}

run_mixed() {
    export ACTIVE_TASK="MIXED"
    : "${INSERT_DATA_FILEPATH:?INSERT_DATA_FILEPATH is required}"
    : "${MIXED_INSERT_BATCH_SIZE:?MIXED_INSERT_BATCH_SIZE is required}"
    : "${MIXED_QUERY_BATCH_SIZE:?MIXED_QUERY_BATCH_SIZE is required}"
    : "${INSERT_BALANCE_STRATEGY:?INSERT_BALANCE_STRATEGY is required}"
    : "${QUERY_BALANCE_STRATEGY:?QUERY_BALANCE_STRATEGY is required}"
    : "${MIXED_QUERY_DATA_FILEPATH:?MIXED_QUERY_DATA_FILEPATH is required}"
    : "${MIXED_INSERT_DATA_FILEPATH:?MIXED_INSERT_DATA_FILEPATH is required}"
    : "${INSERT_START_ID:?INSERT_START_ID is required}"
    MIXED_RESULT_PATH="${MIXED_RESULT_PATH:-mixed_logs}"
    EFSearch="${EFSearch:-$QUERY_EF_SEARCH}"
    export MIXED_RESULT_PATH MIXED_INSERT_CLIENTS MIXED_QUERY_CLIENTS EFSearch

    mkdir -p "$MIXED_RESULT_PATH"
    env "${PYTHON_ENV_VARS[@]}" "$MIXED_BINARY_PATH"
}

run_mixed_timeline() {
    local mixed_timeline_metric="dot"
    if [[ "$DISTANCE_METRIC" == "COSINE" ]]; then
        mixed_timeline_metric="cosine"
    elif [[ "$DISTANCE_METRIC" == "L2" ]]; then
        mixed_timeline_metric="l2"
    fi

    local mixed_timeline_args=(
        ./mixed_timeline.py
        --log-dir "$MIXED_RESULT_PATH"
        --insert-vectors "$MIXED_INSERT_DATA_FILEPATH"
        --query-vectors "$MIXED_QUERY_DATA_FILEPATH"
        --metric "$mixed_timeline_metric"
        --insert-id-offset "$INSERT_START_ID"
    )
    if [[ -n "$MIXED_INSERT_CORPUS_SIZE" ]]; then
        mixed_timeline_args+=(--insert-max-rows "$MIXED_INSERT_CORPUS_SIZE")
    fi
    if [[ -n "$MIXED_QUERY_CORPUS_SIZE" ]]; then
        mixed_timeline_args+=(--query-max-rows "$MIXED_QUERY_CORPUS_SIZE")
    fi

    if [[ -z "$RESTORE_DIR" ]]; then
        mixed_timeline_args+=(--init-vectors "$INSERT_DATA_FILEPATH")
        if [[ -n "$INSERT_CORPUS_SIZE" ]]; then
            mixed_timeline_args+=(--init-max-rows "$INSERT_CORPUS_SIZE")
        fi
    fi

    env "${PYTHON_ENV_VARS[@]}" python3 "${mixed_timeline_args[@]}"
}

summarize_insert() {
    env "${PYTHON_ENV_VARS[@]}" python3 ./multi_client_summary.py
    [[ -f times.csv ]] && mv times.csv insert_times.txt
    [[ -f summary.csv ]] && mv summary.csv insert_summary.txt
}

stage_insert_client_outputs() {
    mkdir -p uploadNPY
    shopt -s nullglob
    local files=(./*.npy)
    if (( ${#files[@]} > 0 )); then
        mv "${files[@]}" uploadNPY/
    fi
    shopt -u nullglob
}

summarize_query() {
    env "${PYTHON_ENV_VARS[@]}" python3 ./multi_client_summary.py
    [[ -f times.csv ]] && mv times.csv query_times.txt
    [[ -f summary.csv ]] && mv summary.csv query_summary.txt
}

stage_query_client_outputs() {
    mkdir -p queryNPY
    shopt -s nullglob
    local files=(./*.npy)
    if (( ${#files[@]} > 0 )); then
        mv "${files[@]}" queryNPY/
    fi
    shopt -u nullglob
}

calculate_recall_if_enabled() {
    [[ "${CALCULATE_RECALL:-False}" == "True" ]] || return 0

    shopt -s nullglob
    local query_id_files=(queryNPY/query_result_ids*.npy)
    shopt -u nullglob
    if (( ${#query_id_files[@]} == 0 )); then
        echo "CALCULATE_RECALL=True but no queryNPY/query_result_ids*.npy files were produced." >&2
        return 1
    fi

    env "${PYTHON_ENV_VARS[@]}" python3 ./compute_recall.py \
        "$GROUND_TRUTH_FILEPATH" \
        "${query_id_files[@]}" \
        --top-k "${TOP_K:-10}" \
        --output recall.csv
}

cleanup_client_timings() {
    mkdir -p "$RUN_DIR/clientTimings"
    shopt -s nullglob
    local candidate
    local timing_files=()
    for candidate in \
        "$RUN_DIR"/*_times.txt \
        "$RUN_DIR"/*_summary.txt \
        "$RUN_DIR"/times.csv \
        "$RUN_DIR"/summary.csv \
        "$RUN_DIR"/index_time.txt \
        "$RUN_DIR"/collection_time.txt
    do
        [[ -e "$candidate" ]] || continue
        timing_files+=("$candidate")
    done
    if (( ${#timing_files[@]} > 0 )); then
        mv "${timing_files[@]}" "$RUN_DIR/clientTimings"/
    fi
    shopt -u nullglob
}

move_yaml_files_to_runtime_state() {
    local yaml_file rel_path target_path

    while IFS= read -r -d '' yaml_file; do
        rel_path="${yaml_file#$RUN_DIR/}"
        target_path="$RUNTIME_STATE_DIR/$rel_path"
        mkdir -p "$(dirname "$target_path")"
        mv "$yaml_file" "$target_path"
    done < <(
        find "$RUN_DIR" \
            -path "$RUNTIME_STATE_DIR" -prune -o \
            -path "$RUN_DIR/volumes" -prune -o \
            -type f \( -name '*.yaml' -o -name '*.yml' \) -print0
    )
}

run_restore_status() {
    export EXPECTED_CORPUS_SIZE
    env "${PYTHON_ENV_VARS[@]}" python3 ./status.py
}

main() {
    cd "$RUN_DIR"
    local should_summarize_insert=0
    local should_summarize_query=0
    ensure_runtime_tools
    case "${MODE^^}" in
        STANDALONE)
            start_local_minio
            start_local_milvus
            ;;
        DISTRIBUTED)
            start_local_distributed_cluster
            ;;
        *)
            echo "Unsupported MODE '$MODE' for local_main.sh" >&2
            exit 1
            ;;
    esac

    if [[ -z "$RESTORE_DIR" ]]; then
        run_setup_collection

        if [[ "$TASK" == "IMPORT" ]]; then
            run_bulk_upload
            touch "$RUNTIME_STATE_DIR/flag.txt"
        else
            if [[ "$TASK" == "INSERT" ]]; then
                run_insert
            else
                run_insert_for_task
            fi

            if [[ "$TASK" == "INSERT" ]]; then
                touch "$RUNTIME_STATE_DIR/flag.txt"
            fi

            if [[ "$TASK" == "INSERT" ]] || [[ "$TASK" == "MIXED" && "$(normalize_insert_method)" == "traditional" ]]; then
                should_summarize_insert=1
                stage_insert_client_outputs
            fi

            if [[ "$TASK" == "INDEX" || "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
                run_index

                if [[ "$TASK" == "INDEX" ]]; then
                    touch "$RUNTIME_STATE_DIR/workflow_end.txt"
                    touch "$RUNTIME_STATE_DIR/flag.txt"
                fi
            fi

            sleep 5
        fi
    else
        run_restore_status
    fi

    if [[ "$TASK" == "QUERY" ]]; then
        run_query
        should_summarize_query=1
        stage_query_client_outputs
    elif [[ "$TASK" == "MIXED" ]]; then
        run_mixed
        run_mixed_timeline
    elif [[ "$TASK" != "INSERT" && "$TASK" != "IMPORT" && "$TASK" != "INDEX" && "$TASK" != "QUERY" && "$TASK" != "MIXED" ]]; then
        echo "Unsupported TASK '$TASK' for local_main.sh" >&2
        exit 1
    fi

    if (( should_summarize_insert )); then
        summarize_insert
    fi

    if (( should_summarize_query )); then
        summarize_query
    fi

    cleanup_client_timings
    move_yaml_files_to_runtime_state
    calculate_recall_if_enabled
}

main "$@"
