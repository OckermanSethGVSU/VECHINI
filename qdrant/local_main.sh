#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="${RUN_DIR:-$(pwd)}"

if [[ -f "$RUN_DIR/run_config.env" ]]; then
    set -a
    source "$RUN_DIR/run_config.env"
    set +a
fi

cd "$RUN_DIR"

export myDIR="${myDIR:-$(basename "$RUN_DIR")}"
export RESULT_PATH="${RESULT_PATH:-mixed_logs}"
export CLIENT_TIMING_DIR="${CLIENT_TIMING_DIR:-clientTiming}"
export MIXED_INSERT_MODE="${MIXED_INSERT_MODE:-}"
export INSERT_OPS_PER_SEC="${INSERT_OPS_PER_SEC:-}"
export MIXED_QUERY_MODE="${MIXED_QUERY_MODE:-}"
export QUERY_OPS_PER_SEC="${QUERY_OPS_PER_SEC:-}"
export INSERT_START_ID="${INSERT_START_ID:-0}"
export COLLECTION_NAME
export TOP_K="${TOP_K:-}"
export QUERY_EF_SEARCH="${QUERY_EF_SEARCH:-}"
export TOTAL_QUERY_CLIENTS="${TOTAL_QUERY_CLIENTS:-}"
export INSERT_STREAMING="${INSERT_STREAMING:-}"
export QUERY_STREAMING="${QUERY_STREAMING:-}"
export RPC_TIMEOUT="${RPC_TIMEOUT:-}"
export QDRANT_REGISTRY_PATH="${QDRANT_REGISTRY_PATH:-$RUN_DIR/ip_registry.txt}"
export INSERT_BATCH_MIN="${INSERT_BATCH_MIN:-}"
export INSERT_BATCH_MAX="${INSERT_BATCH_MAX:-}"
export QUERY_BATCH_MIN="${QUERY_BATCH_MIN:-}"
export QUERY_BATCH_MAX="${QUERY_BATCH_MAX:-}"

export WORKLOAD_MODE="${WORKLOAD_MODE:-}"
export EXPECTED_CORPUS_SIZE="${EXPECTED_CORPUS_SIZE:-0}"
export N_WORKERS=1

CONTAINER_NAME="${QDRANT_LOCAL_NAME:-qdrant-local}"
IMAGE="${QDRANT_LOCAL_IMAGE:-qdrant/qdrant:latest}"
HOST="${QDRANT_LOCAL_HOST:-127.0.0.1}"
HTTP_PORT="${QDRANT_LOCAL_HTTP_PORT:-6333}"
GRPC_PORT="${QDRANT_LOCAL_GRPC_PORT:-6334}"
P2P_PORT="${QDRANT_LOCAL_P2P_PORT:-6335}"
DATA_DIR="${QDRANT_LOCAL_DATA_DIR:-$RUN_DIR/.local/qdrant/storage}"
CONFIG_DIR="${QDRANT_LOCAL_CONFIG_DIR:-$RUN_DIR/.local/qdrant/config}"
SNAPSHOT_DIR="${QDRANT_LOCAL_SNAPSHOT_DIR:-$RUN_DIR/.local/qdrant/snapshots}"
export RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-$RUN_DIR/runtime_state}"
BATCH_CLIENT_BINARY_PATH="${BATCH_CLIENT_BINARY_PATH:-}"
MIXED_BINARY_PATH="${MIXED_BINARY_PATH:-}"
QUERY_DEBUG_RESULTS="${QUERY_DEBUG_RESULTS:-true}"
LOCAL_RECREATE_COLLECTION="${LOCAL_RECREATE_COLLECTION:-true}"

export QDRANT_HOST="$HOST"
export QDRANT_REST_PORT="$HTTP_PORT"
export QDRANT_GRPC_PORT="$GRPC_PORT"
export QDRANT_URL="http://${HOST}:${GRPC_PORT}"

mkdir -p "$DATA_DIR" "$CONFIG_DIR" "$SNAPSHOT_DIR" "$RUNTIME_STATE_DIR"

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

BATCH_CLIENT_BINARY_PATH="$(pick_binary \
    "$BATCH_CLIENT_BINARY_PATH" \
    "$ROOT_DIR/clients/batch_client/target/release/batch_client" \
    "$ROOT_DIR/clients/batch_client/target/debug/batch_client" \
    "$ROOT_DIR/batch_client")"

MIXED_BINARY_PATH="$(pick_binary \
    "$MIXED_BINARY_PATH" \
    "$ROOT_DIR/clients/mixed/target/release/mixed" \
    "$ROOT_DIR/clients/mixed/target/debug/mixed" \
    "$ROOT_DIR/mixed")"

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
}

ensure_binaries() {
    if [[ "$TASK" == "LAUNCH" ]]; then
        return 0
    fi

    if [[ "$TASK" == "MIXED" || "$WORKLOAD_MODE" == "mixed" ]]; then
        if [[ ! -x "$MIXED_BINARY_PATH" ]]; then
            echo "Missing mixed binary at $MIXED_BINARY_PATH" >&2
            echo "Build it with: (cd $ROOT_DIR/clients/mixed && cargo build --release)" >&2
            exit 1
        fi
    else
        if [[ ! -x "$BATCH_CLIENT_BINARY_PATH" ]]; then
            echo "Missing batch_client binary at $BATCH_CLIENT_BINARY_PATH" >&2
            echo "Build it with: (cd $ROOT_DIR/clients/batch_client && cargo build --release)" >&2
            exit 1
        fi
    fi
}

start_qdrant() {
    if "$CONTAINER_RUNTIME" ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        echo "Qdrant container '$CONTAINER_NAME' is already running."
    else
        if "$CONTAINER_RUNTIME" ps -a --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
            echo "Starting existing Qdrant container '$CONTAINER_NAME'..."
            "$CONTAINER_RUNTIME" start "$CONTAINER_NAME" >/dev/null
        else
            echo "Launching Qdrant container '$CONTAINER_NAME' from image '$IMAGE'..."
            "$CONTAINER_RUNTIME" run -d \
                --name "$CONTAINER_NAME" \
                -p "${HTTP_PORT}:6333" \
                -p "${GRPC_PORT}:6334" \
                -p "${P2P_PORT}:6335" \
                -v "${DATA_DIR}:/qdrant/storage" \
                -v "${CONFIG_DIR}:/qdrant/config/local" \
                -v "${SNAPSHOT_DIR}:/qdrant/snapshots" \
                "$IMAGE" >/dev/null
        fi
    fi

    printf '0,%s,%s\n' "$HOST" "$P2P_PORT" > "$QDRANT_REGISTRY_PATH"
    rm -f "$RUNTIME_STATE_DIR/workflow_start.txt" "$RUNTIME_STATE_DIR/workflow_stop.txt"

    echo "Waiting for Qdrant health check on http://${HOST}:${HTTP_PORT}/healthz ..."
    for _ in {1..60}; do
        if curl -fsS "http://${HOST}:${HTTP_PORT}/healthz" >/dev/null; then
            echo "Qdrant is ready."
            return 0
        fi
        sleep 1
    done

    echo "Qdrant did not become healthy within 60 seconds." >&2
    echo "Inspect logs with: ${CONTAINER_RUNTIME} logs ${CONTAINER_NAME}" >&2
    exit 1
}

stop_qdrant() {
    if "$CONTAINER_RUNTIME" ps --format '{{.Names}}' | grep -Fxq "$CONTAINER_NAME"; then
        echo "Stopping Qdrant container '$CONTAINER_NAME'..."
        "$CONTAINER_RUNTIME" stop "$CONTAINER_NAME" >/dev/null
    fi
}

setup_local_collection() {
    if [[ "$LOCAL_RECREATE_COLLECTION" != "true" ]]; then
        return 0
    fi

    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python3 ./configure_collection.py
}

standard_collection_name() {
    printf '%s\n' "${COLLECTION_NAME:?COLLECTION_NAME is required}"
}

prepare_cluster_state() {
    if [[ -n "$RESTORE_DIR" ]]; then
        return 0
    fi

    if [[ "$TASK" == "MIXED" || "$WORKLOAD_MODE" == "mixed" ]]; then
        mkdir -p "$RESULT_PATH"
        setup_local_collection
    else
        export COLLECTION_NAME="$(standard_collection_name)"
        setup_local_collection
    fi
}

run_insert() {
    echo "Running local insert workflow..."
    export ACTIVE_TASK="INSERT"
    export COLLECTION_NAME="$(standard_collection_name)"
    INSERT_CORPUS_SIZE="${INSERT_CORPUS_SIZE:-}"
    INSERT_CLIENTS_PER_WORKER="${INSERT_CLIENTS_PER_WORKER:-1}"
    INSERT_BATCH_SIZE="${INSERT_BATCH_SIZE:-1}"
    INSERT_BALANCE_STRATEGY="${INSERT_BALANCE_STRATEGY:-NO_BALANCE}"
    INSERT_DATA_FILEPATH="${INSERT_DATA_FILEPATH:?INSERT_DATA_FILEPATH is required}"
    "$BATCH_CLIENT_BINARY_PATH"
}

run_query() {
    echo "Running local query workflow..."
    export ACTIVE_TASK="QUERY"
    export COLLECTION_NAME="$(standard_collection_name)"
    QUERY_CORPUS_SIZE="${QUERY_CORPUS_SIZE:-}"
    QUERY_CLIENTS_PER_WORKER="${QUERY_CLIENTS_PER_WORKER:-1}"
    TOTAL_QUERY_CLIENTS="${TOTAL_QUERY_CLIENTS:-}"
    QUERY_BATCH_SIZE="${QUERY_BATCH_SIZE:-1}"
    QUERY_BALANCE_STRATEGY="${QUERY_BALANCE_STRATEGY:-NO_BALANCE}"
    QUERY_DATA_FILEPATH="${QUERY_DATA_FILEPATH:?QUERY_DATA_FILEPATH is required}"
    "$BATCH_CLIENT_BINARY_PATH"
}

run_mixed() {
    echo "Running local mixed insert/query workflow..."
    INSERT_CORPUS_SIZE="${INSERT_CORPUS_SIZE:-}"
    INSERT_DATA_FILEPATH="${INSERT_DATA_FILEPATH:?INSERT_DATA_FILEPATH is required}"
    MIXED_INSERT_DATA_FILEPATH="${MIXED_INSERT_DATA_FILEPATH:?MIXED_INSERT_DATA_FILEPATH is required}"
    MIXED_QUERY_DATA_FILEPATH="${MIXED_QUERY_DATA_FILEPATH:?MIXED_QUERY_DATA_FILEPATH is required}"
    MIXED_INSERT_CLIENTS="${MIXED_INSERT_CLIENTS:-1}"
    MIXED_QUERY_CLIENTS="${MIXED_QUERY_CLIENTS:-1}"
    MIXED_INSERT_BATCH_SIZE="${MIXED_INSERT_BATCH_SIZE:-1}"
    MIXED_QUERY_BATCH_SIZE="${MIXED_QUERY_BATCH_SIZE:-1}"
    INSERT_BALANCE_STRATEGY="${INSERT_BALANCE_STRATEGY:-NO_BALANCE}"
    QUERY_BALANCE_STRATEGY="${QUERY_BALANCE_STRATEGY:-NO_BALANCE}"
    MIXED_INSERT_CORPUS_SIZE="${MIXED_INSERT_CORPUS_SIZE:-$INSERT_CORPUS_SIZE}"
    MIXED_QUERY_CORPUS_SIZE="${MIXED_QUERY_CORPUS_SIZE:-$QUERY_CORPUS_SIZE}"
    MIXED_INSERT_MODE="${MIXED_INSERT_MODE:-}"
    INSERT_OPS_PER_SEC="${INSERT_OPS_PER_SEC:-}"
    INSERT_START_ID="${INSERT_START_ID:-0}"
    MIXED_QUERY_MODE="${MIXED_QUERY_MODE:-}"
    QUERY_OPS_PER_SEC="${QUERY_OPS_PER_SEC:-}"
    export COLLECTION_NAME
    TOP_K="${TOP_K:-}"
    QUERY_EF_SEARCH="${QUERY_EF_SEARCH:-}"
    RPC_TIMEOUT="${RPC_TIMEOUT:-}"
    QDRANT_REGISTRY_PATH="${QDRANT_REGISTRY_PATH:-$RUN_DIR/ip_registry.txt}"
    INSERT_BATCH_MIN="${INSERT_BATCH_MIN:-}"
    INSERT_BATCH_MAX="${INSERT_BATCH_MAX:-}"
    QUERY_BATCH_MIN="${QUERY_BATCH_MIN:-}"
    QUERY_BATCH_MAX="${QUERY_BATCH_MAX:-}"
    mkdir -p "$RESULT_PATH"
    "$MIXED_BINARY_PATH"
}

summarize_standard_run() {
    local task_name="$1"
    local npy_dir="$2"
    [[ -d "$npy_dir" ]] || return 0
    shopt -s nullglob
    local npy_files=("$npy_dir"/*.npy)
    shopt -u nullglob
    (( ${#npy_files[@]} > 0 )) || return 0
    mkdir -p "$CLIENT_TIMING_DIR"
    ACTIVE_TASK="$task_name" python3 ./summarize_client_timings.py \
        --npy-dir "$npy_dir" \
        --output-dir "$CLIENT_TIMING_DIR" \
        --times-csv "./${task_name,,}_times.csv"
}

move_standard_npy_files() {
    local target_dir="$1"
    mkdir -p "$target_dir"
    shopt -s nullglob
    local npy_files=(./*.npy)
    if (( ${#npy_files[@]} > 0 )); then
        mv "${npy_files[@]}" "$target_dir"/
    fi
    shopt -u nullglob
}

finalize_local_run() {
    touch flag.txt "$RUNTIME_STATE_DIR/flag.txt"
    mkdir -p "$CLIENT_TIMING_DIR"
    mkdir -p systemStats
    shopt -s nullglob
    local system_files=(./*_system_*.csv)
    if (( ${#system_files[@]} > 0 )); then
        mv "${system_files[@]}" systemStats/
    fi
    local timing_files=()
    [[ -f ./index_time.txt ]] && timing_files+=(./index_time.txt)
    timing_files+=(./*_times.csv ./*_summary.csv)
    if (( ${#timing_files[@]} > 0 )); then
        mv "${timing_files[@]}" "$CLIENT_TIMING_DIR"/
    fi
    shopt -u nullglob
    sleep 2
    rm -f flag.txt
    if [[ -f ./ip_registry.txt ]]; then
        mv ./ip_registry.txt "$RUNTIME_STATE_DIR"/
    fi
    if [[ -d ./ip_registry.d ]]; then
        mv ./ip_registry.d "$RUNTIME_STATE_DIR"/
    fi
}

wait_for_launch_stop_flag() {
    echo "TASK=LAUNCH: Qdrant is up and will stay running until you create flag.txt or runtime_state/flag.txt in this run directory."

    while [[ ! -e flag.txt && ! -e "$RUNTIME_STATE_DIR/flag.txt" ]]; do
        sleep 1
    done

    echo "TASK=LAUNCH: stop flag detected."
    touch flag.txt "$RUNTIME_STATE_DIR/flag.txt"
    stop_qdrant
}

run_restore_status() {
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python3 ./collection_status.py
}

run_index() {
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python3 ./build_index.py
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
        --log-dir "$RESULT_PATH"
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
        mixed_timeline_args+=(
            --init-vectors "$INSERT_DATA_FILEPATH"
        )
        if [[ -n "$INSERT_CORPUS_SIZE" ]]; then
            mixed_timeline_args+=(--init-max-rows "$INSERT_CORPUS_SIZE")
        fi
    fi

    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python3 "${mixed_timeline_args[@]}"
}

main() {
    cd "$RUN_DIR"
    ensure_runtime_tools
    ensure_binaries
    start_qdrant

    if [[ "$TASK" == "LAUNCH" ]]; then
        wait_for_launch_stop_flag
        return 0
    fi

    prepare_cluster_state

    if [[ -n "$RESTORE_DIR" ]]; then
        run_restore_status
    else
        run_insert

        if [[ "$TASK" == "INSERT" ]]; then
            move_standard_npy_files uploadNPY
            summarize_standard_run INSERT uploadNPY
            finalize_local_run
            return 0
        fi

        move_standard_npy_files uploadNPY

        if [[ "$TASK" == "INDEX" ]]; then
            run_index
            summarize_standard_run INSERT uploadNPY
            finalize_local_run
            return 0
        fi

        if [[ "$TASK" == "QUERY" ]]; then
            run_index
        fi

        if [[ "$TASK" == "MIXED" || "$WORKLOAD_MODE" == "mixed" ]]; then
            run_index
            run_mixed
            run_mixed_timeline
            echo "Mixed logs written to: $RESULT_PATH"
            return 0
        fi
    fi

    if [[ "$TASK" == "QUERY" ]]; then
        run_query
        move_standard_npy_files queryNPY
        summarize_standard_run INSERT uploadNPY
        summarize_standard_run QUERY queryNPY
        finalize_local_run
        return 0
    fi

    if [[ "$TASK" == "MIXED" || "$WORKLOAD_MODE" == "mixed" ]]; then
        run_mixed
        run_mixed_timeline
        echo "Mixed logs written to: $RESULT_PATH"
        return 0
    fi

    if [[ "$TASK" != "INSERT" && "$TASK" != "INDEX" && "$TASK" != "QUERY" && "$TASK" != "MIXED" && "$TASK" != "LAUNCH" ]]; then
        echo "Unsupported TASK '$TASK' for local_main.sh" >&2
        exit 1
    fi
}

main "$@"
