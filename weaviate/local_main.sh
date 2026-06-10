#!/usr/bin/env bash
set -euo pipefail

run_summary() {
    local task="$1"     # INSERT or QUERY
    local prefix="$2"   # insert or query
    ACTIVE_TASK="$task" python3 multi_client_summary.py
}


resolve_mixed_insert_start_id() {
    if [[ "$TASK" != "MIXED" || -n "${MIXED_INSERT_START_ID:-}" ]]; then
        return 0
    fi

    if [[ -n "${RESTORE_DIR:-}" ]]; then
        export MIXED_INSERT_START_ID="${EXPECTED_CORPUS_SIZE:?EXPECTED_CORPUS_SIZE is required when RESTORE_DIR is set}"
    elif [[ -n "${INSERT_CORPUS_SIZE:-}" ]]; then
        export MIXED_INSERT_START_ID="$INSERT_CORPUS_SIZE"
    elif [[ -n "${INSERT_DATA_FILEPATH:-}" ]]; then
        if ! export MIXED_INSERT_START_ID="$(env "${PYTHON_ENV_VARS[@]}" python3 ./npy_inspect.py "$INSERT_DATA_FILEPATH")"; then
            echo "Error: failed to derive MIXED_INSERT_START_ID from INSERT_DATA_FILEPATH using npy_inspect.py." >&2
            exit 1
        fi
    else
        echo "Error: TASK=MIXED requires MIXED_INSERT_START_ID, INSERT_CORPUS_SIZE, RESTORE_DIR, or INSERT_DATA_FILEPATH." >&2
        exit 1
    fi
}

if [[ -f ./run_config.env ]]; then
    set -a
    source ./run_config.env
    set +a
fi

if [[ -z "${BASE_DIR:-}" ]]; then
    BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

RUN_DIR="${RUN_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

TOTAL=1
MAX_RANK=0
export N_WORKERS=$TOTAL

mkdir -p "$RUN_DIR/runtime_state"
cd "$RUN_DIR"

./launchWeaviateNodeLocal.sh "$STORAGE_MEDIUM" "$TOTAL" &
LAUNCH_PID=$!

echo "[INFO] Waiting for ${TOTAL} local Weaviate worker to become ready..."
for r in $(seq 0 "${MAX_RANK}"); do
    target="./runtime_state/weaviate_running${r}.txt"
    while [[ ! -e "${target}" ]]; do
        sleep 0.5
    done
done
echo "[INFO] All ${TOTAL} local worker(s) are ready"

NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    python3 ./health_check.py --registry ./ip_registry.txt

CREATE_COLLECTION_ARGS=(
    ./create_basic_collection.py
    --registry ./ip_registry.txt
    --rank 0
)
if [[ "${LOCAL_RECREATE_CLASS:-false}" == "true" ]]; then
    CREATE_COLLECTION_ARGS+=(--drop-if-exists)
fi


NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 create_basic_collection.py


if [ "$TASK" = "INSERT" ]; then
    export ACTIVE_TASK="INSERT"
elif [ "$TASK" = "INDEX" ] || [ "$TASK" = "QUERY" ] || [ "$TASK" = "MIXED" ]; then
    export ACTIVE_TASK="INDEX"
else
    echo "Unknown TASK: $TASK"
    exit 1
fi


NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client
mkdir -p uploadNPY
mv *.npy uploadNPY/

if [[ "$TASK" == "INSERT" || "$TASK" == "INDEX" ]]; then
        touch "runtime_state/flag.txt"
fi


if [[ "$TASK" == "QUERY" ]]; then
    export ACTIVE_TASK="QUERY"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client

    mkdir -p queryNPY
    mv *.npy queryNPY/
    run_summary QUERY query
    touch "runtime_state/flag.txt"
fi

if [[ "$TASK" == "MIXED" ]]; then
    export ACTIVE_TASK="MIXED"
    resolve_mixed_insert_start_id
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./mixed
    touch "runtime_state/flag.txt"
    python3 mixed_timeline.py --log-dir mixed_logs/ --throughput-only

fi



run_summary INSERT insert
