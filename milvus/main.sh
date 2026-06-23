
if [[ -n "${PBS_O_WORKDIR:-}" ]]; then
    SCRIPT_DIR="$PBS_O_WORKDIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"

if [[ -f ./run_config.env ]]; then
    set -a
    source ./run_config.env
    set +a
fi

launch_role() {
    local role="$1"                  # e.g., STREAMING, QUERY, DATA
    local total_ranks="$2"           # e.g., $STREAMING_NODES
    local ranks_per_node="$3"        # e.g., $STREAMING_NODES_PER_CN
    local storage_medium="$4"        # e.g., $STORAGE_MEDIUM
    local script="$5"                # e.g., ./launch_milvus_part.sh

    if (( total_ranks <= 0 || ranks_per_node <= 0 )); then
        echo "ERROR [$role]: ranks and ranks_per_node must be > 0" >&2
        exit 1
    fi

    # ceil(total_ranks / ranks_per_node)
    local nodes_needed=$(( (total_ranks + ranks_per_node - 1) / ranks_per_node ))

    # NODES[0] is reserved
    local available=$(( ${#NODES[@]} - 1 ))

    if (( nodes_needed > available )); then
        echo "ERROR [$role]: need $nodes_needed nodes but only $available available (excluding NODES[0])" >&2
        exit 1
    fi

    # Build hostlist from NODES[1..nodes_needed]
    local hostlist=""
    for ((i=1; i<=nodes_needed; i++)); do
        hostlist+="${NODES[i]},"
    done
    hostlist="${hostlist%,}"

    echo "Launching $role:"
    echo "  total ranks      = $total_ranks"
    echo "  ranks per node   = $ranks_per_node"
    echo "  nodes needed     = $nodes_needed"
    echo "  hosts            = $hostlist"
    echo "  storage medium   = $storage_medium"

    mpirun -n "$total_ranks" \
        --ppn "$ranks_per_node" \
        --cpu-bind none \
        --host "$hostlist" \
        "$script" "$storage_medium" "$role" &
}

wait_for_signal_files() {
    local missing=1

    while (( missing )); do
        missing=0
        for signal_file in "$@"; do
            if [[ ! -e "$signal_file" ]]; then
                missing=1
                break
            fi
        done

        if (( missing )); then
            sleep 0.1
        fi
    done
}

run_summary() {
    local task="$1"     # INSERT or QUERY
    local prefix="$2"   # insert or query
    env "${PYTHON_ENV_VARS[@]}" ACTIVE_TASK="$task" python3 multi_client_summary.py
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

compute_local_shared_storage_path() {
    if [[ -n "${LOCAL_SHARED_STORAGE_PATH:-}" ]]; then
        printf '%s\n' "$LOCAL_SHARED_STORAGE_PATH"
        return 0
    fi

    case "$STORAGE_MEDIUM" in
        lustre)
            printf '%s/%s/localfs-shared\n' "$BASE_DIR" "$myDIR"
            ;;
        DAOS)
            printf '/tmp/%s/%s/%s/localfs-shared\n' \
                "${DAOS_PROJECT:?DAOS_PROJECT is required when STORAGE_MEDIUM=DAOS}" \
                "${DAOS_CONTAINER:?DAOS_CONTAINER is required when STORAGE_MEDIUM=DAOS}" \
                "$myDIR"
            ;;
        *)
            echo "DISTRIBUTED MINIO_MODE=off requires a shared STORAGE_MEDIUM; unsupported STORAGE_MEDIUM='$STORAGE_MEDIUM'." >&2
            exit 1
            ;;
    esac
}

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

PYTHON_ENV_VARS=(
    NO_PROXY=""
    no_proxy=""
    http_proxy=""
    https_proxy=""
    HTTP_PROXY=""
    HTTPS_PROXY=""
)


RUN_DIR="${RUN_DIR:-$SCRIPT_DIR}"
BASE_DIR="${BASE_DIR:-$(dirname "$RUN_DIR")}"


export myDIR="$(basename "$RUN_DIR")"
export RESULT_PATH="$RUN_DIR"
export RUNTIME_STATE_DIR="${RUN_DIR}/runtime_state"
echo "[INFO] Using run directory: $RUN_DIR"
cd "$RUN_DIR"

mkdir -p "$RUNTIME_STATE_DIR"

if [[ "${MODE^^}" == "DISTRIBUTED" ]]; then
    export PROXY_REGISTRY_PATH="${RUN_DIR}/PROXY/PROXY_registry.txt"
else
    export PROXY_REGISTRY_PATH="${RUN_DIR}/runtime_state/PROXY_registry.txt"
fi

if [[ "$PLATFORM" == "POLARIS" ]]; then
    ml use /soft/modulefiles
    ml spack-pe-base/0.8.1
    ml use /soft/spack/testing/0.8.1/modulefiles
    ml apptainer/main
    ml load e2fsprogs

    module use /soft/modulefiles; module load conda; conda activate base

elif [[ "$PLATFORM" == "AURORA" ]]; then
    module load apptainer
    module load frameworks
    
    PYTHON_ENV_VARS+=(NUMEXPR_NUM_THREADS=108 NUMEXPR_MAX_THREADS=108)
fi

# Activate Python env
source $ENV_PATH/bin/activate
cat $PBS_NODEFILE > all_nodefile.txt

resolve_mixed_insert_start_id



if [[ "$STORAGE_MEDIUM" == "DAOS" || "$ETCD_MEDIUM" == "DAOS" || ( "$MODE" == "DISTRIBUTED" && "$MINIO_MEDIUM" == "DAOS" ) ]]; then
    module use /soft/modulefiles
    module load daos
    DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when a Milvus storage medium uses DAOS}"
    DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when a Milvus storage medium uses DAOS}"

    launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
    mkdir -p /tmp/${DAOS_POOL}/${DAOS_CONT}/$myDIR
fi

if [[ "$TRACING" == "True" ]]; then
    ready=0
    bash launch_otel.sh &
    python3 net_mapping.py --rank 0 --name otel
    IP_ADDR=$(jq -r '.hsn0.ipv4[0]' "otel0.json")
    echo $IP_ADDR > otel.ip
    rm otel0.json
    for i in $(seq 1 90); do
        if env "${PYTHON_ENV_VARS[@]}" curl -fsS "http://${IP_ADDR}:13133/" >/dev/null 2>&1; then
            ready=1
            export OTLP_GRPC_ENDPOINT="${IP_ADDR}:4317"
            sleep 10 # buffer 
            echo "Collector is ready."
            break
        fi

        if [ $((i % 10)) -eq 0 ]; then
            echo "  still waiting (${i}s elapsed)..."
        fi
        sleep 1
    done
    
    if [ "${ready}" = "0" ]; then
        echo "Collector did not become healthy within 90s."
        exit 1
    fi
fi


if [[ "$MODE" == "STANDALONE" ]]; then
    second_node=$(sed -n '2p' "$PBS_NODEFILE")

    if [[ "$MINIO_MODE" == "single" ]]; then
        echo "Launching standalone MinIO: single node on ${second_node}"
        mpirun -n 1 --ppn 1 --no-vni --cpu-bind none --host "$second_node" ./launch_minio.sh "$MINIO_MEDIUM" &
    elif [[ "$MINIO_MODE" != "off" ]]; then
        echo "Unsupported MINIO_MODE='$MINIO_MODE' for standalone. Expected 'off' or 'single'." >&2
        exit 1
    fi

    if [[ -z "${CORES:-}" ]]; then
        echo "Launching standalone: unrestricted cores"

        mpirun -n 1 --ppn 1 --cpu-bind none --host $second_node ./standaloneLaunch.sh 0 $STORAGE_MEDIUM $PLATFORM STANDALONE $WAL &
    else
        echo "Launching standalone: ${CORES} cores"
        mpirun -n 1 --ppn 1 -d $CORES --cpu-bind depth  --host $second_node ./standaloneLaunch.sh 0 $STORAGE_MEDIUM $PLATFORM STANDALONE $WAL &
    fi

    # launch profiling on worker and client nodes
    mpirun -n 1 --ppn 1 --cpu-bind none --host $second_node python3 profile.py worker_0 $PLATFORM &
    python3 profile.py client_node $PLATFORM & 

    TARGET="$RUNTIME_STATE_DIR/milvus_running.txt"
    while [ ! -e "$TARGET" ]; do
    sleep 0.1
    done

    env "${PYTHON_ENV_VARS[@]}" python3 poll.py
    IP_ADDR=$(jq -r '.hsn0.ipv4[0]' interfaces0.json)
    echo "$IP_ADDR" > "$RUNTIME_STATE_DIR/worker.ip"


elif [[ "$MODE" == "DISTRIBUTED" ]]; then
    export ETCD_MODE=$ETCD_MODE
    if [[ "$MINIO_MODE" == "off" ]]; then
        export LOCAL_SHARED_STORAGE_PATH
        LOCAL_SHARED_STORAGE_PATH="$(compute_local_shared_storage_path)"
        export LOCAL_SHARED_STORAGE_PATH
        mkdir -p "$LOCAL_SHARED_STORAGE_PATH"
        echo "Distributed localfs mode enabled; shared storage path: $LOCAL_SHARED_STORAGE_PATH"
    fi
    

    mapfile -t NODES < <(awk '!seen[$0]++' "$PBS_NODEFILE")

    # spreads etcd evenly on up to 3 nodes
    if [[ "$ETCD_MODE" == "replicated" ]]; then        
        ETCD_INSTANCES=3
        TOTAL_NODES=${#NODES[@]}
        if (( TOTAL_NODES < 2 )); then
            echo "ERROR: Need at least 2 nodes (NODES[0] reserved) for ETCD_MODE=replicated"
            exit 1
        fi

        ETCD_SPREAD=$((TOTAL_NODES - 1))
        (( ETCD_SPREAD > 3 )) && ETCD_SPREAD=3

        HOSTS=""
        for ((i=1; i<=ETCD_SPREAD; i++)); do
            HOSTS+="${HOSTS:+,}${NODES[$i]}"
        done
        
        # enough ranks per node to fit 3 total
        PPN=$(( (ETCD_INSTANCES + ETCD_SPREAD - 1) / ETCD_SPREAD ))
        mpirun -n "$ETCD_INSTANCES" --ppn "$PPN" --cpu-bind none --host "$HOSTS" \
            ./launch_etcd.sh "$ETCD_MEDIUM" &

    # Launch 1 etcd instance
    elif [[ "$ETCD_MODE" == "single" ]]; then
        mpirun -n 1 --ppn 1 --cpu-bind none --host ${NODES[1]} ./launch_etcd.sh "$ETCD_MEDIUM" &
    fi
    
    # spread MINIO on up to 4 nodes
    if [[ "$MINIO_MODE" == "stripped" ]]; then
        TOTAL_NODES=${#NODES[@]}
        MINIO_INSTANCES="${MINIO_INSTANCES:-4}"
        MINIO_SPREAD=$((TOTAL_NODES - 1))
        
        if [[ "$MINIO_SPREAD" -gt 4 ]]; then
            MINIO_SPREAD=4
        fi

        HOSTS=""
        for ((r=0; r<MINIO_INSTANCES; r++)); do
            # skip index 0
            node_idx=$(( 1 + (r % MINIO_SPREAD) ))
            host="${NODES[$node_idx]}"
            HOSTS+="${HOSTS:+,}${host}"
        done
        PPN=$(( (MINIO_INSTANCES + MINIO_SPREAD - 1) / MINIO_SPREAD ))
        
        mpirun -n 4 --ppn $PPN --no-vni --cpu-bind none --host "$HOSTS" \
        ./launch_minio.sh  $MINIO_MEDIUM & # must be lustre or DAOS for erasure coding to work

    elif [[ "$MINIO_MODE" == "single" ]]; then
        # Launch 1 Minio instance
        mpirun -n 1 --ppn 1 --no-vni --cpu-bind none --host "${NODES[1]}"  ./launch_minio.sh $MINIO_MEDIUM &
    elif [[ "$MINIO_MODE" != "off" ]]; then
        echo "Unsupported MINIO_MODE='$MINIO_MODE' for distributed. Expected 'off', 'single', or 'stripped'." >&2
        exit 1
    fi

    
    # setup ETCD/Minio info which all parts will need
    if [[ ! -d ./configs ]]; then
        cp -r "${BASE_DIR}/${MILVUS_CONFIG_DIR}/configs/" .
    fi
    rm -f ./configs/milvus.yaml
    python3 replace_unified.py --mode distributed

    # Launch coordinator nodes
    launch_role COORDINATOR "$COORDINATOR_NODES" "$COORDINATOR_NODES_PER_CN" "$STORAGE_MEDIUM" ./launch_milvus_part.sh
    
    # Launch streaming nodes
    launch_role STREAMING "$STREAMING_NODES" "$STREAMING_NODES_PER_CN" "$STORAGE_MEDIUM" ./launch_milvus_part.sh

    # Launch query nodes
    launch_role QUERY "$QUERY_NODES" "$QUERY_NODES_PER_CN" "$STORAGE_MEDIUM" ./launch_milvus_part.sh

    # Launch data nodes
    launch_role DATA "$DATA_NODES" "$DATA_NODES_PER_CN" "$STORAGE_MEDIUM" ./launch_milvus_part.sh
    
    # Launch proxy
    launch_role PROXY "$NUM_PROXIES" "$NUM_PROXIES_PER_CN" "$STORAGE_MEDIUM" ./launch_milvus_part.sh

    SIGNAL_FILES=()
    for spec in \
        "cord:$COORDINATOR_NODES" \
        "query:$QUERY_NODES" \
        "data:$DATA_NODES" \
        "streaming:$STREAMING_NODES" \
        "proxy:$NUM_PROXIES"
    do
        IFS=":" read -r name count <<< "$spec"

        for ((rank=0; rank<count; rank++)); do
            SIGNAL_FILES+=("$RUNTIME_STATE_DIR/${name}${rank}_running.txt")
        done
    done

    # execute.sh only writes these markers after each component passes its health check.
    wait_for_signal_files "${SIGNAL_FILES[@]}"
    
    IP_ADDR=$(jq -r '.hsn0.ipv4[0]' PROXY/PROXY0.json)
    echo "$IP_ADDR" > "$RUNTIME_STATE_DIR/worker.ip"
    sleep 30
    
    env "${PYTHON_ENV_VARS[@]}" python3 poll.py
fi




should_summarize_insert=0
should_summarize_query=0

# if we are not restoring, run insert and/or indexing
if [ -z "$RESTORE_DIR" ]; then
    env "${PYTHON_ENV_VARS[@]}" python3 setup_collection.py

    export ACTIVE_TASK="INSERT"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client
    should_summarize_insert=1
    mkdir -p uploadNPY
    mv *.npy uploadNPY/

    if [[ "$TASK" == "INSERT" || "$TASK" == "IMPORT" ]]; then
        touch "$RUNTIME_STATE_DIR/flag.txt"
    fi

    if [[ "$TASK" == "INDEX" || "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
        export ACTIVE_TASK="INDEX"
        if [[ "$TASK" == "INDEX" ]]; then
            touch "$RUNTIME_STATE_DIR/workflow_start.txt"
        fi
        env "${PYTHON_ENV_VARS[@]}" python3 index.py
        
        if [[ "$TASK" == "INDEX" ]]; then
            touch "$RUNTIME_STATE_DIR/workflow_end.txt"
            touch "$RUNTIME_STATE_DIR/flag.txt"
        fi
    fi
    sleep 30
else
    env "${PYTHON_ENV_VARS[@]}" python3 status.py
fi

if [[ "$TASK" == "QUERY" ]]; then
    
    export ACTIVE_TASK="QUERY"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client
    should_summarize_query=1
    mkdir -p queryNPY
    mv *.npy queryNPY/
    

    if [[ "$TRACING" == "True" ]]; then
        touch "$RUNTIME_STATE_DIR/flag.txt"
        while [[ ! -f traces.jsonl ]]; do
            sleep 1
        done
    fi

fi 

if [[ "$TASK" == "MIXED" ]]; then
    if [[ -z "${MIXED_RESULT_PATH:-}" ]]; then
        export MIXED_RESULT_PATH="$BASE_DIR/$myDIR/mixed_logs"
    elif [[ "$MIXED_RESULT_PATH" != /* ]]; then
        export MIXED_RESULT_PATH="$BASE_DIR/$myDIR/$MIXED_RESULT_PATH"
    else
        export MIXED_RESULT_PATH="$MIXED_RESULT_PATH"
    fi

    export INSERT_START_ID=$INSERT_START_ID
    COLLECTION_NAME="${COLLECTION_NAME:-standalone}"
    VECTOR_FIELD="${VECTOR_FIELD:-vector}"
    ID_FIELD="${ID_FIELD:-id}"
    TOP_K="${TOP_K:-10}"
    export EFSearch=$QUERY_EF_SEARCH
    export MIXED_INSERT_BATCH_MIN=${MIXED_INSERT_BATCH_MIN:-$INSERT_BATCH_MIN}
    export MIXED_INSERT_BATCH_MAX=${MIXED_INSERT_BATCH_MAX:-$INSERT_BATCH_MAX}
    export MIXED_QUERY_BATCH_MIN=${MIXED_QUERY_BATCH_MIN:-$QUERY_BATCH_MIN}
    export MIXED_QUERY_BATCH_MAX=${MIXED_QUERY_BATCH_MAX:-$QUERY_BATCH_MAX}
    export MIXED_INSERT_CLIENTS
    export MIXED_QUERY_CLIENTS
    mkdir -p "$MIXED_RESULT_PATH"

    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./mixed

    MIXED_TIMELINE_METRIC="dot"
    if [[ "$DISTANCE_METRIC" == "COSINE" ]]; then
        MIXED_TIMELINE_METRIC="cosine"
    elif [[ "$DISTANCE_METRIC" == "L2" ]]; then
        MIXED_TIMELINE_METRIC="l2"
    fi

    MIXED_TIMELINE_ARGS=(
        mixed_timeline.py
        --log-dir "$MIXED_RESULT_PATH"
        --insert-vectors "$MIXED_INSERT_DATA_FILEPATH"
        --query-vectors "$MIXED_QUERY_DATA_FILEPATH"
        --metric "$MIXED_TIMELINE_METRIC"
        --insert-id-offset "$INSERT_START_ID"
        --throughput-only
    )
    if [[ -n "$MIXED_INSERT_CORPUS_SIZE" ]]; then
        MIXED_TIMELINE_ARGS+=(--insert-max-rows "$MIXED_INSERT_CORPUS_SIZE")
    fi
    if [[ -n "$MIXED_QUERY_CORPUS_SIZE" ]]; then
        MIXED_TIMELINE_ARGS+=(--query-max-rows "$MIXED_QUERY_CORPUS_SIZE")
    fi

    if [[ -z "$RESTORE_DIR" ]]; then
        MIXED_TIMELINE_ARGS+=(--init-vectors "$INSERT_DATA_FILEPATH")
        if [[ -n "$INSERT_CORPUS_SIZE" ]]; then
            MIXED_TIMELINE_ARGS+=(--init-max-rows "$INSERT_CORPUS_SIZE")
        fi
    fi

    env "${PYTHON_ENV_VARS[@]}" python3 "${MIXED_TIMELINE_ARGS[@]}"
fi


(( should_summarize_insert )) && run_summary INSERT insert
(( should_summarize_query ))  && run_summary QUERY query

mkdir -p client_timings
mv *.csv *.txt client_timings/
mv ./configs/ runtime_state/
mv *.out *.json runtime_state/

if [[ "$TRACING" == "True" ]]; then
        mv *.jsonl runtime_state/
        python3 analyze_traces.py > analysis.txt
fi


if [[ "${AUTO_CLEANUP}" ]]; then
    if [[ "$STORAGE_MEDIUM" == "DAOS" || "$ETCD_MEDIUM" == "DAOS" || "$MINIO_MEDIUM" == "DAOS" ]]; then
        DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when a Milvus storage medium uses DAOS}"
        DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when a Milvus storage medium uses DAOS}"
        rm -fr /tmp/${DAOS_POOL}/${DAOS_CONT}/$myDIR
    elif [[ "$STORAGE_MEDIUM" == "lustre" || "$MODE" == "DISTRIBUTED" ]]; then
        rm -fr ./milvusDir/
    fi
fi

calculate_recall_if_enabled
