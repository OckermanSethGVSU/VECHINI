run_summary() {
    local task="$1"     # INSERT or QUERY
    local prefix="$2"   # insert or query
    ACTIVE_TASK="$task" python3 multi_client_summary.py
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

    python3 ./compute_recall.py \
        "$GROUND_TRUTH_FILEPATH" \
        "${query_id_files[@]}" \
        --top-k "${TOP_K:-10}" \
        --output recall.csv
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



RUN_DIR="${RUN_DIR:-$SCRIPT_DIR}"
if [[ -z "${BASE_DIR:-}" ]]; then
    BASE_DIR="$(dirname "$RUN_DIR")"
fi

echo "[INFO] Using run directory: $RUN_DIR"
cd "$RUN_DIR"
rm -f ./runtime_state/workflow_start.txt ./runtime_state/workflow_end.txt \
    ./runtime_state/weaviate_running*.txt ./runtime_state/flag.txt


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

    if [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
        DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when STORAGE_MEDIUM=DAOS}"
        DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when STORAGE_MEDIUM=DAOS}"
        module use /soft/modulefiles
        module load daos
        
        launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
        mkdir -p /tmp/${DAOS_POOL}/${DAOS_CONT}/$myDIR
    fi
fi

if [[ -n "${ENV_PATH:-}" ]]; then
    echo "Activating Python environment: $ENV_PATH"
    source "$ENV_PATH/bin/activate"
else
    echo "ENV_PATH not set; using current Python environment: $(command -v python3)"
fi



TOTAL=$((NODES * WORKERS_PER_NODE))
MAX_RANK=$((TOTAL - 1))
export N_WORKERS=$TOTAL
# Use unique worker hosts (skip first PBS node reserved for client)
tail -n +2 "$PBS_NODEFILE" | awk '!seen[$0]++' > worker_nodefile.txt
cat $PBS_NODEFILE > all_nodefile.txt
WORKER_HOSTS=$(paste -sd, worker_nodefile.txt)

# ---------------------------------------------------------------
# Launch Weaviate cluster via MPI
# ---------------------------------------------------------------
if [[ -z "${CORES:-}" ]]; then
    echo "Launching Weaviate without CPU binding"
    mpirun -n $TOTAL --ppn $WORKERS_PER_NODE --no-vni \
     --cpu-bind none --host "$WORKER_HOSTS" \
    ./launchWeaviateNode.sh "$STORAGE_MEDIUM" "$TOTAL" &
    
else
    echo "Binding Weaviate bound to $CORES cores each. ${TOTAL} workers, ${WORKERS_PER_NODE} workers per node. ${CORES} split among each node's workers."
    mpirun -n $TOTAL --ppn $WORKERS_PER_NODE --no-vni \
     --cpu-bind depth -d $CORES --host "$WORKER_HOSTS" \
    ./launchWeaviateNode.sh "$STORAGE_MEDIUM" "$TOTAL" &
fi
MPI_PID=$!

echo "[INFO] Waiting for ${TOTAL} Weaviate workers to become ready..."
for r in $(seq 0 "${MAX_RANK}"); do
    target="./runtime_state/weaviate_running${r}.txt"
     while [[ ! -e "${target}" ]]; do
        sleep 0.5
     done 
done
echo "[INFO] All ${TOTAL} workers are ready"

# Launch profiling on each node
allNodes=$((NODES + 1))
for ((i=0; i<allNodes; i++)); do
    
    # +1 b/c it uses 1 indexing
    line_num=$((i + 1))
    entry=$(sed -n "${line_num}p" "$PBS_NODEFILE")
    echo "Launching profiling for node ${i}"

    if [[ $i -eq 0 ]]; then
        profile_arg="client_node"
    else
        profile_arg="worker_$((i - 1))"
    fi

    mpirun -n 1 --ppn 1 --cpu-bind none --host $entry python3 profile.py $profile_arg $PLATFORM &
    sleep 1
    
done

NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 health_check.py
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
touch ./runtime_state/insert_index_done.event

if [[ "$TASK" == "INSERT" || "$TASK" == "INDEX" ]]; then
        touch "runtime_state/flag.txt"
fi

if [[ "$TASK" == "QUERY" ]]; then
    export ACTIVE_TASK="QUERY"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client

    mkdir -p queryNPY
    mv *.npy queryNPY/

    touch ./runtime_state/query_done.event
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
calculate_recall_if_enabled
