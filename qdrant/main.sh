
summarize_standard_run() {
    local task_name="$1"
    local npy_dir="$2"
    [[ -d "$npy_dir" ]] || return 0
    shopt -s nullglob
    local npy_files=("$npy_dir"/*.npy)
    shopt -u nullglob
    (( ${#npy_files[@]} > 0 )) || return 0
    mkdir -p clientTiming
    ACTIVE_TASK="$task_name" python3 summarize_client_timings.py \
        --npy-dir "$npy_dir" \
        --output-dir clientTiming \
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
        "${TOP_K:-10}" \
        --output recall.csv
}

finalize_cluster_run() {
    touch flag.txt
    touch ./runtime_state/flag.txt
    mkdir -p systemStats
    shopt -s nullglob
    local file
    mkdir -p clientTiming
    local timing_files=()
    [[ -f ./index_time.txt ]] && timing_files+=(./index_time.txt)
    timing_files+=(./*_times.csv ./*_summary.csv)
    if (( ${#timing_files[@]} > 0 )); then
        mv "${timing_files[@]}" clientTiming/
    fi
    sleep 30
    for file in ./*_system_*.csv ./*_final.csv; do
        [[ -e "$file" ]] || continue
        mv "$file" systemStats/
    done
    shopt -u nullglob
    rm -f flag.txt
    if [[ -f ./ip_registry.txt ]]; then
        mv ./ip_registry.txt ./runtime_state/
    fi
    if [[ -d ./ip_registry.d ]]; then
        mv ./ip_registry.d ./runtime_state/
    fi
    for file in ./all_nodefile.txt ./worker_nodefile.txt ./config.yaml; do
        [[ -e "$file" ]] || continue
        mv "$file" ./runtime_state/
    done
}

wait_for_launch_stop_flag() {
    touch ./runtime_state/workflow_start.txt ./runtime_state/workflow_stop.txt
    echo "TASK=LAUNCH: Qdrant cluster is up and will stay running until you create flag.txt or runtime_state/flag.txt in this run directory."

    while [[ ! -e flag.txt && ! -e ./runtime_state/flag.txt ]]; do
        sleep 1
    done

    echo "TASK=LAUNCH: stop flag detected; stopping Qdrant cluster."
    touch flag.txt ./runtime_state/flag.txt
    sleep 30
}

# Retry configure_collection.py until it drops ready.flag (create + optional rebalance).
run_configure_collection() {
    local target_file="ready.flag"
    while [[ ! -e "$target_file" ]]; do
        NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 configure_collection.py
        sleep 30
    done
    rm "$target_file"
    sleep 3
}

# Every rank's endpoints, for external query drivers. REST in endpoints.txt;
# gRPC (p2p - 1) in endpoints_grpc.txt -- supernova storm's qdrant target uses
# the Rust client, which speaks gRPC, so its QDRANT_URL wants the 6334-family
# port, not 6333.
write_endpoints_file() {
    awk -F',' '{printf "http://%s:%d\n", $2, $3 - 2}' ip_registry.txt > ./runtime_state/endpoints.txt
    awk -F',' '{printf "http://%s:%d\n", $2, $3 - 1}' ip_registry.txt > ./runtime_state/endpoints_grpc.txt
    echo "Cluster endpoints written to runtime_state/endpoints.txt (REST) and endpoints_grpc.txt (gRPC, for storm)"
}

# Install prebuilt shard-builder artifacts into the running cluster, then relaunch it
# on the installed data. Sequence (verified upstream in the shard-builder README): 
# create the collection from the same document the artifacts were built from 
# -> verify-config (a fingerprint mismatch means a full re-index on the first write, 
# so it aborts before any shard is touched) -> read the shard-to-peer assignment 
# consensus made -> stop every qdrant (shards are read at startup only) -> 
# move/copy each rank's shards into its storage tree -> relaunch ->
# verify count, placement, and a smoke query.
artifact_install_flow() {
    local head_line head_id head_ip head_p2p head_url
    head_line=$(head -n 1 ip_registry.txt)
    IFS=',' read -r head_id head_ip head_p2p <<< "$head_line"
    head_url="http://${head_ip}:$((head_p2p - 2))"
    local sb=./qdrant-shard-builder

    echo "[artifact-install] creating collection ${COLLECTION_NAME} from ${COLLECTION_DOCUMENT}"
    run_configure_collection

    echo "[artifact-install] verify-config against ${head_url}"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        "$sb" verify-config --config "$COLLECTION_DOCUMENT" \
        --url "$head_url" --collection "$COLLECTION_NAME" \
        || { echo "[artifact-install] verify-config FAILED; aborting before any shard is touched." >&2; exit 1; }

    echo "[artifact-install] reading the shard assignment consensus made"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        "$sb" install-plan --url "$head_url" --collection "$COLLECTION_NAME" \
        > ./runtime_state/install_plan.txt \
        || { echo "[artifact-install] install-plan failed" >&2; exit 1; }
    cat ./runtime_state/install_plan.txt
    python3 make_install_map.py --plan ./runtime_state/install_plan.txt \
        --registry ip_registry.txt --output ./runtime_state/install_map.tsv \
        || { echo "[artifact-install] could not map install-plan peers to ranks" >&2; exit 1; }

    echo "[artifact-install] stopping qdrant on all ${TOTAL} ranks"
    touch ./runtime_state/install_ready.txt
    for ((rank=0; rank<TOTAL; rank++)); do
        while [[ ! -e "./runtime_state/qdrant_stopped${rank}.txt" ]]; do sleep 0.5; done
    done

    echo "[artifact-install] installing shards from ${ARTIFACT_DIR} (INSTALL_MODE=${INSTALL_MODE:-move})"
    # Storage is on Lustre (validated at submit time), so the head node installs every
    # rank's shards itself -- a move is a metadata-only rename from any node. Each
    # rank's install_done flag releases its relaunch.
    for ((rank=0; rank<TOTAL; rank++)); do
        ./install_shards.sh "$rank" || { echo "[artifact-install] install failed for rank $rank" >&2; exit 1; }
    done

    echo "[artifact-install] waiting for all ranks to come back healthy on the installed data"
    for ((rank=0; rank<TOTAL; rank++)); do
        while [[ ! -e "./runtime_state/qdrant_running_installed${rank}.txt" ]]; do sleep 1; done
    done

    echo "[artifact-install] verifying the install"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
        python3 verify_restore.py \
        || { echo "[artifact-install] verify_restore FAILED -- do not serve this cluster" >&2; exit 1; }

    if [[ -n "${SCATTER_WORK_DIR:-}" ]]; then
        echo "[artifact-install] verify-placement against the scatter sidecars"
        local url_args=() reg_rank reg_ip reg_p2p
        while IFS=',' read -r reg_rank reg_ip reg_p2p; do
            url_args+=(--url "http://${reg_ip}:$((reg_p2p - 2))")
        done < ip_registry.txt
        NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
            "$sb" verify-placement --config "$COLLECTION_DOCUMENT" \
            --work "$SCATTER_WORK_DIR" --collection "$COLLECTION_NAME" "${url_args[@]}" \
            || { echo "[artifact-install] verify-placement FAILED -- do not serve this cluster" >&2; exit 1; }
    else
        echo "[artifact-install] SCATTER_WORK_DIR not set; skipped verify-placement (count check stands in)"
    fi

    echo "[artifact-install] install verified; cluster is serving the artifacts"
}

if [[ -n "${RUN_DIR:-}" ]]; then
    SCRIPT_DIR="$RUN_DIR"
elif [[ -n "${PBS_O_WORKDIR:-}" ]]; then
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
BASE_DIR="${BASE_DIR:-$(dirname "$RUN_DIR")}"

echo "[INFO] Using run directory: $RUN_DIR"
cd "$RUN_DIR"

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
fi

if [[ -n "${ENV_PATH:-}" ]]; then
    echo "Activating Python environment: $ENV_PATH"
    source "$ENV_PATH/bin/activate"
else
    echo "ENV_PATH not set; using current Python environment: $(command -v python3)"
fi

if [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
    DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when STORAGE_MEDIUM=DAOS}"
    DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when STORAGE_MEDIUM=DAOS}"
    module use /soft/modulefiles
    module load daos
    
    launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
    mkdir -p /tmp/${DAOS_POOL}/${DAOS_CONT}/$myDIR
fi


TOTAL=$((NODES * WORKERS_PER_NODE))
MAX_RANK=$((TOTAL - 1))
export N_WORKERS=$TOTAL

rm -f ip_registry.txt
rm -rf ip_registry.d
> ip_registry.txt

tail -n +2 $PBS_NODEFILE > worker_nodefile.txt
cat $PBS_NODEFILE > all_nodefile.txt

# create configs for each rank, 1 launched per node
mpirun -n $TOTAL --ppn $WORKERS_PER_NODE --cpu-bind none --hostfile worker_nodefile.txt  \
    python3 gen_dirs.py --storage_medium $STORAGE_MEDIUM --path /tmp/${DAOS_POOL}/${DAOS_CONT}/$myDIR --log_level "$LOG_LEVEL"

# Artifact installs need the in-container launch script to run the quiesce/relaunch
# protocol; the flag rides into apptainer via launchQdrantNode.sh.
if [[ -n "${ARTIFACT_DIR:-}" ]]; then
    export ARTIFACT_INSTALL=1
fi

# launch qdrant nodes
for ((i=0; i<NODES; i++)); do
    # +1 b/c it uses 1 indexing and +1 b/c we are using the first node for clients
    line_num=$((i + 2))
    entry=$(sed -n "${line_num}p" "$PBS_NODEFILE")
    for ((j=0; j<WORKERS_PER_NODE; j++)); do
        index=$(((i * WORKERS_PER_NODE) + j))
        if [[ -n "${CORES:-}" ]]; then
            echo "Launching node ${index} with cores ${CORES}"
        else
            echo "Launching node ${index} with cpu-bind none"
        fi

        # Empty CORES means do not request explicit core depth binding.
        if [[ -z "${CORES:-}" ]]; then
            mpirun -n 1 --ppn 1 --cpu-bind none --host $entry ./launchQdrantNode.sh $index $STORAGE_MEDIUM &
        else
            mpirun -n 1 --ppn 1 -d $CORES --cpu-bind depth --host $entry ./launchQdrantNode.sh $index $STORAGE_MEDIUM &
        fi
        sleep 0.5
    done
done



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

# Wait until all of the Qdrant ranks are running
while true; do
  all_running=1
  for ((rank=0; rank<=MAX_RANK; rank++)); do
    if [ ! -e "./runtime_state/qdrant_running${rank}.txt" ]; then
      all_running=0
      break
    fi
  done

  if [[ "$all_running" -eq 1 ]]; then
    break
  fi

  sleep 0.1
done

while true; do
  registry_count=$(find ./ip_registry.d -maxdepth 1 -type f | wc -l)
  if [[ "$registry_count" -eq "$TOTAL" ]]; then
    break
  fi
  sleep 0.1
done

sort -t, -k1,1n ./ip_registry.d/* > ip_registry.txt
echo "Qdrant Cluster setup"
mkdir interfaces
mv interfaces*.json interfaces/

sleep 30

########## Workflow ###############
line=$(head -n 1 ip_registry.txt)
IFS=',' read -r id ip port <<< "$line"
port=$((port - 1))

if [[ "$TASK" == "LAUNCH" ]]; then
    if [[ -n "${ARTIFACT_DIR:-}" ]]; then
        artifact_install_flow
    elif [[ "${CREATE_COLLECTION:-False}" == "True" ]]; then
        run_configure_collection
    fi
    write_endpoints_file
    wait_for_launch_stop_flag
else

if [ -z "$RESTORE_DIR" ]; then

    # Setup the cluster
    run_configure_collection

    export ACTIVE_TASK="INSERT"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client
    
    # tell the profs to close and give them time to do so
    if [[ "$TASK" == "INSERT" ]]; then
        finalize_cluster_run
    fi

    move_standard_npy_files uploadNPY
   
   
    if [[ "$TASK" == "INDEX" ]]; then

        # TODO: parameterize index
        NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 build_index.py
        summarize_standard_run INSERT uploadNPY
        
        finalize_cluster_run
    fi
else
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 collection_status.py
fi


if [[ "$TASK" == "QUERY" ]]; then
    # index the data
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 build_index.py

    export ACTIVE_TASK="QUERY"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./batch_client

    move_standard_npy_files queryNPY
    summarize_standard_run INSERT uploadNPY
    summarize_standard_run QUERY queryNPY

    finalize_cluster_run

fi


if [[ "$TASK" == "MIXED" ]]; then
    # set the insert offset
    if [[ -z "${INSERT_START_ID:-}" ]]; then
        if [[ -n "${RESTORE_DIR:-}" ]]; then
            INSERT_START_ID="$EXPECTED_CORPUS_SIZE"
        elif [[ -n "${INSERT_CORPUS_SIZE:-}" ]]; then
            INSERT_START_ID="$INSERT_CORPUS_SIZE"
        elif [[ -n "${INSERT_DATA_FILEPATH:-}" ]]; then
            if ! INSERT_START_ID="$(python3 npy_inspect.py "$INSERT_DATA_FILEPATH")"; then
                echo "Error: failed to derive INSERT_START_ID from INSERT_DATA_FILEPATH using npy_inspect.py." >&2
                exit 1
            fi
        else
            echo "Error: TASK=MIXED requires INSERT_START_ID, INSERT_CORPUS_SIZE, RESTORE_DIR, or INSERT_DATA_FILEPATH." >&2
            exit 1
        fi
        export INSERT_START_ID
        echo "INSERT_START_ID=$INSERT_START_ID"
    fi

    if [[ -z "$RESTORE_DIR"  ]]; then
        # index the data
        NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 build_index.py
    fi

    export ACTIVE_TASK="MIXED"
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./mixed

    MIXED_TIMELINE_METRIC="dot"
    if [[ "$DISTANCE_METRIC" == "COSINE" ]]; then
        MIXED_TIMELINE_METRIC="cosine"
    elif [[ "$DISTANCE_METRIC" == "L2" ]]; then
        MIXED_TIMELINE_METRIC="l2"
    fi

    MIXED_TIMELINE_ARGS=(
        mixed_timeline.py
        --log-dir "$RESULT_PATH"
        --insert-vectors "$MIXED_INSERT_DATA_FILEPATH"
        --query-vectors "$MIXED_QUERY_DATA_FILEPATH"
        --metric "$MIXED_TIMELINE_METRIC"
        --insert-id-offset "$INSERT_START_ID"
    )
    if [[ -n "$MIXED_INSERT_CORPUS_SIZE" ]]; then
        MIXED_TIMELINE_ARGS+=(
            --insert-max-rows "$MIXED_INSERT_CORPUS_SIZE"
        )
    fi
    if [[ -n "$MIXED_QUERY_CORPUS_SIZE" ]]; then
        MIXED_TIMELINE_ARGS+=(
            --query-max-rows "$MIXED_QUERY_CORPUS_SIZE"
        )
    fi
    if [[ -z "$RESTORE_DIR" ]]; then
        MIXED_TIMELINE_ARGS+=(
            --init-vectors "$INSERT_DATA_FILEPATH"
        )
        if [[ -n "$INSERT_CORPUS_SIZE" ]]; then
            MIXED_TIMELINE_ARGS+=(
                --init-max-rows "$INSERT_CORPUS_SIZE"
                --throughput-only
            )
        fi
    fi
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" python3 "${MIXED_TIMELINE_ARGS[@]}"

    finalize_cluster_run
fi

fi

if [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then

    # techincally optional but still good to do
    clean-dfuse.sh  ${DAOS_POOL}:${DAOS_CONT}
fi

if [[ "$TASK" == "INSERT" ]]; then
    summarize_standard_run INSERT uploadNPY
fi

mkdir workerOut
mv rank*.out workerOut
calculate_recall_if_enabled
