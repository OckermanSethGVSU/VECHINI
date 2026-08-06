#!/bin/bash

# get passed in variables
RANK=${1:?Usage: $0 <rank>}
RANK=$((RANK))
STORAGE_MEDIUM=${2:?Usage: $0 <rank> <storage_medium>}

# get ipv4
python3 mapping.py --rank $RANK
IP_ADDR=$(jq -r '.hsn0.ipv4[0]' interfaces${RANK}.json)
P2P_PORT=$((6335 + RANK * 100))

# register IP,port into file
OUTPUT_DIR="ip_registry.d"
mkdir -p "$OUTPUT_DIR"
printf '%s,%s,%s\n' "$RANK" "$IP_ADDR" "$P2P_PORT" > "${OUTPUT_DIR}/${RANK}"


if [[ "$STORAGE_MEDIUM" == "memory" ]]; then
    TARGET_BASE="/dev/shm/qdrantDir"
    (( RANK == 0 )) && echo "Using memory for persistence"

APPTAINER_ARGS=()
elif [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
    DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when STORAGE_MEDIUM=DAOS}"
    DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when STORAGE_MEDIUM=DAOS}"
    TARGET_BASE="/tmp/${DAOS_POOL}/${DAOS_CONT}/${myDIR}/qdrantDir"
    echo $TARGET_BASE
    (( RANK == 0 )) && echo "Using DAOS for persistence"

elif [[ "$STORAGE_MEDIUM" == "lustre" ]]; then
    TARGET_BASE="./qdrantDir"
    (( RANK == 0 )) && echo "Using lustre for persistence"

elif [[ "$STORAGE_MEDIUM" == "SSD" ]]; then
    TARGET_BASE="/local/scratch/qdrantDir"
    (( RANK == 0 )) && echo "Using SSD for persistence"

else
    (( RANK == 0 )) && echo "Error: unknown STORAGE_MEDIUM '$STORAGE_MEDIUM'" >&2
    exit 1
fi


GPU_ARGS=()
if [[ "$GPU_INDEX" == "True" ]]; then
    GPU_ARGS+=(
        --env QDRANT__GPU__INDEXING=1
        --nv
    )
else
    GPU_ARGS+=(--env QDRANT__GPU__INDEXING=0)

fi

BUILD_ARGS=()
if [ -n "$QDRANT_EXECUTABLE" ]; then
    BUILD_ARGS+=(--bind ./qdrant:/qdrant/qdrant)
fi 

if [ -n "$RESTORE_DIR" ]; then
    rm -fr ${TARGET_BASE}/data/node$RANK
    echo "Restoring from ${RESTORE_DIR} to ${TARGET_BASE}/data/"
    cp -r $RESTORE_DIR/data/node$RANK/ ${TARGET_BASE}/data/
    python3 fix_peer_id.py --path ${TARGET_BASE}/data/node${RANK}/raft_state.json --ip $IP_ADDR --port $P2P_PORT
fi 


# === Launch Qdrant Nodes ===
apptainer exec \
    --fakeroot \
    --writable-tmpfs \
    --pwd /qdrant \
    --bind ./runtime_state/:/runtime_state/ \
    --bind ./ip_registry.txt:/ip_registry.txt \
    --bind ./ip_registry.d:/qdrant/ip_registry.d \
    --bind ./launch.sh:/qdrant/launch.sh \
    --bind ${TARGET_BASE}/data/node$RANK:/qdrant/storage \
    --bind ${TARGET_BASE}/config/node$RANK:/qdrant/config \
    --bind ${TARGET_BASE}/snapshots/node$RANK:/qdrant/snapshots \
    --env PERF=$PERF \
    --env PERF_EVENTS=$PERF_EVENTS \
    --env ARTIFACT_INSTALL=${ARTIFACT_INSTALL:-0} \
    --env INSERT_TRACE=$INSERT_TRACE \
    --env QUERY_TRACE=$QUERY_TRACE \
    "${BUILD_ARGS[@]}" \
    "${APPTAINER_ARGS[@]}" \
    "${GPU_ARGS[@]}" \
    qdrant.sif bash launch.sh $IP_ADDR $P2P_PORT $RANK > "rank${RANK}.out" 2>&1 &
PID=$! 
wait $PID
