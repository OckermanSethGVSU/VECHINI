#!/bin/bash

STORAGE_MEDIUM=${1:?Usage: $0 <storage_medium>}
TYPE=${2:?Usage: $0 <storage_medium> <type>}
RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-./runtime_state}"
mkdir -p "$RUNTIME_STATE_DIR"

RANK="${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-}}}"

if [[ "$STORAGE_MEDIUM" == "memory" ]]; then
    TARGET_BASE="/dev/shm/milvusDir"

DAOS_ARGS=()
elif [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
    DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when STORAGE_MEDIUM=DAOS}"
    DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when STORAGE_MEDIUM=DAOS}"
    TARGET_BASE="/tmp/${DAOS_POOL}/${DAOS_CONT}/${myDIR}/milvusDir"

elif [[ "$STORAGE_MEDIUM" == "lustre" ]]; then
    TARGET_BASE="./milvusDir"

elif [[ "$STORAGE_MEDIUM" == "SSD" ]]; then
    TARGET_BASE="/local/scratch/milvusDir"
else
    (( RANK == 0 )) && echo "Error: unknown STORAGE_MEDIUM '$STORAGE_MEDIUM'" >&2
    exit 1
fi

mkdir -p ${TYPE}

python3 net_mapping.py --rank ${RANK} --name ${TYPE}
MY_IP_ADDR=$(jq -er '.hsn0.ipv4[0]' "${TYPE}${RANK}.json")
mv "${TYPE}${RANK}.json" ${TYPE}/

sleep $((RANK * 5))

BLOCK=8
case "$TYPE" in
  COORDINATOR)        BASE=20000 ;;
  PROXY)              BASE=20001 ;;
  INTERNAL_PROXY)     BASE=20002 ;;
  COORDINATOR_QUERY)  BASE=20003 ;;
  QUERY)              BASE=20004 ;;
  COORDINATOR_DATA)   BASE=20005 ;;
  DATA)               BASE=20006 ;;
  STREAMING)          BASE=20007 ;;
  *)
    echo "Unknown MODE='$MODE' (expected COORDINATOR|PROXY|INTERNAL_PROXY|COORDINATOR_QUERY|QUERY|COORDINATOR_DATA|DATA|STREAMING)" >&2
    exit 1
    ;;
esac
# Compute port = base + BLOCK*rank
PORT=$(( BASE + BLOCK * RANK ))

METRICS_BASE_BLOCK=30000
METRICS_BASE=$(( METRICS_BASE_BLOCK + (BASE - 20000) ))
METRICS_PORT=$(( METRICS_BASE + BLOCK * RANK ))
if (( METRICS_PORT > 65535 )); then
  echo "ERROR: METRICS_PORT=$METRICS_PORT out of range" >&2
  exit 2
fi

OUTPUT_FILE="./${TYPE}/${TYPE}_registry.txt"
echo "${RANK},${MY_IP_ADDR},${PORT},${METRICS_PORT}" >> $OUTPUT_FILE


# create configuration for micro-service component (basically just set its IP in the config file)
if [[ -z "$RESTORE_DIR" ]]; then
    rm -fr $TARGET_BASE/${TYPE}${RANK}/
fi
mkdir -p $TARGET_BASE/${TYPE}${RANK}/
if [[ "${MINIO_MODE:-}" == "off" && -n "${LOCAL_SHARED_STORAGE_PATH:-}" ]]; then
    mkdir -p "${LOCAL_SHARED_STORAGE_PATH}"
fi

python3 replace_unified.py --mode ${TYPE} --rank $RANK
cp -r ./configs/ $TARGET_BASE/${TYPE}${RANK}/
cp $TARGET_BASE/${TYPE}${RANK}/configs/${TYPE}${RANK}.yaml $TARGET_BASE/${TYPE}${RANK}/configs/milvus.yaml

GPU_ARGS=()
if [[ "$GPU_INDEX" == "True" ]]; then

    GPU_ARGS+=()
else
    GPU_ARGS+=(
        --env CUDA_VISIBLE_DEVICES="" 
    )
fi

base=${BASE_DIR}

POLARIS_BINDS=()
if [[ "$PLATFORM" == "POLARIS" ]]; then
    POLARIS_BINDS+=(
        -B "/opt/nvidia/hpc_sdk:/opt/nvidia/hpc_sdk:ro"
        --env PLATFORM=POLARIS
    )
 
fi

BUILD_ARGS=()
if [ -n "$MILVUS_BUILD_DIR" ]; then
    BUILD_ARGS+=(
        -B ${base}/${MILVUS_BUILD_DIR}/:/milvus/
        -B /usr/lib64/libatomic.so.1:/usr/lib64/libatomic.so.1
    )
fi 

SHARED_STORAGE_ARGS=()
if [[ "${MINIO_MODE:-}" == "off" && -n "${LOCAL_SHARED_STORAGE_PATH:-}" ]]; then
    SHARED_STORAGE_ARGS+=(
        -B "${LOCAL_SHARED_STORAGE_PATH}:${LOCAL_SHARED_STORAGE_PATH}"
        --env COMMON_STORAGETYPE=local
    )
else
    SHARED_STORAGE_ARGS+=(
        --env COMMON_STORAGETYPE=remote
    )
fi


apptainer exec --fakeroot \
  --writable-tmpfs \
  --pwd /milvus \
  --env TYPE=$TYPE \
  --env MILVUSCONF=/milvus/configs/ \
  --env METRICS_PORT=$METRICS_PORT \
  --env MILVUS_HEALTH_HOST=$MY_IP_ADDR \
  --env MILVUS_HEALTH_PORT=$METRICS_PORT \
  --env RUNTIME_STATE_DIR=/runtime_state \
  --env LOCAL_SHARED_STORAGE_PATH="${LOCAL_SHARED_STORAGE_PATH:-}" \
  --env PERF=$PERF \
  --env PERF_EVENTS=$PERF_EVENTS \
  -B ./execute.sh:/milvus/app_execute.sh \
  -B "${RUNTIME_STATE_DIR}:/runtime_state/" \
  -B $TARGET_BASE/${TYPE}${RANK}/:/var/lib/milvus \
  -B $TARGET_BASE/${TYPE}${RANK}/configs/:/milvus/configs/ \
   "${BUILD_ARGS[@]}" \
   "${SHARED_STORAGE_ARGS[@]}" \
   "${POLARIS_BINDS[@]}" \
  "${GPU_ARGS[@]}" \
  milvus.sif bash app_execute.sh $RANK > ${TYPE}/${TYPE}${RANK}.out 2>&1
