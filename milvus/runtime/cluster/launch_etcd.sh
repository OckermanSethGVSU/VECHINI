#!/bin/bash

ETCD_MEDIUM=${1:?Usage: $0 <etcd_medium>}
RANK="${PMI_RANK:-${PMIX_RANK:-${OMPI_COMM_WORLD_RANK:-}}}"
RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-./runtime_state}"
mkdir -p "$RUNTIME_STATE_DIR"

if [[ "$ETCD_MEDIUM" == "memory" ]]; then
    TARGET_BASE="/dev/shm/"
    ETCD_BASE=$TARGET_BASE
    
    (( RANK == 0 )) && echo "ETCD using memory for persistence"

DAOS_ARGS=()
elif [[ "$ETCD_MEDIUM" == "DAOS" ]]; then
    DAOS_POOL="${DAOS_PROJECT:?DAOS_PROJECT is required when ETCD_MEDIUM=DAOS}"
    DAOS_CONT="${DAOS_CONTAINER:?DAOS_CONTAINER is required when ETCD_MEDIUM=DAOS}"
    TARGET_BASE="/tmp/${DAOS_POOL}/${DAOS_CONT}/${myDIR}/milvusDir"
    (( RANK == 0 )) && echo "ETCD using DAOS for persistence"
    ETCD_BASE=./milvusDir/


elif [[ "$ETCD_MEDIUM" == "lustre" ]]; then
    TARGET_BASE="./milvusDir"
    ETCD_BASE=$TARGET_BASE

    (( RANK == 0 )) && echo "ETCD using lustre for persistence"
elif [[ "$ETCD_MEDIUM" == "SSD" ]]; then
    TARGET_BASE="/local/scratch/milvusDir"
    ETCD_BASE=$TARGET_BASE

    (( RANK == 0 )) && echo "ETCD using SSD for persistence"

else
    (( RANK == 0 )) && echo "Error: unknown ETCD_MEDIUM '$ETCD_MEDIUM'" >&2
    exit 1
fi

# ---- Per-rank ports to avoid collisions when colocated ----
CLIENT_PORT=$((2379 + 100 * RANK))
PEER_PORT=$((2380 + 100 * RANK))

# ---- Discover per-rank IP ----
python3 net_mapping.py --rank "$RANK" --name etcd
IP_ADDR=$(jq -r '.hsn0.ipv4[0]' "etcd${RANK}.json")
mkdir -p etcdFiles/
mv etcd${RANK}.json etcdFiles/

if [[ -z "$IP_ADDR" || "$IP_ADDR" == "null" ]]; then
  echo "ERROR: Failed to extract IP for rank=$RANK from etcd${RANK}.json" >&2
  cat "etcd${RANK}.json" >&2 || true
  exit 1
fi

# slight stagger
sleep $((RANK * 2))

OUTPUT_FILE="./etcdFiles/etcd_registry.txt"

# ----- Mode-dependent registry + expected size -----
if [[ "$ETCD_MODE" == "single" ]]; then
  if [[ "$RANK" -ne 0 ]]; then
    echo "ETCD_MODE=single: rank $RANK not launching etcd (rank0 only)."
    exit 0
  fi

  rm -f "$OUTPUT_FILE"
  echo "0,${IP_ADDR},${CLIENT_PORT},${PEER_PORT}" >> "$OUTPUT_FILE"
  expected=1

else
  # replicated: ranks 0..2 participate. If more ranks were launched accidentally, ignore them.
  if (( RANK > 2 )); then
    echo "ETCD_MODE=replicated: rank $RANK > 2 ignoring (only ranks 0..2 launch etcd)."
    exit 0
  fi

  if [[ "$RANK" -eq 0 ]]; then
    rm -f "$OUTPUT_FILE"
  fi

  # small stagger to reduce write races
  sleep $((RANK * 1))
  echo "${RANK},${IP_ADDR},${CLIENT_PORT},${PEER_PORT}" >> "$OUTPUT_FILE"
  expected=3
fi

# ----- Wait for expected entries -----
for _ in $(seq 1 90); do
  lines=$(awk 'NF' "$OUTPUT_FILE" 2>/dev/null | wc -l || true)
  if [[ "$lines" -ge "$expected" ]]; then
    break
  fi
  sleep 1
done

lines=$(awk 'NF' "$OUTPUT_FILE" 2>/dev/null | wc -l || true)
if [[ "$lines" -lt "$expected" ]]; then
  echo "ERROR: timed out waiting for $expected etcd registry entries; saw $lines" >&2
  echo "Current $OUTPUT_FILE:" >&2
  cat "$OUTPUT_FILE" >&2 || true
  exit 1
fi

# ----- Build INITIAL_CLUSTER depending on mode -----
if [[ "$ETCD_MODE" == "single" ]]; then
  INITIAL_CLUSTER="etcd-0=http://${IP_ADDR}:${PEER_PORT}"
else
  # Pull rank->ip,peer_port
  IP0=$(awk -F, '$1==0{print $2}' "$OUTPUT_FILE")
  PP0=$(awk -F, '$1==0{print $4}' "$OUTPUT_FILE")
  IP1=$(awk -F, '$1==1{print $2}' "$OUTPUT_FILE")
  PP1=$(awk -F, '$1==1{print $4}' "$OUTPUT_FILE")
  IP2=$(awk -F, '$1==2{print $2}' "$OUTPUT_FILE")
  PP2=$(awk -F, '$1==2{print $4}' "$OUTPUT_FILE")

  if [[ -z "$IP0" || -z "$PP0" || -z "$IP1" || -z "$PP1" || -z "$IP2" || -z "$PP2" ]]; then
    echo "ERROR: missing entries in registry:" >&2
    echo "  r0 ip='$IP0' peer='$PP0'  r1 ip='$IP1' peer='$PP1'  r2 ip='$IP2' peer='$PP2'" >&2
    cat "$OUTPUT_FILE" >&2 || true
    exit 1
  fi

  INITIAL_CLUSTER="etcd-0=http://${IP0}:${PP0},etcd-1=http://${IP1}:${PP1},etcd-2=http://${IP2}:${PP2}"
fi

# ----- Per-rank volume -----
ETCD_VOL="$ETCD_BASE/volumes/etcd_volume_${RANK}"
if [[ -z "$RESTORE_DIR" ]]; then
  rm -rf "$ETCD_VOL"
fi
mkdir -p "$ETCD_VOL"
ETCD_NAME="etcd-${RANK}"

INITIAL_CLUSTER_STATE="new"
if [[ -n "$RESTORE_DIR" ]]; then
  INITIAL_CLUSTER_STATE="existing"
fi



# ---- Launch etcd ----
apptainer exec --fakeroot \
  --writable-tmpfs \
  --env http_proxy= --env https_proxy= --env HTTP_PROXY= --env HTTPS_PROXY= \
  --env NO_PROXY= --env no_proxy= \
  --env ETCD_AUTO_COMPACTION_MODE=revision \
  --env ETCD_AUTO_COMPACTION_RETENTION=1000 \
  --env ETCD_QUOTA_BACKEND_BYTES=4294967296 \
  -B "$ETCD_VOL:/etcd" \
  "${APPTAINER_ARGS[@]}" \
  etcd_v3.5.18.sif \
    etcd \
      --name "$ETCD_NAME" \
      --advertise-client-urls "http://${IP_ADDR}:${CLIENT_PORT}" \
      --listen-client-urls "http://0.0.0.0:${CLIENT_PORT}" \
      --initial-advertise-peer-urls "http://${IP_ADDR}:${PEER_PORT}" \
      --listen-peer-urls "http://0.0.0.0:${PEER_PORT}" \
      --initial-cluster "${INITIAL_CLUSTER}" \
      --initial-cluster-state "${INITIAL_CLUSTER_STATE}" \
      --data-dir /etcd \
  > "etcdFiles/etcd${RANK}.out" 2>&1
