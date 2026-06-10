#!/bin/bash

RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-/runtime_state}"
PERF_BIN="${PERF_BIN:-$RUNTIME_STATE_DIR/perf}"
DEFAULT_PERF_STAT_EVENTS="cycles,instructions,branches,branch-misses,cache-misses"

if [[ "$PERF" == "STAT" || "$PERF" == "RECORD" ]]; then
    if [[ ! -x "$PERF_BIN" ]]; then
        echo "Perf mode $PERF requires an executable at $PERF_BIN" >&2
        exit 1
    fi
    apt-get update && apt-get install -y libatomic1 zlib1g-dev libzstd-dev liblzma-dev libelf-dev libdw-dev libslang2-dev libnuma-dev libtraceevent-dev curl
    export PATH="$RUNTIME_STATE_DIR:$PATH"
fi

healthcheck() {
    local host=$1
    local port=$2
    local status

    exec 3<>"/dev/tcp/${host}/${port}" || return 1
    printf 'GET /healthz HTTP/1.1\r\nHost: %s\r\nConnection: close\r\n\r\n' "$host" >&3
    IFS=$'\r' read -r status <&3 || {
        exec 3>&-
        exec 3<&-
        return 1
    }
    exec 3>&-
    exec 3<&-

    [[ "$status" == *" 200 "* ]]
}

is_truth() {
    local value="${1:-}"
    shopt -s nocasematch
    if [[ "$value" =~ ^(1|true|yes|y|on)$ ]]; then
        shopt -u nocasematch
        return 0
    fi
    shopt -u nocasematch
    return 1
}

QDRANT_LAUNCH_ENV=(
    NO_PROXY=""
    no_proxy=""
    http_proxy=""
    https_proxy=""
    HTTP_PROXY=""
    HTTPS_PROXY=""
)

if is_truth "$QUERY_TRACE"; then
    QDRANT_LAUNCH_ENV+=(QDRANT_QUERY_PATH=1)
fi

if is_truth "$INSERT_TRACE"; then
    QDRANT_LAUNCH_ENV+=(QDRANT_INSERT_PATH=1)
fi

IP_ADDR=$1
P2P_PORT=$2
RANK=$3
USEPERF=$4

if [[ $RANK -eq 0 ]]; then
  while true; do
    env "${QDRANT_LAUNCH_ENV[@]}" \
    ./qdrant --uri "http://${IP_ADDR}:${P2P_PORT}" --config-path /qdrant/config/config.yaml &
    QDRANT_PID=$!
    HTTP_PORT=$((P2P_PORT - 2))

    healthy=false
    for i in {1..30}; do
        if NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" healthcheck "$IP_ADDR" "$HTTP_PORT"; then
            echo "Rank ${RANK} qdrant ${IP_ADDR}:${P2P_PORT} is healthy"
            healthy=true
            touch /runtime_state/qdrant_running${RANK}.txt
            break
        fi
        sleep 1
    done

    if [[ "$healthy" == true ]]; then
        break
    fi

    echo "Rank ${RANK} qdrant ${IP_ADDR}:${P2P_PORT} failed to become healthy, restarting..."
    kill "$QDRANT_PID" 2>/dev/null || true
    wait "$QDRANT_PID" 2>/dev/null || true
    sleep 5
  done
else
  # wait until the first rank is online
  TARGET="/runtime_state/qdrant_running0.txt"
  while [ ! -e "$TARGET" ]; do
    sleep 0.1
  done

  bootstrapIP=$(cut -d',' -f2 /qdrant/ip_registry.d/0)

  while true; do
    env "${QDRANT_LAUNCH_ENV[@]}" \
    ./qdrant --uri "http://${IP_ADDR}:${P2P_PORT}" --bootstrap http://$bootstrapIP:6335 --config-path /qdrant/config/config.yaml &
    QDRANT_PID=$!
    HTTP_PORT=$((P2P_PORT - 2))
    
    # Wait for health
    healthy=false
    for i in {1..30}; do
        if NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" healthcheck "$IP_ADDR" "$HTTP_PORT"; then
            echo "Rank ${RANK} qdrant ${IP_ADDR}:${P2P_PORT} is healthy"
            healthy=true
            touch /runtime_state/qdrant_running${RANK}.txt
            break
        fi
        sleep 1
    done

     if [[ "$healthy" == true ]]; then
      echo "Rank ${RANK} qdrant ${IP_ADDR}:${P2P_PORT} is healthy"
        break   # 🚀 breaks out of the OUTER while loop
    fi

    echo "Rank ${RANK} qdrant ${IP_ADDR}:${P2P_PORT} failed to become healthy, restarting..."
    kill "$QDRANT_PID" 2>/dev/null || true
    wait "$QDRANT_PID" 2>/dev/null || true
    sleep 5
  done 
 

fi

# wait until the file exists
TARGET="/runtime_state/workflow_start.txt"
while [ ! -e "$TARGET" ]; do
  sleep 0.1
done


if [[ "$PERF" == "RECORD" ]]; then
    echo "Rank ${RANK} Launching perf record"
    "$PERF_BIN" record -F 99 --call-graph fp -g --proc-map-timeout 5000 -o "$RUNTIME_STATE_DIR/perf${RANK}.data" -p "$QDRANT_PID" &
    PERF_PID=$!

elif [[ "$PERF" == "STAT" ]]; then
    echo "Rank ${RANK} Launching perf stat"
    PERF_STAT_EVENTS="${PERF_EVENTS:-$DEFAULT_PERF_STAT_EVENTS}"
    "$PERF_BIN" stat -e "$PERF_STAT_EVENTS" -o "$RUNTIME_STATE_DIR/perf${RANK}.data" -p "$QDRANT_PID" &
    PERF_PID=$!
fi


# wait until the file exists
TARGET="/runtime_state/workflow_stop.txt"
while [ ! -e "$TARGET" ]; do
  sleep 0.1
done

# stop perf cleanly (same as Ctrl-C)
if [[ "$PERF" == "RECORD" || "$PERF" == "STAT" ]]; then
    if kill -0 "$PERF_PID" 2>/dev/null; then
        echo "Rank ${RANK} stopping perf"
        kill -INT "$PERF_PID"
    fi
    if ! wait "$PERF_PID"; then
        echo "Rank ${RANK} perf exited with an error" >&2
    fi
fi


# wait until our main script signals it is safe to close
TARGET="/runtime_state/flag.txt"
while [ ! -e "$TARGET" ]; do
  sleep 0.1
done

echo "Rank ${RANK} closing"
kill "$QDRANT_PID" 2>/dev/null || true
wait "$QDRANT_PID" 2>/dev/null || true
