#!/bin/bash

RANK=$1

# LD_PRELOADS

if [ -n "$MILVUS_BUILD_DIR" ]; then
  export LD_PRELOAD=""
  LIBDIR=/milvus/internal/core/output/lib && \
  export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH}" && \
  export LD_PRELOAD="/usr/lib64/libatomic.so.1 /milvus/internal/core/output/lib/libjemalloc.so"

fi

if [[ "$PLATFORM" == "POLARIS" ]]; then
  export CUDAToolkit_ROOT=/usr/local/cuda
  export CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
  export PATH="/usr/local/cuda/bin:$PATH"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
  export PATH="/usr/local/cuda/bin:$PATH"
fi


MAX_LAUNCH_RETRIES="${MAX_LAUNCH_RETRIES:-5}"
STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-300}"
HEALTH_REQUEST_TIMEOUT_SECONDS="${HEALTH_REQUEST_TIMEOUT_SECONDS:-5}"
HEALTH_CHECK_INTERVAL_SECONDS="${HEALTH_CHECK_INTERVAL_SECONDS:-10}"
HEALTH_HOST="${MILVUS_HEALTH_HOST:-127.0.0.1}"
HEALTH_PORT="${MILVUS_HEALTH_PORT:-${METRICS_PORT:-9091}}"
APT_RETRIES="${APT_RETRIES:-5}"
APT_RETRY_DELAY_SECONDS="${APT_RETRY_DELAY_SECONDS:-10}"
DEFAULT_PERF_STAT_EVENTS="cycles,instructions,topdown-be-bound,topdown-mem-bound,topdown-retiring"
RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-/runtime_state}"
PERF_BIN="${PERF_BIN:-$RUNTIME_STATE_DIR/perf}"

# if Milvus is restoring itself, give it longer to launch
if [ -n "$RESTORE_DIR" ]; then
  STARTUP_TIMEOUT_SECONDS=600
fi


retry_command() {
  local attempts="$1"
  local delay="$2"
  shift 2

  local attempt
  for attempt in $(seq 1 "$attempts"); do
    if "$@"; then
      return 0
    fi

    if [[ "$attempt" -lt "$attempts" ]]; then
      echo "Command failed on attempt ${attempt}/${attempts}: $*"
      sleep "$delay"
    fi
  done

  return 1
}

is_milvus_healthy() {
  local url="http://${HEALTH_HOST}:${HEALTH_PORT}/healthz"
  local response=""

  if command -v curl >/dev/null 2>&1; then
    response=$(
      NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
      curl --silent --show-error --fail --max-time "$HEALTH_REQUEST_TIMEOUT_SECONDS" "$url" 2>/dev/null
    ) || return 1
  elif command -v wget >/dev/null 2>&1; then
    response=$(
      NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
      wget -q -T "$HEALTH_REQUEST_TIMEOUT_SECONDS" -O - "$url" 2>/dev/null
    ) || return 1
  else
    echo "Neither curl nor wget is available for Milvus health checks" >&2
    return 1
  fi

  [[ "${response,,}" == *ok* ]]
}

wait_for_milvus_health() {
  local deadline=$((SECONDS + STARTUP_TIMEOUT_SECONDS))

  while (( SECONDS < deadline )); do
    if ! kill -0 "$MILVUS_PID" 2>/dev/null; then
      echo "Milvus process exited before becoming healthy"
      return 1
    fi

    if is_milvus_healthy; then
      echo "Milvus healthy at http://${HEALTH_HOST}:${HEALTH_PORT}/healthz"
      return 0
    fi

    echo "Waiting for Milvus health: http://${HEALTH_HOST}:${HEALTH_PORT}/healthz"
    sleep "$HEALTH_CHECK_INTERVAL_SECONDS"
  done

  echo "Timed out after ${STARTUP_TIMEOUT_SECONDS}s waiting for Milvus health"
  return 1
}

stop_milvus() {
  if kill -0 "$MILVUS_PID" 2>/dev/null; then
    kill "$MILVUS_PID" 2>/dev/null || true
    wait "$MILVUS_PID" 2>/dev/null || true
  fi
}

launch_milvus_with_retry() {
  local launch_attempt
  for launch_attempt in $(seq 1 "$MAX_LAUNCH_RETRIES"); do
    echo "Launching Milvus (attempt ${launch_attempt}/${MAX_LAUNCH_RETRIES})"
    launch_milvus

    if wait_for_milvus_health; then
      return 0
    fi

    echo "Milvus failed health check on launch attempt ${launch_attempt}/${MAX_LAUNCH_RETRIES}"
    stop_milvus
  done

  return 1
}

install_perf_dependencies() {
  retry_command "$APT_RETRIES" "$APT_RETRY_DELAY_SECONDS" apt-get update || return 1
  retry_command "$APT_RETRIES" "$APT_RETRY_DELAY_SECONDS" \
    apt-get install -y libelf-dev libdw-dev libslang2-dev libperl-dev python3-dev libnuma-dev libtraceevent-dev
}



if [[ "$PERF" == "RECORD" || "$PERF" == "STAT" ]]; then
  if [[ ! -x "$PERF_BIN" ]]; then
    echo "Perf mode $PERF requires an executable at $PERF_BIN" >&2
    exit 1
  fi
  install_perf_dependencies || exit 1
  echo "Runtime and perf dependencies installed; launching Milvus"
else
  echo "Launching Milvus"
fi

launch_milvus() {
  if [[ "$TYPE" == "STANDALONE" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run standalone &
      MILVUS_PID=$!
      SIGNAL_FILE="milvus_running.txt"

  elif [[ "$TYPE" == "COORDINATOR" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run mixcoord &
      MILVUS_PID=$!
      SIGNAL_FILE="cord${RANK}_running.txt"

  elif [[ "$TYPE" == "PROXY" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run proxy &
      MILVUS_PID=$!
      SIGNAL_FILE="proxy${RANK}_running.txt"

  elif [[ "$TYPE" == "DATA" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run datanode &
      MILVUS_PID=$!
      SIGNAL_FILE="data${RANK}_running.txt"

  elif [[ "$TYPE" == "QUERY" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run querynode &
      MILVUS_PID=$!
      SIGNAL_FILE="query${RANK}_running.txt"

  elif [[ "$TYPE" == "STREAMING" ]]; then
    NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
    ./bin/milvus run streamingnode &
      MILVUS_PID=$!
      SIGNAL_FILE="streaming${RANK}_running.txt"
  else
    echo "Unknown TYPE '$TYPE'" >&2
    exit 1
  fi
}

cd /milvus

launch_milvus_with_retry || exit 1

mkdir -p "$RUNTIME_STATE_DIR"
touch "$RUNTIME_STATE_DIR/$SIGNAL_FILE"

# wait until the start file exists
TARGET="$RUNTIME_STATE_DIR/workflow_start.txt"
while [ ! -e "$TARGET" ]; do
  sleep 0.1
done

if [[ "$PERF" == "RECORD" ]]; then
    echo "Rank ${RANK} Launching perf record"
    "$PERF_BIN" record -F 99 --call-graph fp -g --proc-map-timeout 5000 -o "$RUNTIME_STATE_DIR/perf${RANK}.data"  -p "$MILVUS_PID" &
    PERF_PID=$!

elif [[ "$PERF" == "STAT" ]]; then
    echo "Rank ${RANK} Launching perf stat"
    PERF_STAT_EVENTS="${PERF_EVENTS:-$DEFAULT_PERF_STAT_EVENTS}"
    "$PERF_BIN" stat  -e "$PERF_STAT_EVENTS" -o "$RUNTIME_STATE_DIR/perf${RANK}.data"  -p "$MILVUS_PID" &
    PERF_PID=$!
fi

# wait until the stop file exists
TARGET="$RUNTIME_STATE_DIR/workflow_end.txt"
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

echo "Rank ${RANK} done"
sleep 15

# apt-get update && apt-get install -y libatomic1 libelf-dev libdw-dev libslang2-dev libperl-dev python3-dev libnuma-dev libtraceevent-dev
# apt-get install -y \
    # build-essential ca-certificates pkg-config \
    # libssl-dev zlib1g-dev libcurl4-openssl-dev gnupg


# NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" milvus run standalone > /workerOut/standalone.txt 2>&1

# # NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" ./bin/milvus run standalone > /workerOut/standalone.txt 2>&1 &
# NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
#  /runtime_state/perf record  -F 99 -g --call-graph fp --proc-map-timeout 5000 -o /workerOut/perf.data -- ./bin/milvus run standalone > /workerOut/standalone.txt 2>&1 &
# # NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
# # /runtime_state/perf record  -F 99 -g --call-graph fp --proc-map-timeout 5000 -o /workerOut/perf.data -- ./bin/milvus run standalone > /workerOut/standalone.txt 2>&1 &

# NO_PROXY="" no_proxy="" http_proxy="" https_proxy="" HTTP_PROXY="" HTTPS_PROXY="" \
#  /runtime_state/perf record  -F 99 -g --no-buildid-cache --call-graph dwarf --proc-map-timeout 5000 -o /workerOut/perf.data -- ./bin/milvus run standalone > /workerOut/standalone.txt 2>&1 &
# # /runtime_state/perf probe -x milvus -F > probe.txt

# while true; do
#   if [ -f /workerOut/stop.txt ]; then
#     echo "stop.txt detected, sending SIGTERM."
#     # QDRANT_PIDS=$(pgrep -fa './qdrant' | grep -v 'perf'  | awk '{print $1}')
#     PIDS=$(pgrep -fa './milvus' | grep -v 'perf'  | awk '{print $1}')
#     echo "Target PIDs: ${PIDS}"
#     kill -2 $PIDS

#     sleep 60
#     exit

#   fi

# done

# /runtime_state/perf record -F 99 --call-graph dwarf -g --proc-map-timeout 5000 -o /workerOut/perf.data -- ./bin/milvus run standalone &
# & > /workerOut/standalone.txt 2>&1 &
# /runtime_state/perf record -F 99 --call-graph dwarf -g --proc-map-timeout 5000 -o /workerOut/perf.data -- ./bin/milvus run standalone  > /workerOut/standalone.txt 2>&1
