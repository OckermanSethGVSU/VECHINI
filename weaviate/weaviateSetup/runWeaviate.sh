#!/bin/sh

set -u

RUNTIME_STATE_DIR="${RUNTIME_STATE_DIR:-/runtime_state}"
PERF_BIN="${PERF_BIN:-$RUNTIME_STATE_DIR/perf}"
PERF_RANK="${PERF_RANK:-0}"
PERF_STOP_TIMEOUT_SEC="${PERF_STOP_TIMEOUT_SEC:-10}"
DEFAULT_PERF_STAT_EVENTS="cycles,instructions,branches,branch-misses,cache-misses"
PERF_LIBRARY_PATH="${RUNTIME_STATE_DIR}/perf-libs${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

perf_enabled() {
    [ "${PERF:-NONE}" = "STAT" ] || [ "${PERF:-NONE}" = "RECORD" ]
}

stop_perf() {
    elapsed=0
    kill -INT "$PERF_PID" 2>/dev/null || true
    while kill -0 "$PERF_PID" 2>/dev/null && [ "$elapsed" -lt "$PERF_STOP_TIMEOUT_SEC" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    if kill -0 "$PERF_PID" 2>/dev/null; then
        echo "Rank ${PERF_RANK} perf did not stop after SIGINT; sending SIGTERM" >&2
        kill -TERM "$PERF_PID" 2>/dev/null || true
        sleep 1
    fi
    if kill -0 "$PERF_PID" 2>/dev/null; then
        echo "Rank ${PERF_RANK} perf did not stop after SIGTERM; sending SIGKILL" >&2
        kill -KILL "$PERF_PID" 2>/dev/null || true
    fi
}

if perf_enabled; then
    if [ ! -x "$PERF_BIN" ]; then
        echo "Perf mode $PERF requires an executable at $PERF_BIN" >&2
        exit 1
    fi
    if ! LD_LIBRARY_PATH="$PERF_LIBRARY_PATH" "$PERF_BIN" --version; then
        echo "The staged perf executable cannot run in this Weaviate image: $PERF_BIN" >&2
        exit 1
    fi
fi

"$@" &
WEAVIATE_PID=$!

if perf_enabled; then
    while [ ! -e "$RUNTIME_STATE_DIR/workflow_start.txt" ]; do
        if ! kill -0 "$WEAVIATE_PID" 2>/dev/null; then
            wait "$WEAVIATE_PID"
            exit $?
        fi
        sleep 0.1
    done

    if [ "$PERF" = "RECORD" ]; then
        echo "Rank ${PERF_RANK} launching perf record"
        LD_LIBRARY_PATH="$PERF_LIBRARY_PATH" "$PERF_BIN" record \
            -F 99 \
            --call-graph fp \
            -g \
            --proc-map-timeout 5000 \
            -o "$RUNTIME_STATE_DIR/perf${PERF_RANK}.data" \
            -p "$WEAVIATE_PID" &
    else
        echo "Rank ${PERF_RANK} launching perf stat"
        PERF_STAT_EVENTS="${PERF_EVENTS:-$DEFAULT_PERF_STAT_EVENTS}"
        LD_LIBRARY_PATH="$PERF_LIBRARY_PATH" "$PERF_BIN" stat \
            -e "$PERF_STAT_EVENTS" \
            -o "$RUNTIME_STATE_DIR/perf${PERF_RANK}.data" \
            -p "$WEAVIATE_PID" &
    fi
    PERF_PID=$!

    while [ ! -e "$RUNTIME_STATE_DIR/workflow_end.txt" ]; do
        if ! kill -0 "$WEAVIATE_PID" 2>/dev/null; then
            break
        fi
        sleep 0.1
    done

    if kill -0 "$PERF_PID" 2>/dev/null; then
        echo "Rank ${PERF_RANK} stopping perf"
        stop_perf
    fi
    if ! wait "$PERF_PID"; then
        echo "Rank ${PERF_RANK} perf exited with an error" >&2
    fi
fi

wait "$WEAVIATE_PID"
