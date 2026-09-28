#!/usr/bin/env bash
# Sourced by both launchers from the staged run directory. Container runtimes
# accept the same --env arguments; each launcher supplies its own bind syntax.
WEAVIATE_TRACE_ARGS=()
WEAVIATE_TRACE_HOST_DIR=""
for trace_setting in \
    EXPERIMENTAL_OTEL_ENABLED \
    EXPERIMENTAL_OTEL_SERVICE_NAME \
    EXPERIMENTAL_OTEL_ENVIRONMENT \
    EXPERIMENTAL_OTEL_EXPORTER_OTLP_PROTOCOL \
    EXPERIMENTAL_OTEL_EXPORTER_OTLP_ENDPOINT \
    EXPERIMENTAL_OTEL_TRACES_SAMPLER_ARG \
    EXPERIMENTAL_OTEL_BSP_EXPORT_TIMEOUT \
    EXPERIMENTAL_OTEL_BSP_MAX_EXPORT_BATCH_SIZE \
    OTEL_BSP_MAX_QUEUE_SIZE; do
    if [[ -n "${!trace_setting:-}" ]]; then
        WEAVIATE_TRACE_ARGS+=(--env "$trace_setting=${!trace_setting}")
    fi
done

if [[ "${EXPERIMENTAL_OTEL_ENABLED:-false}" == "true" && "${EXPERIMENTAL_OTEL_EXPORTER_OTLP_PROTOCOL:-file}" == "file" ]]; then
    trace_file="${EXPERIMENTAL_OTEL_EXPORTER_FILE:-}"
    if [[ -z "$trace_file" ]]; then
        trace_file='traces/node{rank}.jsonl'
    fi
    trace_file="${trace_file//\{rank\}/$RANK}"
    mkdir -p "$(dirname "$trace_file")" || return 1
    WEAVIATE_TRACE_HOST_DIR="$(cd "$(dirname "$trace_file")" && pwd)" || return 1
    WEAVIATE_TRACE_ARGS+=(--env "EXPERIMENTAL_OTEL_EXPORTER_FILE=/weaviate-traces/$(basename "$trace_file")")
fi
