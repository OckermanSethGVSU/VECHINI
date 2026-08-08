#!/bin/bash

print_config_summary() {
    echo "            Experiment Configuration"
    echo "================================================="
    echo "Platform:                 $PLATFORM"
    echo "Task:                     $TASK"
    echo "Storage Medium:           $STORAGE_MEDIUM"
    echo "Perf:                     $PERF"
    if [[ "$PERF" == "STAT" ]]; then
        if [[ -n "$PERF_EVENTS" ]]; then
            echo "Perf Events:              $PERF_EVENTS"
        else
            echo "Perf Events:              default"
        fi
    fi
    if [[ -n "${COLLECTION_DOCUMENT:-}" ]]; then
        echo "Collection Document:      $COLLECTION_DOCUMENT (env collection vars ignored)"
    else
        echo "Vector Dim:               $VECTOR_DIM"
        echo "Distance Metric:          $DISTANCE_METRIC"
        echo "Quantization Type:        $QUANTIZATION_TYPE"
    fi
    echo "GPU Index:                $GPU_INDEX"
    echo "Qdrant Version:           ${QDRANT_VERSION:-default}"
    echo "Qdrant SIF:               ${QDRANT_SIF:-n/a}"
    echo "Qdrant Local Image:       ${QDRANT_LOCAL_IMAGE:-qdrant/qdrant:latest}"
    echo "Qdrant Executable:        $QDRANT_EXECUTABLE"
    echo "Insert Trace:             $INSERT_TRACE"
    echo "Query Trace:              $QUERY_TRACE"

    if [[ "${TASK^^}" == "STORM" ]]; then
        echo "Nova Storm Bin:           $NOVA_STORM_BIN"
        echo "Storm Configs:            $STORM_CONFIG"
        echo "Storm Hold After:         ${STORM_HOLD:-False}"
    fi
    if [[ -n "${ARTIFACT_DIR:-}" ]]; then
        echo "Artifact Dir:             $ARTIFACT_DIR (INSTALL_MODE=${INSTALL_MODE:-move} -- move CONSUMES the artifacts)"
        echo "Shard Builder Bin:        $SHARD_BUILDER_BIN"
        echo "Scatter Work Dir:         ${SCATTER_WORK_DIR:-<unset: count check only, no verify-placement>}"
        echo "Expected Corpus Size:     $EXPECTED_CORPUS_SIZE"
    elif [[ -n "$RESTORE_DIR" ]]; then
        echo "Restore Dir:              $RESTORE_DIR"
        echo "Expected Corpus Size:     $EXPECTED_CORPUS_SIZE"
    else
        echo "Insert File:              $INSERT_DATA_FILEPATH"
        echo "Insert Corpus Size:       $INSERT_CORPUS_SIZE"
        echo "Insert Batch Size:        ${INSERT_BATCH_SIZE[*]}"
        echo "Insert Clients/Worker:    $INSERT_CLIENTS_PER_WORKER"
        echo "Insert Balance:           $INSERT_BALANCE_STRATEGY"
        echo "Insert Streaming:         $INSERT_STREAMING"
    fi

    if [[ "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
        echo "Query File:               $QUERY_DATA_FILEPATH"
        echo "Query Corpus Size:        $QUERY_CORPUS_SIZE"
        echo "Query Batch Size:         ${QUERY_BATCH_SIZE[*]}"
        echo "Query Clients/Worker:     $QUERY_CLIENTS_PER_WORKER"
        echo "Total Query Clients:      $TOTAL_QUERY_CLIENTS"
        echo "Query Balance:            $QUERY_BALANCE_STRATEGY"
        echo "Query Streaming:          $QUERY_STREAMING"
    elif [[ "$TASK" == "INDEX" ]]; then
        if [[ -n "$RESTORE_DIR" ]]; then
            echo "Index Corpus Size:        $EXPECTED_CORPUS_SIZE"
        else
            echo "Index Corpus Size:        $INSERT_CORPUS_SIZE"
        fi
    fi
}

apply_scalar_override() {
    local var_name="$1"
    local override_name="${var_name}_OVERRIDE"
    local uppercase_var_name="${var_name^^}"
    local uppercase_override_name="${uppercase_var_name}_OVERRIDE"
    local override_value=""

    if [[ -n "${!override_name:-}" ]]; then
        override_value="${!override_name}"
    elif [[ -n "${!uppercase_override_name:-}" ]]; then
        override_value="${!uppercase_override_name}"
    fi

    if [[ -n "$override_value" ]]; then
        printf -v "$var_name" '%s' "$override_value"
    fi
}

apply_array_override() {
    local var_name="$1"
    local override_name="${var_name}_OVERRIDE"
    local uppercase_var_name="${var_name^^}"
    local uppercase_override_name="${uppercase_var_name}_OVERRIDE"
    local override_value=""

    if [[ -n "${!override_name:-}" ]]; then
        override_value="${!override_name}"
    elif [[ -n "${!uppercase_override_name:-}" ]]; then
        override_value="${!uppercase_override_name}"
    fi

    if [[ -n "$override_value" ]]; then
        read -r -a "$var_name" <<< "$override_value"
    fi
}

apply_overrides() {
    apply_array_override NODES
    apply_array_override WORKERS_PER_NODE
    apply_array_override CORES
    apply_array_override INSERT_BATCH_SIZE
    apply_array_override QUERY_BATCH_SIZE

    apply_scalar_override WALLTIME
    apply_scalar_override queue
    apply_scalar_override TASK
    apply_scalar_override RUN_MODE
    apply_scalar_override STORAGE_MEDIUM
    apply_scalar_override PERF
    apply_scalar_override PERF_EVENTS
    apply_scalar_override VECTOR_DIM
    apply_scalar_override DISTANCE_METRIC
    apply_scalar_override GPU_INDEX
    apply_scalar_override INSERT_DATA_FILEPATH
    apply_scalar_override INSERT_CORPUS_SIZE
    apply_scalar_override INSERT_BALANCE_STRATEGY
    apply_scalar_override INSERT_CLIENTS_PER_WORKER
    apply_scalar_override INSERT_STREAMING
    apply_scalar_override QUERY_DATA_FILEPATH
    apply_scalar_override QUERY_CORPUS_SIZE
    apply_scalar_override QUERY_BALANCE_STRATEGY
    apply_scalar_override QUERY_CLIENTS_PER_WORKER
    apply_scalar_override TOTAL_QUERY_CLIENTS
    apply_scalar_override QUERY_STREAMING
    apply_scalar_override PLATFORM
    apply_scalar_override QDRANT_VERSION
    apply_scalar_override QDRANT_SIF
    apply_scalar_override QDRANT_LOCAL_IMAGE
    apply_scalar_override QDRANT_EXECUTABLE
    apply_scalar_override RESTORE_DIR
    apply_scalar_override EXPECTED_CORPUS_SIZE
    apply_scalar_override INSERT_TRACE
    apply_scalar_override QUERY_TRACE
}
