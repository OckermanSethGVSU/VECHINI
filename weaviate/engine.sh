#!/bin/bash

source "$ROOT_DIR/common/engine_schema_lib.sh"

ENGINE_NAME="weaviate"
ENGINE_SCHEMA_PREFIX="WEAVIATE"
schema_engine_init "weaviate" "$ENGINE_SCHEMA_PREFIX" "Weaviate"

weaviate_cores_label() {
    local cores_value="${CORES_CURRENT:-${CORES:-}}"

    if [[ -z "$cores_value" ]]; then
        printf '%s\n' "none"
    else
        printf '%s\n' "$cores_value"
    fi
}

# Print Weaviate-specific help plus the schema variable table.
weaviate_print_help() {
    cat <<'EOF'
Weaviate Help
=============

Use `--set NAME=value` to override any variable.
Any variable may be a single value or a sweep list separated by spaces.

Command examples:
  ./pbs_submit_manager.sh --engine weaviate --set TASK=INSERT --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj
  ./pbs_submit_manager.sh --generate-only --engine weaviate --set TASK=QUERY --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj

Variables
---------
EOF
    echo
    schema_print_registry_table "$ENGINE_SCHEMA_PREFIX"
}

# Load Weaviate schema defaults into the shared engine contract.
engine_set_defaults() {
    schema_load "$ENGINE_SCHEMA_PREFIX" "$ROOT_DIR/common/schema.sh"
    schema_append_file "$ENGINE_SCHEMA_PREFIX" "$ENGINE_DIR/schema.sh"
    schema_apply_defaults "$ENGINE_SCHEMA_PREFIX"
    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
    REQUIRES_DAOS="false"
}

# Apply --set/env overrides and fill BASE_DIR from the current directory.
engine_apply_overrides() {
    schema_sync_values_from_current_globals "$ENGINE_SCHEMA_PREFIX"
    schema_apply_overrides_from_env "$ENGINE_SCHEMA_PREFIX"
    if [[ -z "${WEAVIATE_VALUES[BASE_DIR]:-}" ]]; then
        WEAVIATE_VALUES["BASE_DIR"]="$(pwd)"
    fi
    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
}

# Validate required values and schema choices after overrides are resolved.
engine_validate_config() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX"

    if [[ "${RUN_MODE^^}" == "PBS" && "${ALLOW_REMOTE_WEAVIATE_IMAGE:-false}" != "true" && -z "${WEAVIATE_SIF:-}" ]]; then
        echo "WEAVIATE_SIF is required for PBS runs unless ALLOW_REMOTE_WEAVIATE_IMAGE=true." >&2
        return 1
    fi

    if [[ "${RUN_MODE^^}" == "PBS" && -n "${WEAVIATE_SIF:-}" && "$WEAVIATE_SIF" == */* ]]; then
        echo "WEAVIATE_SIF must be a filename under $ENGINE_DIR/sifs, not a path: $WEAVIATE_SIF" >&2
        return 1
    fi
}

# Print resolved Weaviate settings before run directory generation/submission.
engine_print_summary() {
    schema_print_resolved_summary "$ENGINE_SCHEMA_PREFIX"
}

# Entry point used by the root submit manager for --help --engine weaviate.
engine_show_help() {
    weaviate_print_help
}

# Emit all schema sweep combinations for the root submit manager.
engine_iterate_matrix() {
    schema_emit_combos_recursive "$ENGINE_SCHEMA_PREFIX" 0 ""
}

# Load one matrix combo into scalar globals used by naming and env emission.
engine_load_combo() {
    schema_load_combo_assignments "$1"

    TASK="${TASK^^}"

    NODES_CURRENT="$NODES"
    WORKERS_PER_NODE_CURRENT="$WORKERS_PER_NODE"
    INSERT_CLIENTS_CURRENT="$INSERT_CLIENTS_PER_WORKER"
    QUERY_BATCH_CURRENT="$QUERY_BATCH_SIZE"
    INSERT_BATCH_CURRENT="$INSERT_BATCH_SIZE"
    CORES_CURRENT="$CORES"
    TOTAL_NODES=$((NODES_CURRENT + 1))
    JOB_NAME="${TASK}_${STORAGE_MEDIUM}_N${NODES_CURRENT}"
    REQUIRES_DAOS="false"
}

# Validate a loaded combo after sweep expansion.
engine_validate_combo() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX" || return 1
}

# Build the Weaviate run directory name for the loaded combo.
engine_make_run_dir_name() {
    local timestamp
    timestamp="$(date +"%Y-%m-%d_%H_%M_%S")"

    echo "${TASK}_${STORAGE_MEDIUM}_N${NODES_CURRENT}_${timestamp}"
}

# Select the Weaviate launch script copied into submit.sh.
engine_main_script_path() {
    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        echo "local_main.sh"
    else
        echo "main.sh"
    fi
}

# Emit run_config.env contents consumed by the Weaviate launch scripts.
engine_emit_runtime_env() {
    schema_emit_runtime_env "$ENGINE_SCHEMA_PREFIX"

    printf 'NODES=%s\n' "$NODES_CURRENT"
    printf 'WORKERS_PER_NODE=%s\n' "$WORKERS_PER_NODE_CURRENT"
    printf 'CORES=%s\n' "$CORES_CURRENT"
    printf 'QUERY_BATCH_SIZE=%s\n' "$QUERY_BATCH_CURRENT"
    printf 'INSERT_BATCH_SIZE=%s\n' "$INSERT_BATCH_CURRENT"
    printf 'INSERT_CLIENTS_PER_WORKER=%s\n' "$INSERT_CLIENTS_CURRENT"
}

# Stage the Weaviate launch files, client binaries, and helper scripts.
engine_copy_payload() {
    local target_dir="$1"
    local batch_client_src_dir="$ENGINE_DIR/clients/batch_client"
    local mixed_client_src_dir="$ENGINE_DIR/clients/mixed"

    copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
        "health_check.py" \
        "create_basic_collection.py" \
        "multi_client_summary.py"
    if [[ "$TASK" == "MIXED" ]]; then
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
            "npy_inspect.py" \
            "mixed_timeline.py"
    fi
    if [[ -d "$ENGINE_DIR/pythonUtils" ]]; then
        copy_optional_engine_items "$ENGINE_DIR" "$target_dir" "pythonUtils"
    fi

    local -a bins
    local -A seen_bins=()
    local bin

    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        copy_engine_items "$ENGINE_DIR/weaviateSetup" "$target_dir" \
            "launchWeaviateNodeLocal.sh"
    else
        if [[ "${ALLOW_REMOTE_WEAVIATE_IMAGE:-false}" != "true" ]]; then
            if [[ ! -f "$ENGINE_DIR/sifs/$WEAVIATE_SIF" ]]; then
                echo "Required payload item missing: $ENGINE_DIR/sifs/$WEAVIATE_SIF" >&2
                echo "Run $ENGINE_DIR/scripts/download_sif.sh [version] to download a Weaviate SIF, then set WEAVIATE_SIF to that filename." >&2
                return 1
            fi
            cp "$ENGINE_DIR/sifs/$WEAVIATE_SIF" "$target_dir/weaviate.sif"
        fi
        copy_engine_items "$ENGINE_DIR/weaviateSetup" "$target_dir" \
            "launchWeaviateNode.sh" \
            "mapping.py"
    fi

    if [[ "$TASK" == "MIXED" ]]; then
        bins=("mixed" "batch_client")
    else
        bins=("batch_client")
    fi

    local -a unique_bins=()
    for bin in "${bins[@]}"; do
        [[ -n "$bin" ]] || continue
        if [[ -z "${seen_bins[$bin]:-}" ]]; then
            seen_bins["$bin"]=1
            unique_bins+=("$bin")
        fi
    done

    for bin in "${unique_bins[@]}"; do
        case "$bin" in
            mixed)
                if [[ ! -f "$mixed_client_src_dir/$bin" ]]; then
                    echo "Required client binary missing: ${mixed_client_src_dir#$ENGINE_DIR/}/$bin" >&2
                    echo "Build it with: (cd $ENGINE_DIR/clients && ./build.sh ${bin})" >&2
                    return 1
                fi
                copy_engine_items "$mixed_client_src_dir" "$target_dir" "$bin"
                ;;
            batch_client)
                if [[ ! -f "$batch_client_src_dir/$bin" ]]; then
                    echo "Required client binary missing: ${batch_client_src_dir#$ENGINE_DIR/}/$bin" >&2
                    echo "Build it with: (cd $ENGINE_DIR/clients && ./build.sh ${bin})" >&2
                    return 1
                fi
                copy_engine_items "$batch_client_src_dir" "$target_dir" "$bin"
                ;;
            *)
                echo "Unknown client binary requested for staging: $bin" >&2
                return 1
                ;;
        esac
    done
    for bin in "${unique_bins[@]}"; do
        if [[ -f "$target_dir/$bin" ]]; then
            chmod +x "$target_dir/$bin"
        fi
    done

}
