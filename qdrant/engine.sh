#!/bin/bash

source "$ROOT_DIR/common/engine_schema_lib.sh"
source "$ENGINE_DIR/utils/utils.sh"

ENGINE_NAME="qdrant"
ENGINE_SCHEMA_PREFIX="QDRANT"
schema_engine_init "qdrant" "$ENGINE_SCHEMA_PREFIX" "Qdrant"

# Print Qdrant-specific help plus the schema variable table.
qdrant_print_help() {
    cat <<'EOF'
Qdrant Help
===========

Use `--set NAME=value` to override any variable.
Any variable may be a single value or a sweep list separated by spaces.

Command examples:
  ./pbs_submit_manager.sh --engine qdrant --set TASK=QUERY --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj
  ./pbs_submit_manager.sh --generate-only --engine qdrant --set TASK=QUERY --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj

Variables
---------
EOF
    echo
    schema_print_registry_table "$ENGINE_SCHEMA_PREFIX"
}

# Normalize the optional CORES value for names and preview output.
qdrant_cores_label() {
    if [[ -n "${CORES_CURRENT:-}" ]]; then
        printf '%s\n' "$CORES_CURRENT"
    else
        printf '%s\n' "none"
    fi
}

qdrant_normalize_version() {
    local version="$1"

    [[ -n "$version" ]] || return 0
    if [[ ! "$version" =~ ^[A-Za-z0-9._-]+$ ]]; then
        echo "Invalid QDRANT_VERSION '$version'. Use a container tag such as 1.16.1, v1.16.1, or latest." >&2
        return 1
    fi

    if [[ "$version" == "latest" || "$version" == v* ]]; then
        printf '%s\n' "$version"
    else
        printf 'v%s\n' "$version"
    fi
}

qdrant_resolve_runtime_version() {
    local normalized_version=""

    if [[ -n "${QDRANT_VERSION:-}" ]]; then
        normalized_version="$(qdrant_normalize_version "$QDRANT_VERSION")" || return 1
        QDRANT_VERSION="$normalized_version"
    fi

    if [[ -z "${QDRANT_LOCAL_IMAGE:-}" ]]; then
        if [[ -n "$normalized_version" ]]; then
            QDRANT_LOCAL_IMAGE="qdrant/qdrant:${normalized_version}"
        else
            QDRANT_LOCAL_IMAGE="qdrant/qdrant:latest"
        fi
    fi

    if [[ "${RUN_MODE^^}" == "PBS" && -z "${QDRANT_SIF:-}" && -n "$normalized_version" ]]; then
        QDRANT_SIF="qdrant_${normalized_version}.sif"
    fi
}

qdrant_validate_runtime_selection() {
    if [[ -n "${QDRANT_VERSION:-}" ]]; then
        qdrant_normalize_version "$QDRANT_VERSION" >/dev/null || return 1
    fi

    if [[ "${RUN_MODE^^}" == "PBS" && -z "${QDRANT_SIF:-}" && -z "${QDRANT_VERSION:-}" ]]; then
        echo "Qdrant PBS runs require QDRANT_SIF or QDRANT_VERSION." >&2
        echo "Download a versioned SIF with: $ENGINE_DIR/utils/download_sif.sh [version]" >&2
        return 1
    fi

    if [[ "${RUN_MODE^^}" == "PBS" && "${QDRANT_SIF:-}" == */* ]]; then
        echo "QDRANT_SIF must be a filename under $ENGINE_DIR/sifs, not a path: $QDRANT_SIF" >&2
        return 1
    fi
}

# Load Qdrant schema defaults into the shared engine contract.
engine_set_defaults() {
    schema_load "$ENGINE_SCHEMA_PREFIX" "$ROOT_DIR/common/schema.sh"
    schema_append_file "$ENGINE_SCHEMA_PREFIX" "$ENGINE_DIR/schema.sh"
    schema_apply_defaults "$ENGINE_SCHEMA_PREFIX"
    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
    REQUIRES_DAOS="false"
}

# Apply --set/env overrides and expose the resolved scalar globals.
engine_apply_overrides() {
    schema_sync_values_from_current_globals "$ENGINE_SCHEMA_PREFIX"
    schema_apply_overrides_from_env "$ENGINE_SCHEMA_PREFIX"
    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
    apply_scalar_override INSERT_START_ID
}

# Validate schema values.
engine_validate_config() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX"

    if [[ "${RUN_MODE^^}" == "PBS" && -z "${ENV_PATH:-}" && "${ALLOW_SYSTEM_PYTHON:-False}" != "True" ]]; then
        cat >&2 <<'EOF'
Qdrant PBS runs require ENV_PATH by default so the Python environment is explicit.

Set one of these on the command line:
  --set ENV_PATH=/path/to/python/env

Or, if the job's loaded modules/current environment already provide every Python dependency:
  --set ALLOW_SYSTEM_PYTHON=True

You can also put the same setting in your config file:
  ENV_PATH=/path/to/python/env
  # or
  ALLOW_SYSTEM_PYTHON=True
EOF
        return 1
    fi

    qdrant_validate_runtime_selection || return 1

    INSERT_START_ID="${INSERT_START_ID:-}"
}

# Print resolved Qdrant settings before run directory generation/submission.
engine_print_summary() {
    schema_print_resolved_summary "$ENGINE_SCHEMA_PREFIX"
}

# Entry point used by the root submit manager for --help --engine qdrant.
engine_show_help() {
    qdrant_print_help
}

# Print one concise preview entry for a generated matrix combo.
engine_preview_combo() {
    local run_dir_name="$1"

    cat <<EOF
- run_dir: ${run_dir_name}
  task: ${TASK}
  nodes: ${NODES_CURRENT}
  workers_per_node: ${WORKERS_PER_NODE_CURRENT}
  cores: ${CORES_CURRENT}
  insert_batch_size: ${INSERT_BATCH_CURRENT}
  query_batch_size: ${QUERY_BATCH_CURRENT}
EOF
}

# Emit all schema sweep combinations for the root submit manager.
engine_iterate_matrix() {
    schema_emit_combos_recursive "$ENGINE_SCHEMA_PREFIX" 0 ""
}

# Load one matrix combo into scalar globals used by naming and env emission.
engine_load_combo() {
    schema_load_combo_assignments "$1"
    qdrant_resolve_runtime_version || return 1

    NODES_CURRENT="$NODES"
    WORKERS_PER_NODE_CURRENT="$WORKERS_PER_NODE"
    if [[ "${TASK^^}" == "MIXED" ]]; then
        QUERY_BATCH_CURRENT="$MIXED_QUERY_BATCH_SIZE"
        INSERT_BATCH_CURRENT="$MIXED_INSERT_BATCH_SIZE"
    else
        QUERY_BATCH_CURRENT="$QUERY_BATCH_SIZE"
        INSERT_BATCH_CURRENT="$INSERT_BATCH_SIZE"
    fi
    CORES_CURRENT="$CORES"
    TOTAL_NODES=$((NODES_CURRENT + 1))
    JOB_NAME="${TASK,,}_${NODES_CURRENT}n_${WORKERS_PER_NODE_CURRENT}w_$(qdrant_cores_label)c_q${QUERY_BATCH_CURRENT}"
    if [[ "$STORAGE_MEDIUM" == "DAOS" ]]; then
        REQUIRES_DAOS="true"
    else
        REQUIRES_DAOS="false"
    fi
}

qdrant_perf_enabled() {
    [[ "${PERF^^}" == "STAT" || "${PERF^^}" == "RECORD" ]]
}

qdrant_validate_perf_payload() {
    local perf_path="$ENGINE_DIR/runtime_state/perf"

    qdrant_perf_enabled || return 0

    if [[ ! -f "$perf_path" ]]; then
        echo "Qdrant PERF=$PERF requires a staged perf executable at $perf_path" >&2
        return 1
    fi
    if [[ ! -x "$perf_path" ]]; then
        echo "Qdrant perf payload is not executable: $perf_path" >&2
        return 1
    fi
}

# Validate a loaded combo after sweep expansion.
engine_validate_combo() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX" || return 1
    schema_validate_recall_config || return 1
    qdrant_validate_runtime_selection || return 1
    qdrant_validate_perf_payload
}

# Build the Qdrant run directory name for the loaded combo.
engine_make_run_dir_name() {
    local timestamp

    timestamp="$(date +"%Y-%m-%d_%H_%M_%S")"
    echo "${TASK}_${STORAGE_MEDIUM}_N${NODES_CURRENT}_${timestamp}"
}

# Select the launch script copied into submit.sh for local vs PBS runs.
engine_main_script_path() {
    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        echo "local_main.sh"
    else
        echo "main.sh"
    fi
}

# Emit run_config.env contents consumed by Qdrant main/local_main scripts.
engine_emit_runtime_env() {
    if [[ -z "${BASE_DIR:-}" ]]; then
        BASE_DIR="$ENGINE_DIR"
    fi

    schema_emit_runtime_env "$ENGINE_SCHEMA_PREFIX"
    printf 'INSERT_START_ID=%s\n' "${INSERT_START_ID:-}"
}

# Stage the Qdrant binaries, launch scripts, and task-specific helper scripts.
engine_copy_payload() {
    local target_dir="$1"

    mkdir -p "$target_dir/rustSrc"
    qdrant_validate_perf_payload || return 1
    copy_engine_items "$ENGINE_DIR" "$target_dir" "runtime_state" || return 1
    if [[ "${CALCULATE_RECALL:-False}" == "True" ]]; then
        copy_engine_items "$ROOT_DIR/utils" "$target_dir" "compute_recall.py"
    fi

    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        copy_engine_items "$ENGINE_DIR/clients/batch_client" "$target_dir" "batch_client"
        copy_engine_items "$ENGINE_DIR/clients/batch_client/src" "$target_dir/rustSrc" "main.rs"
        if [[ "$TASK" == "MIXED" ]]; then
            copy_engine_items "$ENGINE_DIR/clients/mixed" "$target_dir" "mixed"
            cp "$ENGINE_DIR/clients/mixed/src/main.rs" "$target_dir/rustSrc/mixed_main.rs"
        fi
    else
        if [[ ! -f "$ENGINE_DIR/sifs/$QDRANT_SIF" ]]; then
            echo "Required payload item missing: $ENGINE_DIR/sifs/$QDRANT_SIF" >&2
            echo "Run $ENGINE_DIR/utils/download_sif.sh [version] to download a Qdrant SIF, then set QDRANT_SIF to that filename." >&2
            return 1
        fi
        cp "$ENGINE_DIR/sifs/$QDRANT_SIF" "$target_dir/qdrant.sif"
        copy_engine_items "$ENGINE_DIR/runtime/cluster" "$target_dir" "launchQdrantNode.sh" "launch.sh"

        if [[ -n "$QDRANT_EXECUTABLE" ]]; then
            copy_engine_items "$ENGINE_DIR/qdrantBuilds" "$target_dir" "$QDRANT_EXECUTABLE"
            mv "$target_dir/$QDRANT_EXECUTABLE" "$target_dir/qdrant"
        fi

        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "profile.py" "gen_dirs.py" "mapping.py"

        copy_engine_items "$ENGINE_DIR/clients/batch_client" "$target_dir" "batch_client"
        copy_engine_items "$ENGINE_DIR/clients/batch_client/src" "$target_dir/rustSrc" "main.rs"
        if [[ "$TASK" == "MIXED" ]]; then
            copy_engine_items "$ENGINE_DIR/clients/mixed" "$target_dir" "mixed"
            cp "$ENGINE_DIR/clients/mixed/src/main.rs" "$target_dir/rustSrc/mixed_main.rs"
        fi
    fi

    if [[ -n "$RESTORE_DIR" ]]; then
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "fix_peer_id.py" "collection_status.py"
    fi

    copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "summarize_client_timings.py"

    if [[ -z "$RESTORE_DIR" ]]; then
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "configure_collection.py"
    fi

    if [[ "$TASK" == "INDEX" || "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "build_index.py"
    fi

    if [[ "$TASK" == "MIXED" ]]; then
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "mixed_timeline.py"
        copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "npy_inspect.py"
    fi
}
