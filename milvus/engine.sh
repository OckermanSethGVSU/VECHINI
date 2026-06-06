#!/bin/bash

source "$ROOT_DIR/common/engine_schema_lib.sh"
source "$ENGINE_DIR/utils/utils.sh"

ENGINE_NAME="milvus"
ENGINE_SCHEMA_PREFIX="MILVUS"
schema_engine_init "milvus" "$ENGINE_SCHEMA_PREFIX" "Milvus"

milvus_cores_label() {
    if [[ -n "${CORES_CURRENT:-}" ]]; then
        printf '%s\n' "$CORES_CURRENT"
    else
        printf '%s\n' "none"
    fi
}

milvus_print_help() {
    cat <<'EOF'
Milvus Help
===========

Use `--set NAME=value` to override any variable.
Any variable may be a single value or a sweep list separated by spaces.

Command examples:
  ./pbs_submit_manager.sh --engine milvus --set TASK=QUERY --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj
  ./pbs_submit_manager.sh --generate-only --engine milvus --set TASK=INSERT --set INSERT_BATCH_SIZE="256 512" --set PLATFORM=AURORA --set WALLTIME=01:00:00 --set QUEUE=debug-scaling --set ACCOUNT=myproj

Variables
---------
EOF
    echo
    schema_print_registry_table "$ENGINE_SCHEMA_PREFIX"
}

engine_set_defaults() {
    schema_load "$ENGINE_SCHEMA_PREFIX" "$ROOT_DIR/common/schema.sh"
    schema_append_file "$ENGINE_SCHEMA_PREFIX" "$ENGINE_DIR/schema.sh"
    schema_apply_defaults "$ENGINE_SCHEMA_PREFIX"
    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
    REQUIRES_DAOS="false"
}

engine_apply_overrides() {
    schema_sync_values_from_current_globals "$ENGINE_SCHEMA_PREFIX"
    schema_apply_overrides_from_env "$ENGINE_SCHEMA_PREFIX"

    if [[ -z "${MILVUS_VALUES[BASE_DIR]:-}" ]]; then
        MILVUS_VALUES["BASE_DIR"]="$(pwd)"
    fi

    if [[ -z "${MILVUS_VALUES[ETCD_MEDIUM]:-}" ]]; then
        MILVUS_VALUES["ETCD_MEDIUM"]="${MILVUS_VALUES[STORAGE_MEDIUM]:-}"
    fi

    if [[ -z "${MILVUS_VALUES[MINIO_MODE]:-}" ]]; then
        if [[ "${MILVUS_VALUES[MODE]:-}" == "DISTRIBUTED" ]]; then
            MILVUS_VALUES["MINIO_MODE"]="stripped"
        else
            MILVUS_VALUES["MINIO_MODE"]="off"
        fi
    fi

    schema_assign_globals_from_values "$ENGINE_SCHEMA_PREFIX"
}

engine_validate_config() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX"

    if [[ "${RUN_MODE^^}" == "PBS" && -z "${ENV_PATH:-}" && "${ALLOW_SYSTEM_PYTHON:-False}" != "True" ]]; then
        cat >&2 <<'EOF'
Milvus PBS runs require ENV_PATH by default so the Python environment is explicit.

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

    case "$TASK" in
        IMPORT)
            if [[ -z "${INSERT_DATA_FILEPATH:-}" ]]; then
                echo "Milvus variable 'INSERT_DATA_FILEPATH' is required for TASK=$TASK." >&2
                return 1
            fi
            ;;
        QUERY)
            if [[ -z "${QUERY_DATA_FILEPATH:-}" ]]; then
                echo "Milvus variable 'QUERY_DATA_FILEPATH' is required for TASK=QUERY." >&2
                return 1
            fi
            ;;
        MIXED)
            if [[ -z "${MIXED_QUERY_DATA_FILEPATH:-}" ]]; then
                echo "Milvus variable 'MIXED_QUERY_DATA_FILEPATH' is required for TASK=MIXED." >&2
                return 1
            fi
            if [[ -z "${MIXED_INSERT_DATA_FILEPATH:-}" ]]; then
                echo "Milvus variable 'MIXED_INSERT_DATA_FILEPATH' is required for TASK=MIXED." >&2
                return 1
            fi
            ;;
    esac

    if [[ -n "${INSERT_START_ID:-}" ]]; then
        INSERT_START_ID="$INSERT_START_ID"
    fi
}

engine_print_summary() {
    schema_print_resolved_summary "$ENGINE_SCHEMA_PREFIX"
}

engine_show_help() {
    milvus_print_help
}

engine_iterate_matrix() {
    schema_emit_combos_recursive "$ENGINE_SCHEMA_PREFIX" 0 ""
}

engine_load_combo() {
    schema_load_combo_assignments "$1"

    NODES_CURRENT="$NODES"
    if [[ "${TASK^^}" == "MIXED" ]]; then
        QUERY_BATCH_CURRENT="$MIXED_QUERY_BATCH_SIZE"
        INSERT_BATCH_CURRENT="$MIXED_INSERT_BATCH_SIZE"
    else
        QUERY_BATCH_CURRENT="$QUERY_BATCH_SIZE"
        INSERT_BATCH_CURRENT="$INSERT_BATCH_SIZE"
    fi
    CORES_CURRENT="$CORES"
    TOTAL_NODES=$((NODES_CURRENT + 1))
    JOB_NAME="${TASK,,}_${MODE,,}_${NODES_CURRENT}n_$(milvus_cores_label)c_q${QUERY_BATCH_CURRENT}"

    if [[ "$STORAGE_MEDIUM" == "DAOS" || "$ETCD_MEDIUM" == "DAOS" || ( "$MODE" == "DISTRIBUTED" && "$MINIO_MEDIUM" == "DAOS" ) ]]; then
        REQUIRES_DAOS="true"
    else
        REQUIRES_DAOS="false"
    fi
}

engine_validate_combo() {
    schema_validate_current_values "$ENGINE_SCHEMA_PREFIX"
    validate_programmatic_submit_config "$NODES_CURRENT" "$CORES_CURRENT" "$INSERT_BATCH_CURRENT" "$QUERY_BATCH_CURRENT"
}

engine_make_run_dir_name() {
    local timestamp

    timestamp="$(date +"%Y-%m-%d_%H_%M_%S")"
    echo "${TASK}_${STORAGE_MEDIUM}_${timestamp}"
}

engine_main_script_path() {
    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        echo "local_main.sh"
    else
        echo "main.sh"
    fi
}

engine_emit_runtime_env() {
    schema_emit_runtime_env "$ENGINE_SCHEMA_PREFIX"
}

stage_go_source_file() {
    local src_dir="$1"
    local target_dir="$2"
    local base_name

    base_name="$(basename "$src_dir")"
    copy_engine_items "$src_dir" "$target_dir/goCode" "main.go"
    if [[ -f "$target_dir/goCode/main.go" ]]; then
        mv "$target_dir/goCode/main.go" "$target_dir/goCode/${base_name}.go"
    fi
}

engine_uses_bulk_import_payload() {
    local task_upper="${TASK^^}"
    local insert_method="${INSERT_METHOD:-traditional}"

    if [[ "$task_upper" == "IMPORT" ]]; then
        return 0
    fi

    insert_method="${insert_method,,}"
    case "$insert_method" in
        bulk|bulk_upload|bulk-upload|import)
            return 0
            ;;
    esac

    return 1
}

engine_needs_npy_inspect() {
    [[ "${TASK^^}" == "MIXED" ]]
}

# Stage Milvus containers, launch scripts, clients, and task-specific helpers.
engine_copy_payload() {
    local target_dir="$1"

    copy_engine_items "$ENGINE_DIR" "$target_dir" "runtime_state"

    if [[ "${RUN_MODE^^}" == "LOCAL" ]]; then
        copy_engine_items "$ENGINE_DIR/clients/batch_client" "$target_dir" \
            "batch_client"
        stage_go_source_file "$ENGINE_DIR/clients/batch_client" "$target_dir"
        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
            "multi_client_summary.py" \
            "replace_unified.py"
        if engine_uses_bulk_import_payload; then
            copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
                "bulk_upload_import.py" \
                "bulk_upload_import_mc.py"
        fi
        if engine_needs_npy_inspect; then
            copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "npy_inspect.py"
        fi
        copy_engine_items "$ENGINE_DIR/runtime" "$target_dir" "configs"

        if [[ "$TASK" == "MIXED" ]]; then
            copy_engine_items "$ENGINE_DIR/clients/mixed" "$target_dir" \
                "mixed"
            stage_go_source_file "$ENGINE_DIR/clients/mixed" "$target_dir"
            copy_engine_items "$ENGINE_DIR/../qdrant/scripts" "$target_dir" "mixed_timeline.py"
        fi

        if [[ -z "$RESTORE_DIR" ]]; then
            copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "setup_collection.py"
            if [[ "$TASK" == "INDEX" || "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
                copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "index.py"
            fi
        else
            copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "status.py"
        fi
    else
        if [[ "$MODE" == "STANDALONE" ]]; then
            copy_engine_items "$ENGINE_DIR/sifs" "$target_dir" "milvus.sif"
            copy_engine_items "$ENGINE_DIR/runtime/cluster" "$target_dir" "standaloneLaunch.sh" "execute.sh"
        elif [[ "$MODE" == "DISTRIBUTED" ]]; then
            copy_engine_items "$ENGINE_DIR/sifs" "$target_dir" "milvus.sif" "etcd_v3.5.18.sif" "minio.sif"
            copy_engine_items "$ENGINE_DIR/runtime/cluster" "$target_dir" \
                "execute.sh" \
                "launch_etcd.sh" \
                "launch_minio.sh" \
                "launch_milvus_part.sh"
        else
            echo "Unknown MODE: $MODE" >&2
            exit 1
        fi

        if [[ "$TRACING" == "True" ]]; then
            copy_engine_items "$ENGINE_DIR/sifs" "$target_dir" "otel-collector.sif"
            copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "launch_otel.sh" "otel_config.yaml" "analyze_traces.py"
        fi

        copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
            "net_mapping.py" \
            "replace_unified.py" \
            "profile.py" \
            "poll.py" \
            "multi_client_summary.py"
        if engine_uses_bulk_import_payload; then
            copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" \
                "bulk_upload_import.py" \
                "bulk_upload_import_mc.py"
        fi
        if engine_needs_npy_inspect; then
            copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "npy_inspect.py"
        fi
        copy_engine_items "$ENGINE_DIR/runtime" "$target_dir" "configs"

        copy_engine_items "$ENGINE_DIR/clients/batch_client" "$target_dir" \
            "batch_client"
        stage_go_source_file "$ENGINE_DIR/clients/batch_client" "$target_dir"

        if [[ "$TASK" == "MIXED" ]]; then
            copy_engine_items "$ENGINE_DIR/clients/mixed" "$target_dir" \
                "mixed"
            stage_go_source_file "$ENGINE_DIR/clients/mixed" "$target_dir"
            copy_engine_items "$ENGINE_DIR/../qdrant/scripts" "$target_dir" "mixed_timeline.py"
        fi

        if [[ -z "$RESTORE_DIR" ]]; then
            copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "setup_collection.py"
            if [[ "$TASK" == "INDEX" || "$TASK" == "QUERY" || "$TASK" == "MIXED" ]]; then
                copy_engine_items "$ENGINE_DIR/scripts" "$target_dir" "index.py"
            fi
        else
            copy_engine_items "$ENGINE_DIR/utils" "$target_dir" "status.py"
        fi
    fi
}
