#!/bin/bash
#
# Milvus schema format:
#   register_milvus_var "NAME" "REQUIREMENT" "DEFAULT" "CHOICES" "DESCRIPTION" ["REQUIRED_IF"]
#
# REQUIREMENT:
#   required    - caller must provide a value; DEFAULT should normally be empty.
#   default     - DEFAULT is used when the caller does not override the variable.
#   conditional - variable is required only when REQUIRED_IF matches the current config.
#
# CHOICES is a space-separated allowlist. Leave it empty to allow any value.
# REQUIRED_IF currently supports one condition in the form OTHER_VAR=value1|value2.
# Every registered variable may be set to one value or a sweep list separated by spaces.

# Engine/runtime selection
register_milvus_var "MODE" "default" "STANDALONE" "DISTRIBUTED STANDALONE" "Milvus deployment mode"
register_milvus_var "MILVUS_BUILD_DIR" "default" "" "" "Milvus build directory"
register_milvus_var "MILVUS_CONFIG_DIR" "default" "runtime" "" "Milvus config directory"
register_milvus_var "PERF" "default" "NONE" "NONE STAT TRACE" "Performance collection mode"
register_milvus_var "PERF_EVENTS" "default" "topdown-be-bound,topdown-mem-bound,topdown-retiring,topdown-fe-bound,topdown-bad-spec" "" "Comma-separated perf stat events"
register_milvus_var "WAL" "default" "woodpecker" "" "Milvus WAL mode"
register_milvus_var "GPU_INDEX" "default" "False" "True False" "Whether to use GPU indexing"
register_milvus_var "TRACING" "default" "False" "True False" "Enable tracing"
register_milvus_var "DEBUG" "default" "False" "True False" "Enable debug behavior"
register_milvus_var "AUTO_CLEANUP" "default" "False" "True False true false TRUE FALSE" "Enable end-of-run storage cleanup"
register_milvus_var "MINIO_MODE" "default" "off" "off single stripped" "MinIO topology"
register_milvus_var "MINIO_MEDIUM" "default" "lustre" "memory DAOS lustre SSD" "Storage medium for MinIO data"
register_milvus_var "ETCD_MEDIUM" "default" "memory" "memory DAOS lustre SSD" "Storage medium for etcd data"
register_milvus_var "LOCAL_SHARED_STORAGE_PATH" "default" "" "" "Shared storage path used when MINIO_MODE=off"

# Insert / import workload
register_milvus_var "INSERT_CLIENTS_PER_PROXY" "default" "8" "" "Insert clients per proxy"
register_milvus_var "INSERT_METHOD" "default" "traditional" "traditional bulk" "Insert method"
register_milvus_var "BULK_UPLOAD_TRANSPORT" "default" "mc" "mc remote" "Bulk upload transport"
register_milvus_var "BULK_UPLOAD_STAGING_MEDIUM" "default" "memory" "memory DAOS lustre SSD" "Bulk upload staging medium"
register_milvus_var "IMPORT_PROCESSES" "default" "204" "" "Bulk import process count"
register_milvus_var "INSERT_START_ID" "default" "" "" "Optional insert id offset override"
register_milvus_var "BULK_IMPORT_PREPARE_ONLY" "default" "TRUE" "TRUE FALSE" "Prepare import request only"
register_milvus_var "BULK_IMPORT_REQUEST_PATH" "default" "" "" "Path to bulk import request file"
register_milvus_var "BULK_IMPORT_LOAD_REQUEST" "default" "" "" "Bulk import request payload override"
register_milvus_var "BULK_IMPORT_SUMMARY_PATH" "default" "" "" "Path to bulk import summary output"

# Collection / index settings
register_milvus_var "INIT_FLAT_INDEX" "default" "FALSE" "TRUE FALSE" "Whether to initialize a flat index"
register_milvus_var "SHARDS" "default" "1" "" "Collection shard count"
register_milvus_var "DML_CHANNELS" "default" "16" "" "DML channel count"
register_milvus_var "MAX_SEGMENT_SIZE" "default" "1024" "" "Milvus dataCoord.segment.maxSize in MB"
register_milvus_var "FLUSH_BEFORE_INDEX" "default" "TRUE" "TRUE FALSE" "Flush collection before indexing"

# Query workload
register_milvus_var "QUERY_CLIENTS_PER_PROXY" "default" "1" "" "Query clients per proxy"

# Mixed workload controls
register_milvus_var "INSERT_OPS_PER_SEC" "default" "" "" "Insert ops/sec when MIXED_INSERT_MODE=rate"
register_milvus_var "QUERY_OPS_PER_SEC" "default" "" "" "Query ops/sec when MIXED_QUERY_MODE=rate"
register_milvus_var "MIXED_RESULT_PATH" "default" "mixed_logs" "" "Output subdirectory for mixed workload logs"

# Optional request tuning
register_milvus_var "VECTOR_FIELD" "default" "" "" "Optional vector field override"
register_milvus_var "ID_FIELD" "default" "" "" "Optional id field override"
register_milvus_var "QUERY_EF_SEARCH" "default" "" "" "Optional query efSearch override"
register_milvus_var "RPC_TIMEOUT" "default" "" "" "Optional RPC timeout override"

register_milvus_var "MIXED_INSERT_BATCH_MIN" "default" "" "" "Optional randomized mixed insert batch lower bound"
register_milvus_var "MIXED_INSERT_BATCH_MAX" "default" "" "" "Optional randomized mixed insert batch upper bound"
register_milvus_var "MIXED_QUERY_BATCH_MIN" "default" "" "" "Optional randomized mixed query batch lower bound"
register_milvus_var "MIXED_QUERY_BATCH_MAX" "default" "" "" "Optional randomized mixed query batch upper bound"
register_milvus_var "INSERT_BATCH_MIN" "default" "" "" "Optional randomized insert batch lower bound"
register_milvus_var "INSERT_BATCH_MAX" "default" "" "" "Optional randomized insert batch upper bound"
register_milvus_var "QUERY_BATCH_MIN" "default" "" "" "Optional randomized query batch lower bound"
register_milvus_var "QUERY_BATCH_MAX" "default" "" "" "Optional randomized query batch upper bound"

# Restore / recovery
register_milvus_var "RESTORE_DIR" "default" "" "" "Restore an existing Milvus state from this directory"
register_milvus_var "EXPECTED_CORPUS_SIZE" "default" "10000000" "" "Expected corpus size when restoring"

# Distributed topology
register_milvus_var "ETCD_MODE" "default" "single" "single replicated" "ETCD topology"
register_milvus_var "STREAMING_NODES" "default" "2" "" "Streaming node count"
register_milvus_var "STREAMING_NODES_PER_CN" "default" "2" "" "Streaming nodes per compute node"
register_milvus_var "QUERY_NODES" "default" "1" "" "Query node count"
register_milvus_var "QUERY_NODES_PER_CN" "default" "1" "" "Query nodes per compute node"
register_milvus_var "DATA_NODES" "default" "1" "" "Data node count"
register_milvus_var "DATA_NODES_PER_CN" "default" "1" "" "Data nodes per compute node"
register_milvus_var "COORDINATOR_NODES" "default" "1" "" "Coordinator node count"
register_milvus_var "COORDINATOR_NODES_PER_CN" "default" "1" "" "Coordinator nodes per compute node"
register_milvus_var "NUM_PROXIES" "default" "1" "" "Proxy count"
register_milvus_var "NUM_PROXIES_PER_CN" "default" "1" "" "Proxies per compute node"
