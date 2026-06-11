#!/bin/bash
#
# Qdrant schema format:
#   register_qdrant_var "NAME" "REQUIREMENT" "DEFAULT" "CHOICES" "DESCRIPTION" ["REQUIRED_IF"]
#
# REQUIREMENT:
#   required    - caller must provide a value; DEFAULT should normally be empty.
#   default     - DEFAULT is used when the caller does not override the variable.
#   conditional - variable is required only when REQUIRED_IF matches the current config.
#
# CHOICES is a space-separated allowlist. Leave it empty to allow any value.
# REQUIRED_IF currently supports one condition in the form OTHER_VAR=value1|value2.
# Every registered variable may be set to one value or a space-separated sweep list.
# The order in this file controls the order shown in `--help --engine qdrant`.

# Worker/shard layout
register_qdrant_var "WORKERS_PER_NODE" "default" "1" "" "Worker processes launched per compute node"
register_qdrant_var "REBALANCE_TOPOLOGY" "default" "False" "True False" "Whether configure_collection should actively move shards to the target topology"

# Engine/runtime selection
register_qdrant_var "QDRANT_SIF" "conditional" "" "" "Qdrant SIF filename under qdrant/sifs, for example qdrant_latest.sif" "RUN_MODE=PBS"
register_qdrant_var "QDRANT_EXECUTABLE" "default" "" "" "Optional local Qdrant executable override copied from qdrantBuilds; empty uses the executable inside the SIF"
register_qdrant_var "LOG_LEVEL" "default" "ERROR" "ERROR DEBUG INFO" "Qdrant log level passed to generated node configs"

# Index
register_qdrant_var "HNSW_M" "default" "16" "" "HNSW M parameter"
register_qdrant_var "HNSW_EF_CONSTRUCTION" "default" "100" "" "HNSW efConstruction parameter"
register_qdrant_var "MAX_SEGMENT_SIZE" "default" "" "" "Optional Qdrant max segment size in KB; only applied when set"
register_qdrant_var "DEFAULT_SEGMENT_NUMBER" "default" "" "" "Optional Qdrant default segment count; only applied when set"
register_qdrant_var "GPU_INDEX" "default" "False" "True False" "Whether to use GPU indexing"

# Quantization
register_qdrant_var "QUANTIZATION_TYPE" "default" "NONE" "NONE SCALAR BINARY PRODUCT TURBO" "Collection quantization method; TURBO requires Qdrant 1.18 or newer"
register_qdrant_var "QUANTIZATION_ALWAYS_RAM" "default" "False" "True False" "Keep quantized vectors in RAM"
register_qdrant_var "QUANTIZATION_SCALAR_QUANTILE" "default" "" "" "Optional scalar quantization bound quantile in the range (0, 1]"
register_qdrant_var "QUANTIZATION_BINARY_ENCODING" "default" "DEFAULT" "DEFAULT TWO_BITS ONE_AND_HALF_BITS" "Binary quantization encoding; DEFAULT uses one bit"
register_qdrant_var "QUANTIZATION_PRODUCT_COMPRESSION" "default" "X16" "X4 X8 X16 X32 X64" "Product quantization compression ratio"
register_qdrant_var "QUANTIZATION_TURBO_BITS" "default" "BITS4" "BITS4 BITS2 BITS1_5 BITS1" "TurboQuant bit depth"


# Insert / preload workload
register_qdrant_var "INSERT_CLIENTS_PER_WORKER" "default" "1" "" "Insert clients per worker"

# Query workload
register_qdrant_var "TOTAL_QUERY_CLIENTS" "default" "" "" "Optional total query clients across the run"
register_qdrant_var "QUERY_CLIENTS_PER_WORKER" "conditional" "1" "" "Query clients per worker" "TASK=QUERY"
register_qdrant_var "HNSW_EF_SEARCH" "default" "64" "" "Query efSearch override"

# Mixed workload controls
register_qdrant_var "INSERT_OPS_PER_SEC" "conditional" "" "" "Required when MIXED_INSERT_MODE=RATE" "MIXED_INSERT_MODE=RATE"
register_qdrant_var "QUERY_OPS_PER_SEC" "conditional" "" "" "Required when MIXED_QUERY_MODE=RATE" "MIXED_QUERY_MODE=RATE"
register_qdrant_var "RESULT_PATH" "default" "mixed_logs" "" "Output subdirectory for mixed workload logs"

register_qdrant_var "INSERT_BATCH_MIN" "default" "" "" "Optional randomized insert batch lower bound"
register_qdrant_var "INSERT_BATCH_MAX" "default" "" "" "Optional randomized insert batch upper bound"
register_qdrant_var "QUERY_BATCH_MIN" "default" "" "" "Optional randomized query batch lower bound"
register_qdrant_var "QUERY_BATCH_MAX" "default" "" "" "Optional randomized query batch upper bound"
register_qdrant_var "RPC_TIMEOUT" "default" "" "" "Optional RPC timeout override"

# Restore / recovery
register_qdrant_var "RESTORE_DIR" "default" "" "" "Restore an existing Qdrant state from this directory"
register_qdrant_var "EXPECTED_CORPUS_SIZE" "default" "10000000" "" "Expected corpus size when restoring"


# Profiling
register_qdrant_var "PERF" "default" "NONE" "NONE STAT RECORD" "Performance collection mode"
register_qdrant_var "PERF_EVENTS" "default" "topdown-be-bound,topdown-mem-bound,topdown-retiring,topdown-fe-bound,topdown-bad-spec" "" "Comma-separated perf stat events"
register_qdrant_var "INSERT_TRACE" "default" "" "" "Optional insert trace file or mode"
register_qdrant_var "QUERY_TRACE" "default" "" "" "Optional query trace file or mode"
