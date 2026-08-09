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
register_qdrant_var "QDRANT_VERSION" "default" "" "" "Optional Qdrant version, for example 1.16.1; derives the PBS SIF filename and local image when explicit overrides are empty"
register_qdrant_var "QDRANT_SIF" "default" "" "" "Explicit Qdrant SIF filename under qdrant/sifs; takes precedence over QDRANT_VERSION"
register_qdrant_var "QDRANT_LOCAL_IMAGE" "default" "" "" "Explicit local Qdrant container image; takes precedence over QDRANT_VERSION"
register_qdrant_var "QDRANT_EXECUTABLE" "default" "" "" "Optional local Qdrant executable override copied from qdrantBuilds; empty uses the executable inside the SIF"
register_qdrant_var "LOG_LEVEL" "default" "ERROR" "ERROR DEBUG INFO" "Qdrant log level passed to generated node configs"

# Collection creation
register_qdrant_var "COLLECTION_DOCUMENT" "default" "" "" "Path to a shard-builder input document; when set, the collection is created verbatim from its collection section and the env collection vars (VECTOR_DIM, DISTANCE_METRIC, HNSW_*, QUANTIZATION_*, MAX_SEGMENT_SIZE, DEFAULT_SEGMENT_NUMBER) are ignored"

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
register_qdrant_var "QUANTIZATION_RESCORE" "default" "" "" "Optional QUERY-only rescore override: empty omits QuantizationSearchParams so the server uses its per-quantization-type default; True/False forces rescoring on/off"


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

# Artifact install
register_qdrant_var "ARTIFACT_DIR" "default" "" "" "Directory of assembled shard-builder artifacts (shard_0/..shard_N-1); with TASK=LAUNCH the artifacts are installed into the cluster (create -> verify-config -> install-plan -> stop -> install -> relaunch -> verify) before serving. Requires STORAGE_MEDIUM=lustre, COLLECTION_DOCUMENT, and SHARD_BUILDER_BIN; mutually exclusive with RESTORE_DIR"
register_qdrant_var "INSTALL_MODE" "default" "move" "move copy" "How shards are installed from ARTIFACT_DIR: move renames each shard directory into place (seconds, but CONSUMES the artifacts -- keep a backup); copy preserves them at the cost of copying every byte"
register_qdrant_var "SHARD_BUILDER_BIN" "default" "" "" "Path to the qdrant-shard-builder binary (verify-config / install-plan / verify-placement); required when ARTIFACT_DIR is set, and must be built for the cluster's glibc"
register_qdrant_var "SCATTER_WORK_DIR" "default" "" "" "Scatter working directory the artifacts were built from; when set, verify-placement runs after the install, otherwise the exact point-count check stands in"
register_qdrant_var "CREATE_COLLECTION" "default" "False" "True False" "TASK=LAUNCH/STORM: create the collection (configure_collection.py) once the cluster is up; implied by ARTIFACT_DIR"

# Storm (TASK=STORM: drive nova-storm query workloads from the client node against the served
# collection -- WHATEVER provided it: ARTIFACT_DIR install, RESTORE_DIR tree, or CREATE_COLLECTION)
register_qdrant_var "NOVA_STORM_BIN" "default" "" "" "Path to the nova-storm binary (supernova); required when TASK=STORM, staged into the run dir, and must be built for the cluster's glibc (same Rocky-8 trick as the shard-builder binary)"
register_qdrant_var "STORM_CONFIG" "default" "" "" "Comma-separated nova-storm YAML config paths, frozen into the run dir at generation time and run SEQUENTIALLY against the served collection; each receives QDRANT_URL (rank 0 gRPC) and the exported run env (COLLECTION_NAME etc.) for its \${VAR} substitutions"
register_qdrant_var "STORM_HOLD" "default" "False" "True False" "TASK=STORM: after the storm configs finish, hold the cluster up (LAUNCH semantics, stop via flag.txt) instead of tearing down"
register_qdrant_var "STORM_REPEATS" "default" "1" "" "TASK=STORM: run the whole config list this many times; per-repeat JSONs are kept and storm_aggregate.py reports median/min/max per config (medians control for cache warm-up and run-to-run jitter)"
register_qdrant_var "STORM_TOP_K" "default" "10" "" "Comma-separated top_k values swept WITHIN the job for every storm config (e.g. 10,100,1000); each invocation exports STORM_TOP_K so YAMLs can use top_k: \${STORM_TOP_K} and ground_truth_column: hit_uuids_\${STORM_TOP_K}"
register_qdrant_var "STORM_BATCH_SIZE" "default" "1" "" "Comma-separated batch_size values swept WITHIN the job (YAMLs use batch_size: \${STORM_BATCH_SIZE})"
register_qdrant_var "STORM_CONCURRENCY" "default" "32" "" "Comma-separated concurrency values swept WITHIN the job (YAMLs use concurrency: \${STORM_CONCURRENCY})"
register_qdrant_var "STORM_SWEEP_QUERY_LIMIT" "default" "5000" "" "Queries loaded per SWEEP invocation (fixed-work = one pass over this many); caps the low-concurrency cells that would otherwise take an hour each at full file size. Winner recall passes always use the full file"
register_qdrant_var "STORM_TIMEOUT_S" "default" "300" "" "Client-side per-query gRPC timeout for storm's qdrant target (the client's own default is 5s -- far too short for a load test; a cancelled slow query is silently retried, doubling server work and misreporting the cell)"
register_qdrant_var "STORM_RESCORE" "default" "false" "true false" "TASK=STORM: quantization rescore, read by the dense/structured YAMLs as search_params.quantization.rescore (the sparse YAML has no quantized tier). false = search the quantized tier only, which stays resident in RAM; true = re-score candidates against the original float16 vectors, which at full scale means random reads from a ~1.5 TB/node matrix.dat that cannot be cached"
register_qdrant_var "STORM_RAG_PAYLOAD" "default" "True" "True False" "TASK=STORM: after the full-recall passes, one more FULL-FILE fixed-work pass per winner with a payload include-selector (STORM_RAG_FIELDS) -- the RAG-shaped workload where every hit returns its document body and payload reads (Lustre) join the measurement; results land as *_rag_rep1"
register_qdrant_var "STORM_RAG_FIELDS" "default" "text" "" "Comma-separated payload fields the RAG pass returns per hit (server-side include selector)"
register_qdrant_var "STORM_FULL_RECALL" "default" "True" "True False" "TASK=STORM: after the sweep, run one FULL-FILE fixed-work pass at the best-qps (batch, concurrency) per (config, k) (storm_pick_best.py) -- the citable exact recall@k over the entire query set, and a sustained full-scale run in its own right; results land as *_fullrecall_rep1"
register_qdrant_var "STORM_ORDER" "default" "rotate" "rotate fixed" "TASK=STORM with repeats: rotate the config order each repeat (each config samples every position, controlling for cache-locality carryover between workloads) or keep it fixed"


# Profiling
register_qdrant_var "PERF" "default" "NONE" "NONE STAT RECORD" "Performance collection mode"
register_qdrant_var "PERF_EVENTS" "default" "topdown-be-bound,topdown-mem-bound,topdown-retiring,topdown-fe-bound,topdown-bad-spec" "" "Comma-separated perf stat events"
register_qdrant_var "INSERT_TRACE" "default" "" "" "Optional insert trace file or mode"
register_qdrant_var "QUERY_TRACE" "default" "" "" "Optional query trace file or mode"
