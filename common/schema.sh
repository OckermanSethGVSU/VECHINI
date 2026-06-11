#!/bin/bash

# Shared schema entries used across all engines.
# register_common_var "NAME" "REQUIREMENT" "DEFAULT" "CHOICES" "DESCRIPTION" ["REQUIRED_IF"]

# High-level control variables
register_common_var "TASK" "required" "" "" "Experiment task"
register_common_var "RUN_MODE" "default" "PBS" "PBS LOCAL local" "Run under PBS or create a local harness"

# Platform resource controls
register_common_var "PLATFORM" "conditional" "" "POLARIS AURORA" "Target platform" "RUN_MODE=PBS"
register_common_var "NODES" "conditional" "" "" "Compute-node count to allocate for worker ranks" "RUN_MODE=PBS"
register_common_var "CORES" "default" "" "" "CPU cores assigned per worker rank; empty disables explicit CPU binding"
register_common_var "STORAGE_MEDIUM" "default" "memory" "memory DAOS lustre SSD" "Storage medium for engine data"
register_common_var "DAOS_PROJECT" "conditional" "" "" "DAOS project/pool name" "STORAGE_MEDIUM=DAOS"
register_common_var "DAOS_CONTAINER" "conditional" "" "" "DAOS container name" "STORAGE_MEDIUM=DAOS"


# PBS variables
register_common_var "ACCOUNT" "conditional" "" "" "PBS project/account to charge for the run" "RUN_MODE=PBS"
register_common_var "WALLTIME" "conditional" "" "" "PBS walltime" "RUN_MODE=PBS"
register_common_var "QUEUE" "conditional" "" "" "PBS queue name" "RUN_MODE=PBS"
register_common_var "ENV_PATH" "default" "" "" "Python environment path"
register_common_var "ALLOW_SYSTEM_PYTHON" "default" "False" "True False" "Allow PBS runs to use the already-loaded Python environment when ENV_PATH is empty"

# Collection Variables
register_common_var "COLLECTION_NAME" "default" "DEFAULT_COLLECTION" "" "Optional collection override"
register_common_var "VECTOR_DIM" "required" "" "" "Vector dimension"
register_common_var "DISTANCE_METRIC" "default" "COSINE" "IP COSINE L2" "Distance metric"


# INSERT variables
register_common_var "INSERT_DATA_FILEPATH" "conditional" "" "" "Insert corpus file path" "TASK=INSERT|INDEX|QUERY|MIXED"
register_common_var "INSERT_CORPUS_SIZE" "default" "" "" "Total vectors available to preload; empty means use all rows in the file"
register_common_var "INSERT_BATCH_SIZE" "default" "512" "" "Insert batch size; single value or sweep list"
register_common_var "INSERT_BALANCE_STRATEGY" "default" "WORKER_BALANCE" "NO_BALANCE WORKER_BALANCE" "Insert balancing policy"
register_common_var "INSERT_STREAMING" "default" "False" "True False" "Enable streaming insert behavior"

# QUERY variables
register_common_var "QUERY_DATA_FILEPATH" "conditional" "" "" "Query vector file path" "TASK=QUERY"
register_common_var "QUERY_CORPUS_SIZE" "default" "" "" "Total queries to execute; empty means use all rows in the file"
register_common_var "QUERY_BATCH_SIZE" "default" "32" "" "Query batch size; single value or sweep list"
register_common_var "QUERY_BALANCE_STRATEGY" "conditional" "NO_BALANCE" "NO_BALANCE WORKER_BALANCE" "Query balancing policy" "TASK=QUERY"
register_common_var "QUERY_STREAMING" "default" "False" "True False" "Enable query streaming behavior"
register_common_var "TOP_K" "default" "10" "" "Optional top-k override"
register_common_var "CALCULATE_RECALL" "default" "False" "True False" "Calculate recall.csv after a QUERY workflow"
register_common_var "GROUND_TRUTH_FILEPATH" "conditional" "" "" "Ground-truth neighbor ID NPY matrix used for recall calculation" "CALCULATE_RECALL=True"

# MIXED variables
register_common_var "MIXED_INSERT_DATA_FILEPATH" "conditional" "" "" "Mixed insert corpus file path" "TASK=MIXED"
register_common_var "MIXED_QUERY_DATA_FILEPATH" "conditional" "" "" "Mixed query vector file path" "TASK=MIXED"
register_common_var "MIXED_INSERT_MODE" "default" "MAX" "MAX RATE max rate" "Mixed insert pacing mode"
register_common_var "MIXED_QUERY_MODE" "default" "MAX" "MAX RATE max rate" "Mixed query pacing mode"
register_common_var "MIXED_INSERT_BATCH_SIZE" "default" "32" "" "Mixed insert batch size"
register_common_var "MIXED_QUERY_BATCH_SIZE" "default" "32" "" "Mixed query batch size"
register_common_var "MIXED_INSERT_CORPUS_SIZE" "default" "" "" "Mixed insert corpus size; empty means use all rows in the file"
register_common_var "MIXED_QUERY_CORPUS_SIZE" "default" "" "" "Mixed query corpus size; empty means use all rows in the file"
register_common_var "MIXED_INSERT_CLIENTS" "default" "1" "" "Mixed insert clients"
register_common_var "MIXED_QUERY_CLIENTS" "default" "1" "" "Mixed query clients"

# Path variables
register_common_var "BASE_DIR" "default" "" "" "Base directory containing generated run directories; auto-filled by the submit manager when empty"
