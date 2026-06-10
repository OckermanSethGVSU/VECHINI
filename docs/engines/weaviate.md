# Weaviate

Weaviate is an active engine in the unified submit workflow and supports both
`RUN_MODE=PBS` and `RUN_MODE=LOCAL` in the current tracked workflow.

## Main Files

- [weaviate/engine.sh](../../weaviate/engine.sh)
- [weaviate/schema.sh](../../weaviate/schema.sh)
- [weaviate/main.sh](../../weaviate/main.sh)
- [weaviate/local_main.sh](../../weaviate/local_main.sh)
- [weaviate/scripts/create_basic_collection.py](../../weaviate/scripts/create_basic_collection.py)

## Active Tasks

- `INSERT`
- `INDEX`
- `QUERY`

## Operational Notes

- The runtime creates the collection before running the batch client.
- Dynamic vector indexing uses `create_basic_collection.py`.
- `DISTANCE_METRIC` is mapped in that script and now pinned into the dynamic,
  HNSW, and flat index configs.
- `HNSW_DYNAMIC_THRESHOLD` is derived from `INSERT_CORPUS_SIZE` or
  `INSERT_DATA_FILEPATH` when unset.

## Build Commands

```bash
cd weaviate/clients
./build.sh batch_client
./build.sh mixed
```
