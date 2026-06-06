# Milvus

Milvus is an active engine in the unified submit workflow and supports both
`RUN_MODE=PBS` and `RUN_MODE=LOCAL`.

## Main Files

- [milvus/engine.sh](../../milvus/engine.sh)
- [milvus/schema.sh](../../milvus/schema.sh)
- [milvus/main.sh](../../milvus/main.sh)
- [milvus/local_main.sh](../../milvus/local_main.sh)

## Active Tasks

- `INSERT`
- `INDEX`
- `QUERY`
- `IMPORT`
- `MIXED`

## Operational Notes

- PBS runs require `ENV_PATH` unless `ALLOW_SYSTEM_PYTHON=True`.
- `MODE` can be `STANDALONE` or `DISTRIBUTED`.
- `MINIO_MODE` is derived automatically when unset.
- `ETCD_MEDIUM` defaults to `STORAGE_MEDIUM` when unset.
- Bulk ingest behavior is controlled by Milvus-specific upload and transport
  variables.
- Mixed runs derive `INSERT_START_ID` when possible.

## Build Commands

```bash
cd milvus/clients
./build.sh batch_client
./build.sh mixed
```
