# General Utilities

## Recall

`compute_recall.py` compares a two-dimensional ground-truth neighbor-ID NPY
matrix with a two-dimensional query-result ID NPY matrix:

```bash
python3 utils/compute_recall.py \
  path/to/ground_truth_ids.npy \
  path/to/query_result_ids.npy \
  10
```

The third argument is `top_k`. The command writes `recall.csv` in the current
directory by default. Select another output path with `--output`:

```bash
python3 utils/compute_recall.py ground_truth.npy query_ids.npy 100 \
  --output results/recall.csv
```

Both files must contain C-order, two-dimensional integer arrays with the same
number of query rows. Returned IDs of `-1` are treated as missing results.

Milvus and Weaviate may produce one query-result matrix per client. Pass all of
them before `top_k`; the utility concatenates files in worker/client order:

```bash
python3 utils/compute_recall.py ground_truth.npy \
  queryNPY/query_result_ids_w*_c*.npy 10
```

Recall for each query is:

```text
unique ground-truth IDs intersecting returned IDs / top_k
```

The output contains aggregate recall, minimum and maximum per-query recall,
standard deviation, total matches, and the number of queries with perfect
recall.

## Workflow Integration

Qdrant, Milvus, and Weaviate query workflows can run the utility automatically:

```bash
TASK=QUERY
CALCULATE_RECALL=True
GROUND_TRUTH_FILEPATH=/path/to/ground_truth_ids.npy
TOP_K=10
```

When enabled, the unified submit manager stages `compute_recall.py` in the
generated run directory. After the query workflow finishes, the runtime writes
`recall.csv` in that run directory.
