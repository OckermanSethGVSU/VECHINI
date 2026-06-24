#!/home/seth-ockerman/Documents/basicEnv/bin/python3
"""
Verifies which dataset sample is loaded in each of the 5 clusters.

Checks:
  1. Whether each cluster's top-1000 results for 5 random queries match sample2's ground truth
  2. Whether the top-3 hit IDs from each sample's parquet actually exist in each cluster
  3. Whether all 5 clusters return identical result sets (i.e. same underlying data)
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pyarrow.parquet as pq
from urllib.parse import urlparse
from qdrant_client import QdrantClient
from qdrant_client.http.models import QueryRequest, SearchParams

# ── load env ──────────────────────────────────────────────────────────────────
with open('env') as f:
    env = {}
    for line in f:
        line = line.strip()
        if line.startswith('export '): line = line[7:]
        if '=' in line:
            k, _, v = line.partition('='); env[k.strip()] = v.strip()

# ── connect to all 5 clusters ─────────────────────────────────────────────────
print("Connecting to clusters...")
clients = {}
for idx in range(1, 6):
    url, key = env[f'QDRANT_URL_{idx}'], env[f'QDRANT_API_KEY_{idx}']
    parsed = urlparse(url)
    clients[idx] = QdrantClient(
        host=parsed.hostname, port=parsed.port - 1, grpc_port=parsed.port,
        api_key=key, prefer_grpc=True, https=True, timeout=60,
        grpc_options={'grpc.enable_http_proxy': 0},
    )
    info = clients[idx].get_collection('sample')
    print(f"  Cluster {idx} ({parsed.hostname[:30]}...): {info.points_count:,} points")

# ── load query embeddings ─────────────────────────────────────────────────────
queries = np.load('fineweb_query_embeddings_5000.embedding.npy').astype('float32')

# ── 1. recall vs each sample GT, for 5 test queries ──────────────────────────
TEST_ROWS = [0, 500, 1500, 3000, 4999]
print(f"\n=== Recall@1000 for {len(TEST_ROWS)} test queries ===")
print(f"  Columns = ground truth file used for scoring (1.000 on diagonal = cluster has correct data)")
print(f"{'':25s}" + "".join(f"   GT/s{i} " for i in range(1, 6)))

for cidx in range(1, 6):
    recalls = {sidx: [] for sidx in range(1, 6)}
    gts = {sidx: np.load(f'groundTruth/sample{sidx}.hit_ids.npy', mmap_mode='r') for sidx in range(1, 6)}

    for row in TEST_ROWS:
        resp = clients[cidx].query_batch_points('sample', [
            QueryRequest(query=queries[row].tolist(), limit=1000, using='dense',
                         params=SearchParams(exact=True),
                         with_payload=False, with_vector=False)
        ])
        returned = set(str(pt.id) for pt in resp[0].points)
        for sidx in range(1, 6):
            gt_set = set(gts[sidx][row, :1000].tolist()); gt_set.discard('')
            recalls[sidx].append(len(returned & gt_set) / 1000)

    row_str = f"  cluster {cidx} (queried exact): "
    for sidx in range(1, 6):
        avg = np.mean(recalls[sidx])
        row_str += f"  {avg:.3f}  "
    print(row_str)

print("\n  Rows = which cluster was queried. Columns = which sample's ground truth was used.")
print("  1.000 on the diagonal means the cluster contains that sample's documents.")

# ── 2. do sample GT point IDs physically exist in the clusters? ───────────────
print("\n=== Do sample GT document IDs actually exist in each cluster? ===")
print("(checking top-3 hit IDs from query 0 of each sample's parquet)")

for sidx in range(1, 6):
    tbl = pq.read_table(f'parquet/sample{sidx}.parquet')
    probe_ids = tbl['hit_ids'][0].as_py()[:3]
    found_in = []
    for cidx in range(1, 6):
        pts = clients[cidx].retrieve('sample', ids=probe_ids, with_payload=False)
        if len(pts) > 0:
            found_in.append(cidx)
    if found_in:
        print(f"  sample{sidx} IDs found in: cluster(s) {found_in}")
    else:
        print(f"  sample{sidx} IDs found in: NONE — documents not loaded into any cluster")

# # ── 3. are all clusters returning identical result sets? ──────────────────────
# print("\n=== Are all 5 clusters returning identical top-1000 results? ===")
# for row in TEST_ROWS:
#     sets = {}
#     for cidx in range(1, 6):
#         resp = clients[cidx].query_batch_points('sample', [
#             QueryRequest(query=queries[row].tolist(), limit=1000, using='dense',
#                          params=SearchParams(exact=True),
#                          with_payload=False, with_vector=False)
#         ])
#         sets[cidx] = set(str(pt.id) for pt in resp[0].points)
#     overlaps = [len(sets[1] & sets[cidx]) for cidx in range(2, 6)]
#     print(f"  query row {row:4d}: cluster1 vs 2/3/4/5 overlap = {overlaps} / 1000")
