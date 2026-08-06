"""Verify an artifact install before the cluster is handed to query clients.

Every failure mode this checks for is otherwise silent: qdrant deletes a segment
directory it considers damaged and comes up green serving less data, and a missing
shard still reports a green collection. Status alone proves nothing.

Checks, in order:
  1. collection reaches green (bounded wait);
  2. every shard 0..N-1 is local on exactly one peer (queried per peer -- the
     shard-scoped view is local-only and not forwarded);
  3. the summed per-shard point counts equal EXPECTED_CORPUS_SIZE, cross-checked
     against the collection-level count. Both use qdrant's estimate path
     (exact=false), which for a filterless count is the id tracker's
     available_point_count -- O(segments), instant, and EXACT for freshly built
     artifacts (no deleted points). exact=true is deliberately avoided: it
     materializes every point id in the shard (read_filtered -> BTreeSet), which
     at a billion points per shard is tens of GB of RAM and a guaranteed timeout;
  4. smoke: scroll one point with its vector, query it back, and expect its own id
     in the top hits (README-verified property of the artifacts: the probe point
     returns as the top hit through quantized search).

Raw REST via urllib on purpose -- same reasoning as configure_collection.py's
document mode: no client-version model drift, no extra dependency.

Env: COLLECTION_NAME, EXPECTED_CORPUS_SIZE; ip_registry.txt in the cwd.
Exits non-zero on the first failed check.
"""

import json
import os
import sys
import time
import urllib.request

GREEN_TIMEOUT_S = int(os.getenv("VERIFY_GREEN_TIMEOUT_S", "3600"))
COUNT_TIMEOUT_S = int(os.getenv("VERIFY_COUNT_TIMEOUT_S", "1800"))


def rest(method: str, url: str, payload: dict | None = None, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": "application/json"}
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def load_nodes(path: str = "ip_registry.txt") -> list:
    nodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rank, ip, p2p_port = line.split(",")
            nodes.append((int(rank), f"http://{ip}:{int(p2p_port) - 2}"))
    nodes.sort()
    return nodes


def fail(message: str) -> None:
    print(f"[verify-restore] FAIL: {message}", flush=True)
    sys.exit(1)


def main() -> None:
    collection = os.environ["COLLECTION_NAME"].strip()
    expected = int(os.environ["EXPECTED_CORPUS_SIZE"])
    nodes = load_nodes()
    head = nodes[0][1]

    # 1. green, with a bounded wait -- shard loading at startup is not instant.
    deadline = time.time() + GREEN_TIMEOUT_S
    status = None
    while time.time() < deadline:
        try:
            info = rest("GET", f"{head}/collections/{collection}")["result"]
            status = info["status"]
            if status == "green":
                break
        except Exception as e:
            print(f"[verify-restore] waiting for collection: {e}", flush=True)
        time.sleep(10)
    if status != "green":
        fail(f"collection status is {status!r} after {GREEN_TIMEOUT_S}s")
    print("[verify-restore] collection is green", flush=True)

    # 2. shard placement: every shard local on exactly one peer, counts per shard.
    shard_owner: dict = {}
    total_from_shards = 0
    for rank, base in nodes:
        cluster = rest("GET", f"{base}/collections/{collection}/cluster")["result"]
        local = cluster.get("local_shards") or []
        shard_list = [(s["shard_id"], s.get("points_count", 0)) for s in local]
        print(f"[verify-restore] rank {rank} ({base}): local shards {shard_list}", flush=True)
        for shard_id, points in shard_list:
            if shard_id in shard_owner:
                fail(
                    f"shard {shard_id} is local on rank {shard_owner[shard_id]} AND "
                    f"rank {rank} -- duplicate install, writes would diverge"
                )
            shard_owner[shard_id] = rank
            total_from_shards += points

    shard_count = len(shard_owner)
    if sorted(shard_owner) != list(range(shard_count)):
        fail(f"shards present: {sorted(shard_owner)} -- not contiguous from 0")
    print(f"[verify-restore] {shard_count} shards, each on exactly one peer", flush=True)

    # 3. counts. Nine of ten shards installed still reports green; the count is the
    # signal. The per-shard sum (from each peer's own local view) is authoritative;
    # the collection-level count cross-checks that the fan-out view agrees, which
    # catches a shard the head node cannot reach. Both are estimate-path counts --
    # instant at any scale, and exact for these no-deletes artifacts. exact=true is
    # NOT used: it materializes every point id per shard (tens of GB at 10B scale).
    if total_from_shards != expected:
        fail(f"per-shard point sum {total_from_shards:,} != expected {expected:,}")
    fanout = rest(
        "POST",
        f"{head}/collections/{collection}/points/count",
        {"exact": False},
        timeout=COUNT_TIMEOUT_S,
    )["result"]["count"]
    if fanout != expected:
        fail(f"collection-level count {fanout:,} != expected {expected:,}")
    print(f"[verify-restore] point count OK: {fanout:,}", flush=True)

    # 4. smoke query: fetch any point with its vector, search it back.
    scroll = rest(
        "POST",
        f"{head}/collections/{collection}/points/scroll",
        {"limit": 1, "with_payload": True, "with_vector": True},
    )["result"]["points"]
    if not scroll:
        fail("scroll returned no points")
    probe = scroll[0]
    vectors = probe["vector"]
    if isinstance(vectors, dict):
        vector_name, vector = next(iter(vectors.items()))
    else:
        vector_name, vector = None, vectors
    query: dict = {"query": vector, "limit": 5, "with_payload": False}
    if vector_name is not None:
        query["using"] = vector_name
    hits = rest(
        "POST", f"{head}/collections/{collection}/points/query", query
    )["result"]["points"]
    hit_ids = [h["id"] for h in hits]
    if probe["id"] not in hit_ids:
        fail(f"probe point {probe['id']} not in its own top-5 ({hit_ids})")
    print(
        f"[verify-restore] smoke query OK: point {probe['id']} found via "
        f"vector {vector_name or '(unnamed)'}",
        flush=True,
    )

    print("[verify-restore] PASS", flush=True)


if __name__ == "__main__":
    main()
