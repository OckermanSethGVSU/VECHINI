"""Join the shard-builder's install-plan output with ip_registry.txt.

install-plan prints the shard-to-peer assignment consensus made, one line per shard:

    shard 3  ->  peer 1125417449097752  http://10.0.0.7:6635/

The URI is the peer's p2p address, which is exactly what each rank registered in
ip_registry.txt (rank,ip,p2p_port). The output is runtime_state/install_map.tsv
(rank<TAB>shard), consumed by install_shards.sh.

The mapping is always read back from the cluster rather than assumed: even with
REBALANCE_TOPOLOGY pinning shard i to rank i, consensus owns the assignment, and a
plan that disagrees with what the cluster actually decided would install shards
where no peer looks for them.
"""

import argparse
import re
import sys
from urllib.parse import urlparse

PLAN_LINE = re.compile(r"^\s*shard\s+(\d+)\s+->\s+peer\s+(\d+)\s+(\S+)")


def load_registry(path: str) -> dict:
    addr_to_rank = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rank, ip, p2p_port = line.split(",")
            addr_to_rank[(ip, int(p2p_port))] = int(rank)
    return addr_to_rank


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, help="Captured install-plan stdout")
    parser.add_argument("--registry", required=True, help="ip_registry.txt")
    parser.add_argument("--output", required=True, help="install_map.tsv to write")
    args = parser.parse_args()

    addr_to_rank = load_registry(args.registry)

    assignments = []
    with open(args.plan) as f:
        for line in f:
            m = PLAN_LINE.match(line)
            if not m:
                continue
            shard_id, peer_id, uri = int(m.group(1)), m.group(2), m.group(3)
            parsed = urlparse(uri)
            key = (parsed.hostname, parsed.port)
            if key not in addr_to_rank:
                print(
                    f"install-plan places shard {shard_id} on peer {peer_id} at "
                    f"{uri}, which matches no rank in {args.registry}",
                    file=sys.stderr,
                )
                return 2
            assignments.append((addr_to_rank[key], shard_id))

    if not assignments:
        print(f"no 'shard N -> peer' lines found in {args.plan}", file=sys.stderr)
        return 2

    shard_ids = sorted(shard for _, shard in assignments)
    if shard_ids != list(range(len(shard_ids))):
        print(f"install-plan shard ids are not contiguous from 0: {shard_ids}", file=sys.stderr)
        return 2

    assignments.sort()
    with open(args.output, "w") as f:
        for rank, shard in assignments:
            f.write(f"{rank}\t{shard}\n")

    for rank, shard in assignments:
        print(f"shard {shard} -> rank {rank}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
