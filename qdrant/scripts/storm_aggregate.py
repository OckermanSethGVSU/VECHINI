"""Aggregate repeated nova-storm runs into per-config medians.

Reads stormResults/<config>_rep<k>.json (one JSON summary per repeat, written
by main.sh's run_nova_storm), groups by config, and writes
stormResults/summary.csv plus a printed table with, per config and metric,
the MEDIAN across repeats and the min..max spread.

Median is the right cross-run summary: within one run the percentiles come
from the full latency sample (never averaged), and across runs the median of
p50s/p95s is robust to a cold-cache first repeat or a one-off hiccup. Recall
is aggregated the same way but its spread doubles as a stability check — a
deterministic search should not move between repeats, so a non-trivial
min..max range on recall is itself a finding (reported loudly).
"""

import csv
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

METRICS = ["qps", "p50_ms", "p95_ms", "p99_ms", "max_ms", "requests", "errors", "timeouts"]
REP = re.compile(r"^(?P<name>.+)_rep(?P<k>\d+)$")


def recall_of(summary: dict):
    bucket = summary.get("total_recall")
    return bucket["mean"] if bucket else None


def main() -> int:
    results_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "stormResults")
    groups = defaultdict(list)
    for path in sorted(results_dir.glob("*_rep*.json")):
        m = REP.match(path.stem)
        if not m:
            continue
        try:
            groups[m.group("name")].append((int(m.group("k")), path.name, json.loads(path.read_text())))
        except json.JSONDecodeError:
            print(f"[storm-aggregate] skipping unparseable {path.name} (failed repeat?)", file=sys.stderr)

    if not groups:
        print("[storm-aggregate] no *_rep*.json files found; nothing to aggregate", file=sys.stderr)
        return 1

    rows = []
    unstable_recall = []
    for name, runs in sorted(groups.items()):
        runs.sort()
        summaries = [s for _, _, s in runs]
        row = {"config": name, "repeats": len(summaries)}
        for metric in METRICS:
            # A summary missing an expected metric is malformed output or a
            # stale nova-storm binary — either invalidates the aggregation,
            # so fail loudly instead of writing a summary.csv with holes.
            missing = [fname for _, fname, s in runs if metric not in s]
            if missing:
                print(
                    f"[storm-aggregate] ERROR: field '{metric}' missing from "
                    f"{', '.join(missing)} — malformed summary or stale nova-storm binary",
                    file=sys.stderr,
                )
                return 1
            values = [s[metric] for s in summaries]
            row[f"{metric}_median"] = statistics.median(values)
            row[f"{metric}_min"] = min(values)
            row[f"{metric}_max"] = max(values)
        recalls = [r for r in (recall_of(s) for s in summaries) if r is not None]
        if recalls:
            row["recall_median"] = statistics.median(recalls)
            row["recall_min"] = min(recalls)
            row["recall_max"] = max(recalls)
            # PER-QUERY recall is deterministic, but the run-level mean wobbles
            # a little across repeats: a timed run cuts off mid-rotation, so
            # the queries get slightly unequal firing counts and the mean
            # shifts by ~1/n_firings. The threshold must sit above that
            # composition wobble (observed ~2e-5 at 30k firings) and below
            # anything a genuinely nondeterministic search would produce.
            if max(recalls) - min(recalls) > 1e-3:
                unstable_recall.append((name, min(recalls), max(recalls)))
        rows.append(row)

    out = results_dir / "summary.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[storm-aggregate] {out}")
    for row in rows:
        recall = (
            f"recall {row['recall_median']:.4f} [{row['recall_min']:.4f}..{row['recall_max']:.4f}]"
            if "recall_median" in row
            else "recall n/a"
        )
        print(
            f"[storm-aggregate] {row['config']}: n={row['repeats']}  "
            f"qps {row['qps_median']:.0f} [{row['qps_min']:.0f}..{row['qps_max']:.0f}]  "
            f"p50 {row['p50_ms_median']:.2f}ms  p99 {row['p99_ms_median']:.2f}ms  "
            f"errors(median) {row['errors_median']:.0f}  {recall}"
        )
    for name, lo, hi in unstable_recall:
        print(
            f"[storm-aggregate] WARNING: recall UNSTABLE across repeats for {name}: "
            f"{lo:.6f}..{hi:.6f} — deterministic search should not move; investigate",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
