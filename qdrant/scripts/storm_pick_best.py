"""Pick the best (batch, concurrency) per (config, k) from a storm sweep.

Reads stormResults/summary.csv (written by storm_aggregate.py after the
fixed-work sweep), considers only sweep rows (``<config>_k<k>_b<b>_c<c>`` —
the ``_besttimed``/``_fullrecall``/``_timed`` rows from later phases are
excluded by the regex), groups by (config, k), and emits one line per group:

    <config> <k> <b> <c>

Selection is max ``qps_median``. Latency is REPORTED next to each winner (and
for the runner-up) on stderr so an operator can spot a winner that bought its
throughput with ugly p99 — but the selection itself stays throughput-only,
matching what the sweep is for: finding the peak operating point.
"""

import csv
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^(?P<name>.+)_k(?P<k>\d+)_b(?P<b>\d+)_c(?P<c>\d+)$")


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "stormResults/summary.csv"
    groups = defaultdict(list)
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            m = ROW.match(row["config"])
            if not m:
                continue
            groups[(m.group("name"), int(m.group("k")))].append(
                (
                    float(row["qps_median"]),
                    float(row["p99_ms_median"]),
                    int(m.group("b")),
                    int(m.group("c")),
                )
            )

    if not groups:
        print(f"no sweep rows found in {path}", file=sys.stderr)
        return 1

    for (name, k), rows in sorted(groups.items()):
        rows.sort(reverse=True)
        qps, p99, b, c = rows[0]
        note = f"[pick-best] {name} k={k}: b={b} c={c} (qps {qps:.0f}, p99 {p99:.2f}ms)"
        if len(rows) > 1:
            q2, p2, b2, c2 = rows[1]
            note += f"; runner-up b={b2} c={c2} (qps {q2:.0f}, p99 {p2:.2f}ms)"
        print(note, file=sys.stderr)
        print(f"{name} {k} {b} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
