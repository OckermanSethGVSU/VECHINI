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
                    # cells with client-side timeouts have corrupt qps AND
                    # censored tails -- they must not win on those numbers
                    float(row.get("timeouts_max", 0) or 0),
                )
            )

    if not groups:
        print(f"no sweep rows found in {path}", file=sys.stderr)
        return 1

    for (name, k), rows in sorted(groups.items()):
        clean = [r for r in rows if r[4] == 0]
        if not clean:
            # every cell timed out at least once: pick the least-contaminated
            # (fewest timeouts, then qps) and say so loudly
            rows.sort(key=lambda r: (r[4], -r[0]))
            qps, p99, b, c, t = rows[0]
            print(
                f"[pick-best] WARNING: {name} k={k}: EVERY cell had client timeouts; "
                f"choosing least-contaminated b={b} c={c} ({t:.0f} timeouts, qps {qps:.0f} "
                f"-- qps/p99 at this cell are unreliable; consider raising STORM_TIMEOUT_S)",
                file=sys.stderr,
            )
            print(f"{name} {k} {b} {c}")
            continue
        skipped = len(rows) - len(clean)
        clean.sort(reverse=True)
        qps, p99, b, c, _ = clean[0]
        note = f"[pick-best] {name} k={k}: b={b} c={c} (qps {qps:.0f}, p99 {p99:.2f}ms)"
        if len(clean) > 1:
            q2, p2, b2, c2, _ = clean[1]
            note += f"; runner-up b={b2} c={c2} (qps {q2:.0f}, p99 {p2:.2f}ms)"
        if skipped:
            note += f"; {skipped} cell(s) excluded for client timeouts"
        print(note, file=sys.stderr)
        print(f"{name} {k} {b} {c}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
