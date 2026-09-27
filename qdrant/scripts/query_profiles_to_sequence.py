#!/usr/bin/env python3
"""Convert VECHINI query profile JSONL files for render_sequence.py.

Each input produces a sibling query_profile_sequence_rank_<rank>.json file. The
sequence is per client rank because batches from different ranks overlap.
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


INPUT_NAME = re.compile(r"query_profile_rank_(\d+)\.jsonl$")


def convert(source: Path) -> Path:
    match = INPUT_NAME.fullmatch(source.name)
    if match is None:
        raise ValueError(f"expected query_profile_rank_<rank>.jsonl: {source}")
    rank = int(match.group(1))
    target = source.with_name(f"query_profile_sequence_rank_{rank}.json")
    first_sent_us = None
    count = 0

    # Write incrementally: large profiles do not need to fit in memory.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=source.parent,
        prefix=f".{target.name}.", suffix=".tmp", delete=False,
    ) as output:
        temporary = Path(output.name)
        try:
            output.write("[\n")
            with source.open(encoding="utf-8") as records:
                for line_number, line in enumerate(records, 1):
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        timing = record["client_timing"]
                        sent_us = timing["sent_unix_us"]
                        rtt_us = timing["rtt_us"]
                        profile = record["profile"]
                        if record["rank"] != rank:
                            raise ValueError("rank differs from input filename")
                        if first_sent_us is None:
                            first_sent_us = sent_us
                        batch = {
                            "batch": count,
                            "rank": rank,
                            "first_query_row": record["first_query_row"],
                            "query_count": record["query_count"],
                            "client": timing,
                            "client_sent_us": sent_us - first_sent_us,
                            "client_recv_us": sent_us - first_sent_us + rtt_us,
                            "client_sent_unix_us": sent_us,
                            "client_recv_unix_us": timing["recv_unix_us"],
                            "profile": profile,
                        }
                    except (KeyError, TypeError, ValueError) as error:
                        raise ValueError(f"{source}:{line_number}: {error}") from error
                    if count:
                        output.write(",\n")
                    json.dump(batch, output, separators=(",", ":"))
                    count += 1
            output.write("\n]\n")
            output.flush()
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    args = parser.parse_args()
    for source in args.profiles:
        print(convert(source))


if __name__ == "__main__":
    main()
