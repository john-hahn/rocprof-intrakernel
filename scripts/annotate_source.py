#!/usr/bin/env python3
"""annotate_source.py — Annotate HIP source with profiling data from trace summary.

Usage:
    python annotate_source.py --source my_kernel.hip --summary trace_summary.json
"""

import argparse
import json
import re
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def annotate(source_path, summary):
    """Print source with inline annotations for region timing."""
    regions = {r["name"]: r for r in summary.get("regions", [])}

    with open(source_path, "r") as f:
        lines = f.readlines()

    # Look for RIKP_TRACE_REC_B / RIKP_TRACE_REC_E patterns.
    begin_re = re.compile(r"RIKP_TRACE_REC_B\s*\(.*?,\s*.*?,\s*(\w+)\s*\)")
    end_re = re.compile(r"RIKP_TRACE_REC_E\s*\(.*?,\s*.*?,\s*(\w+)\s*\)")

    for i, line in enumerate(lines, 1):
        stripped = line.rstrip()
        annotation = ""

        m = begin_re.search(line)
        if m:
            region_id = m.group(1)
            for name, data in regions.items():
                if str(data["region"]) == region_id or name == region_id:
                    annotation = f"  // [BEGIN {name}: mean={data['mean_dur']:.0f}ns, n={data['count']}]"
                    break

        m = end_re.search(line)
        if m:
            region_id = m.group(1)
            for name, data in regions.items():
                if str(data["region"]) == region_id or name == region_id:
                    annotation = f"  // [END {name}]"
                    break

        print(f"{stripped}{annotation}")


def main():
    parser = argparse.ArgumentParser(description="Annotate HIP source with profiling data")
    parser.add_argument("--source", required=True, help="HIP source file")
    parser.add_argument("--summary", required=True, help="Trace summary JSON")
    args = parser.parse_args()

    summary = load_json(args.summary)
    annotate(args.source, summary)


if __name__ == "__main__":
    main()
