#!/usr/bin/env python3
"""analyze_rocprof_join.py — Join trace data with rocprofiler counter/PC sampling data.

Combines:
  - Trace JSON (region timing from trace backend)
  - Counter JSON (hardware counters from rikp_counter_client)
  - PC sampling JSON (from rikp_pcsamp_client)
  - Code object analysis JSON (from rikp_codeobj_profiler)

Into a unified per-region analysis JSON.

Usage:
    python analyze_rocprof_join.py \
        --trace trace.json \
        --summary trace_summary.json \
        --counters rikp_counters.json \
        --pcsamp rikp_pcsamp.json \
        --codeobj rikp_codeobj_analysis.json \
        -o analysis.json
"""

import argparse
import json
import sys


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def join_data(trace_summary, counters=None, pcsamp=None, codeobj=None):
    """Join multiple data sources into a unified per-region analysis."""
    result = {
        "tool": "rocprof-intrakernel",
        "version": "0.1.0",
        "regions": [],
    }

    if not trace_summary:
        return result

    for region in trace_summary.get("regions", []):
        entry = {
            "region_id": region["region"],
            "name": region["name"],
            "timing": {
                "count": region["count"],
                "mean_dur_ns": region["mean_dur"],
                "min_dur_ns": region["min_dur"],
                "max_dur_ns": region["max_dur"],
                "cv": region.get("cv_dur"),
            },
        }

        # TODO: Correlate counters with regions by dispatch ID + timing overlap.
        if counters:
            entry["counters"] = {}

        # TODO: Correlate PC samples with regions by PC offset → code region mapping.
        if pcsamp:
            entry["pc_sampling"] = {"sample_count": 0}

        # TODO: Add static instruction mix from code object analysis.
        if codeobj:
            entry["instruction_mix"] = {}

        result["regions"].append(entry)

    return result


def main():
    parser = argparse.ArgumentParser(description="Join rocprof-intrakernel data sources")
    parser.add_argument("--trace", help="Trace JSON (not used directly, for reference)")
    parser.add_argument("--summary", required=True, help="Trace summary JSON")
    parser.add_argument("--counters", help="Counter collection JSON")
    parser.add_argument("--pcsamp", help="PC sampling JSON")
    parser.add_argument("--codeobj", help="Code object analysis JSON")
    parser.add_argument("-o", "--output", default="analysis.json", help="Output path")
    args = parser.parse_args()

    summary = load_json(args.summary)
    counters = load_json(args.counters) if args.counters else None
    pcsamp = load_json(args.pcsamp) if args.pcsamp else None
    codeobj = load_json(args.codeobj) if args.codeobj else None

    result = join_data(summary, counters, pcsamp, codeobj)

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Analysis written to {args.output}")


if __name__ == "__main__":
    main()
