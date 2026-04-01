#!/usr/bin/env python3
"""plot_trace_summary.py — Visualize trace summary statistics.

Usage:
    python plot_trace_summary.py trace_summary.json [-o output.png]
"""

import argparse
import json
import sys

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_summary(summary, output_path=None):
    if not HAS_MPL:
        print("matplotlib not available. Install with: pip install matplotlib numpy",
              file=sys.stderr)
        # Fall back to text summary.
        for r in summary.get("regions", []):
            print(f"  {r['name']:20s}  mean={r['mean_dur']:10.1f}ns  "
                  f"n={r['count']:8d}  cv={r.get('cv_dur', 'N/A')}")
        return

    regions = summary.get("regions", [])
    if not regions:
        print("No regions found in summary.", file=sys.stderr)
        return

    names = [r["name"] for r in regions]
    means = [r["mean_dur"] for r in regions]

    fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.5)))
    bars = ax.barh(names, means, color="#569cd6")
    ax.set_xlabel("Mean duration (ns)")
    ax.set_title("rocprof-intrakernel: Region Timing Summary")
    ax.invert_yaxis()

    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + max(means) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.0f}", va="center", fontsize=8, color="#d4d4d4")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor="#1e1e1e", edgecolor="none")
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot trace summary")
    parser.add_argument("summary", help="Trace summary JSON file")
    parser.add_argument("-o", "--output", help="Output image path (default: show)")
    args = parser.parse_args()

    summary = load_json(args.summary)
    plot_summary(summary, args.output)


if __name__ == "__main__":
    main()
