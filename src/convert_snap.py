#!/usr/bin/env python3
"""
convert_snap.py  —  Convert SNAP road-network format to bench input format

Usage:
    python3 convert_snap.py roadNet-CA.txt [--seed 42] [--weights random|small|uniform]
                            [--output roadnet_ca.txt]

Input format (SNAP):
    # comment lines starting with #
    FromNodeId<TAB>ToNodeId
    (directed edges; we treat each as undirected)

Output format (bench input):
    n m
    u v w   (0-indexed, one edge per line)

Notes:
    - Duplicate edges (u,v) and (v,u) are deduplicated (keep one direction).
    - Self-loops are removed.
    - Node IDs are re-mapped to 0-indexed contiguous integers.
    - Edge weights are assigned randomly since road networks in SNAP have no weights.
      Use --weights small for integer weights in [1, 100] (many ties),
          --weights uniform for all-ones,
          --weights random (default) for uniform in [1, 10^9].
"""

import sys
import random
import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="SNAP .txt file (e.g. roadNet-CA.txt)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--weights", choices=["random", "small", "uniform"], default="random")
    p.add_argument("--output", default=None, help="Output file (default: <input_stem>.bench.txt)")
    args = p.parse_args()

    rng = random.Random(args.seed)

    def gen_weight():
        if args.weights == "uniform": return 1
        if args.weights == "small":   return rng.randint(1, 100)
        return rng.randint(1, 10**9)

    # ── Read edges ────────────────────────────────────────────────────────────
    raw_edges = set()
    node_set  = set()

    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            u, v = int(parts[0]), int(parts[1])
            if u == v:
                continue                        # skip self-loops
            # treat as undirected: canonical form (min, max)
            raw_edges.add((min(u, v), max(u, v)))
            node_set.add(u)
            node_set.add(v)

    # ── Re-index nodes to 0..n-1 ──────────────────────────────────────────────
    node_list = sorted(node_set)
    node_map  = {old: new for new, old in enumerate(node_list)}
    n = len(node_list)
    m = len(raw_edges)

    # ── Output ────────────────────────────────────────────────────────────────
    out_path = args.output or (Path(args.input).stem + ".bench.txt")
    with open(out_path, "w") as f:
        f.write(f"{n} {m}\n")
        for (u_raw, v_raw) in sorted(raw_edges):
            u = node_map[u_raw]
            v = node_map[v_raw]
            w = gen_weight()
            f.write(f"{u} {v} {w}\n")

    print(f"Converted: n={n:,}  m={m:,}  weights={args.weights}  seed={args.seed}")
    print(f"Output:    {out_path}")


if __name__ == "__main__":
    main()
