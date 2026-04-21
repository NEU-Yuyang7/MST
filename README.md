# BMSBoruvka

## Overview

This project explores a novel approach to the **Minimum Spanning Tree (MST)** problem by integrating ideas from bucket-based classification and Borůvka-style graph contraction.
Inspired by recent advances in shortest path algorithms that reduce reliance on comparison-based sorting, this work investigates whether similar techniques can be applied to MST construction.

## Method

This project implements **BMSBoruvka**, a deterministic minimum spanning tree algorithm that achieves $O(n log^{\frac{2}{3}} n)$ time on sparse graphs $(m = O(n))$ without any global sort.

## Algorithm

The algorithm runs in P = ⌈log(n) / t⌉ super-steps, where t = ⌈(log₂ n)^(2/3)⌉. Each super-step has three phases:

- **Phase A — Edge activation.** `std::nth_element` partitions the unactivated edge suffix so the lightest k = m/P edges are selected. This preserves the cut property without sorting, at O(m − i·k) cost per step.
- **Phase B — Borůvka rounds.** t rounds of Borůvka's algorithm are applied to the activated edge set. Each round halves the component count, so after t rounds the number of components shrinks by a factor of 2^t.
- **Phase C — Graph compression.** The contracted graph is rebuilt over the surviving components, reducing the working edge set for the next super-step.

## Benchmarking

All benchmark files are in `src/`. Run from that directory:

```bash
cd src

# First run: compile all binaries and generate test data
python3 run_bench.py --build --gen

# Subsequent runs (data already exists)
python3 run_bench.py

# Sparse graphs only, 25-second budget per size
python3 run_bench.py --sizes 10k,100k,500k,1m --target-sec 25

# Collect ξ shrinkage data only (no timing)
python3 run_bench.py --xi-only
```

Results are written to `src/results/`:
- `bench_timing.csv` — mean, σ, and t̂ for each algorithm and size
- `bench_xi.csv`     — per-step ξ shrinkage factor data
- `bench_report.md`  — summary tables in Markdown

**Requirements:** `g++` with C++17 and OpenMP support (`-fopenmp`).

## Project Structure

```
project/
│── src/
│   ├── bms_instrumented.cpp     # BMS + ξ collection
│   ├── BMSBoruvka.cpp           # BMSBoruvka
│   ├── Boruvka.cpp              # Borůvka
│   ├── boruvka_xi.cpp           # Borůvka + ξ collection
│   ├── gen.cpp                  # Random graph generator
│   ├── parallel_boruvka.cpp     # Parallel borůvka
│   ├── run_bench.py             # Benchmark script
│
│── BMSSP
│   ├── BMSSP.cpp                # reproduce BMSSP(learning purposes)
│
│── README.md
```

