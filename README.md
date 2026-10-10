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

## Usage

### 1. Compile all binaries

```bash
python3 run_bench.py --build
```

Or manually:

```bash
g++ -O2 -std=c++17 boruvka.cpp          -o boruvka
g++ -O2 -std=c++17 kruskal.cpp          -o kruskal
g++ -O2 -std=c++17 filter_kruskal.cpp   -o filter_kruskal
g++ -O2 -std=c++17 BMSBoruvka.cpp       -o bms
g++ -O2 -std=c++17 -fopenmp parallel_boruvka.cpp -o parallel_boruvka
g++ -O2 -std=c++17 bms_instrumented.cpp -o bms_xi
g++ -O2 -std=c++17 boruvka_xi.cpp       -o boruvka_xi
g++ -O2            gen.cpp              -o gen
```

### 2. Generate test data

```bash
python3 run_bench.py --gen
```

Default test cases (defined in `TEST_CASES`):

| Tag          | n         | m         | Notes                          |
| ------------ | --------- | --------- | ------------------------------ |
| `10k`        | 10,000    | 50,000    | sparse, m ≈ 5n                 |
| `100k`       | 100,000   | 500,000   | sparse                         |
| `500k`       | 500,000   | 2,500,000 | sparse                         |
| `1m`         | 1,000,000 | 5,000,000 | sparse, main experiment        |
| `dense_50k`  | 50,000    | 2,500,000 | dense, m ≈ 50n                 |
| `dense_100k` | 100,000   | 5,000,000 | dense                          |
| `1m_ties`    | 1,000,000 | 5,000,000 | uniform weights (many ties)    |
| `1m_small`   | 1,000,000 | 5,000,000 | small integer weights [1, 100] |

### 3. Run benchmark

```bash
# Full run (all sizes, all algorithms including par_T2 and par_T4)
python3 run_bench.py --build --gen --target-sec 60 --output-dir results/

# Specific sizes only
python3 run_bench.py --sizes 100k,1m

# ξ data only (no multi-trial timing)
python3 run_bench.py --xi-only

# Exclude parallel Borůvka (single-core machine)
python3 run_bench.py --no-par

# Custom thread counts
python3 run_bench.py --par-threads 2,4,8
```

### 4. Real-world graph (roadNet-CA)

```bash
# Download from SNAP: https://snap.stanford.edu/data/roadNet-CA.html
python3 convert_snap.py roadNet-CA.txt --output data/test_roadnet_ca.txt

# Then uncomment the roadnet_ca line in TEST_CASES in run_bench.py, and run:
python3 run_bench.py --sizes roadnet_ca
```

---

## Output Files

All written to `--output-dir` (default `./results/`):

| File               | Contents                                                     |
| ------------------ | ------------------------------------------------------------ |
| `bench_timing.csv` | Per-(size, algorithm): mean\_ms, min, max, σ, t̂, vs\_boruvka ratio, vs\_bms ratio |
| `bench_xi.csv`     | Per-(size, algorithm, repeat): ξ\_μ, ξ\_theory, steps/rounds, per-step ξ sequence |
| `bench_report.md`  | Human-readable Markdown report with ξ table, timing table, and convergence analysis |

## Project Structure

```
project/
|-- src/
|   |-- BMSBoruvka.cpp           # BMSBoruvka
|   |-- bms_instrumented.cpp     # BMS + ξ collection
|   |-- boruvka.cpp              # Borůvka
|   |-- boruvka_xi.cpp           # Borůvka + ξ collection
|   |-- filter_kruskal.cpp       # Filter-Kruskal (recursive pivot-based)
|   |-- gen.cpp                  # Random graph generator
|   |-- kruskal.cpp              # Standard Kruskal
|   |-- parallel_boruvka.cpp     # Parallel Borůvka
|   |-- convert_snap.py          # Convert SNAP road-network format to bench input
|   |-- run_bench.py             # Benchmark driver (build / gen / time / ξ / report)
|
|-- BMSSP/
|   |-- BMSSP.cpp                # reproduce BMSSP(learning purposes)
|
|-- README.md
```

## Paper

The `paper` directory now holds the files for building the paper, as well as the original `.docx` version.