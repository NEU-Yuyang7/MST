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
g++ -O2 -std=c++17 -DBMS_XI BMSBoruvka.cpp -o bms_xi
g++ -O2 -std=c++17 boruvka_xi.cpp       -o boruvka_xi
g++ -O2            gen.cpp              -o gen
```

### 2. Generate test data

```bash
python3 run_bench.py --gen
```

Default test cases (defined in `TEST_CASES`):

Density is set by mean degree `d̄ = 2m/n`, against two thresholds: `ln n`, the
Erdős–Rényi connectivity threshold, and `√n`, where mean degree outgrows any
polylog in `n`.

| Tag            | n         | m          | `d̄`  | Notes                          |
| -------------- | --------- | ---------- | ----: | ------------------------------ |
| `logn_10k`     | 10,000    | 92,103     |  18.4 | d̄ = 2 ln n                     |
| `logn_30k`     | 30,000    | 309,268    |  20.6 | d̄ = 2 ln n                     |
| `logn_100k`    | 100,000   | 1,151,292  |  23.0 | d̄ = 2 ln n                     |
| `logn_300k`    | 300,000   | 3,783,461  |  25.2 | d̄ = 2 ln n                     |
| `logn_1m`      | 1,000,000 | 13,815,510 |  27.6 | d̄ = 2 ln n, convergence series |
| `rootlo_30k`   | 30,000    | 1,299,038  |  86.6 | d̄ = √n/2                       |
| `roothi_30k`   | 30,000    | 5,196,152  | 346.4 | d̄ = 2√n, dense                 |
| `rootlo_100k`  | 100,000   | 7,905,694  | 158.1 | d̄ = √n/2                       |
| `roothi_100k`  | 100,000   | 31,622,776 | 632.5 | d̄ = 2√n, dense                 |
| `1m`           | 1,000,000 | 5,000,000  |  10.0 | weight-distribution control    |
| `1m_ties`      | 1,000,000 | 5,000,000  |  10.0 | uniform weights (many ties)    |
| `1m_small`     | 1,000,000 | 5,000,000  |  10.0 | small integer weights [1, 100] |

The `logn_*` series is the n-scaling series: one density regime, so t̂ across
those rows varies with `n` alone. The `rootlo_*`/`roothi_*` pairs give a
density sweep at fixed `n`, with the matching `logn_*` case as the third point.

The last three sit at `d̄ = 10`, below `ln(10⁶) = 13.8`, and so are the only
cases generated with `gen --connected`. They keep that density because the
weight-distribution results are measured against `1m` as the control.

### 3. Run benchmark

```bash
# Full run (all sizes, all algorithms including par_T2 and par_T4)
python3 run_bench.py --build --gen --target-sec 60 --output-dir results/

# Specific sizes only
python3 run_bench.py --sizes logn_100k,logn_1m

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

---

## How timing works

Each binary times its own MST computation with `std::chrono::steady_clock` —
after parsing the input, before printing the result — and reports it on stderr
as `{"type":"timing","ms":...}`. `run_bench.py` reads that number. **Reported
times are MST construction only**; reading the graph file and writing the MST
out are not included.

The shared support lives in `src/bench_harness.h` and gives every timing binary
two flags:

| Flag | Effect |
| --- | --- |
| `--quiet` | Print `Total weight = ...` but skip the per-edge MST dump. |
| `--repeat K` | Run the MST computation K times on one parse, reporting each timing separately. The first is a warm-up and is discarded by the driver. |

`run_bench.py` passes both automatically. `--repeat` exists because parsing the
1m graph costs ~17.5 s against an algorithm time of ~0.7 s, so paying the parse
per trial limited that case to 3 trials; amortising it over in-process repeats
lifted the 100k case from 4 trials to 60.

This replaced an earlier harness that timed whole processes, which meant ~92–99%
of every measurement was text I/O. See `cluster/FINDINGS-io.md` — the published
results in `analysis/` predate the fix and need regenerating.

Note the trial count is `floor((target_sec/k − parse) / algo)`, so
`--target-sec` has to exceed the parse cost to buy any trials at all. Use
`--target-sec 300` or more for the 1m sizes.

---

## Output Files

All written to `--output-dir` (default `./results/`):

| File               | Contents                                                     |
| ------------------ | ------------------------------------------------------------ |
| `bench_timing.csv` | Per-(size, algorithm): mean\_ms (MST construction only), min, max, σ, parse\_ms, wall\_total\_ms, t̂, vs\_boruvka ratio, vs\_bms ratio |
| `bench_xi.csv`     | Per-(size, algorithm, repeat): ξ\_μ, ξ\_theory, steps/rounds, per-step ξ sequence |
| `bench_report.md`  | Human-readable Markdown report with ξ table, timing table, and convergence analysis |

## Project Structure

```
project/
|-- src/
|   |-- bench_harness.h          # Shared timing support (--quiet, --repeat)
|   |-- BMSBoruvka.cpp           # BMSBoruvka
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
|-- cluster/
|   |-- run_bench.sbatch         # Slurm job script
|   |-- README.md                # Running the benchmarks on a cluster
|   |-- FINDINGS-io.md           # Why the published timings need regenerating
|
|-- README.md
```

## Paper

The `paper` directory now holds the files for building the paper, as well as the original `.docx` version.