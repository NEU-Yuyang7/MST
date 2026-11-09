# BMSBorůvka — Cluster Build and Benchmark Instructions

This guide provides shell commands to clone, compile, verify, and benchmark the MST implementations on a Linux cluster. Run these commands on a **compute node allocated for the job**, not on a shared login node. Commands assume Bash, Python 3, Git, and `g++` with C++17 and OpenMP support.

## 1. Check the environment

```bash
pwd
git --version
python3 --version
g++ --version
nproc
```

If the cluster uses environment modules and the tools are not available, ask the cluster administrator which compiler/Python modules to load (e.g., `module avail` and `module load ...`). No package installation is required by `run_bench.py` beyond the Python standard library.

## 2. Clone the repository

```bash
git clone https://github.com/NEU-Yuyang7/MST.git
cd MST/src
```

Record the exact version being benchmarked:

```bash
git rev-parse HEAD
```

**Branch note:** these commands use the repository's default branch (`main`). If the benchmark code to be evaluated lives on a different branch, run `git switch <branch-name>` after `cd MST/src` and before the build, then record the commit hash.

## 3. Build and run a small verification benchmark

The driver compiles all C++ programs (including the OpenMP version), generates test graphs, verifies that MST weights agree, and writes benchmark files. A small run is recommended before the full experiments.

```bash
mkdir -p results/smoke
set -o pipefail
python3 run_bench.py \
  --build \
  --gen \
  --sizes 10k \
  --target-sec 10 \
  --output-dir results/smoke \
  2>&1 | tee results/smoke/run.log
```

If the command fails, review `results/smoke/run.log`. Do not add `--skip-verify` for this initial test.

**Optional: compile manually** (normally unnecessary):

```bash
g++ -O2 -std=c++17 boruvka.cpp -o boruvka
g++ -O2 -std=c++17 kruskal.cpp -o kruskal
g++ -O2 -std=c++17 filter_kruskal.cpp -o filter_kruskal
g++ -O2 -std=c++17 BMSBoruvka.cpp -o bms
g++ -O2 -std=c++17 -fopenmp parallel_boruvka.cpp -o parallel_boruvka
g++ -O2 -std=c++17 -DBMS_XI BMSBoruvka.cpp -o bms_xi
g++ -O2 -std=c++17 boruvka_xi.cpp -o boruvka_xi
g++ -O2 gen.cpp -o gen
```

## 4. Run the full benchmark

Run after the smoke test succeeds. The default suite covers sparse, dense, tied-weight, and small-integer-weight graph cases. It includes the parallel Borůvka configurations with 2 and 4 threads.

```bash
mkdir -p results/full
set -o pipefail
python3 run_bench.py \
  --build \
  --gen \
  --target-sec 60 \
  --par-threads 2,4 \
  --output-dir results/full \
  2>&1 | tee results/full/run.log
```

Important: `--target-sec 60` specifies a timing budget **per graph-size configuration**, not a total job runtime. Dataset generation, verification, and ξ measurements add time. The cluster job allocation must allow enough wall time and memory for the full experiment, especially the million-vertex graphs. Request **at least 4 CPUs** for the parallel configurations; use an allocation compatible with the cluster's scheduling policy.

### Optional: run specific graph sizes only

```bash
mkdir -p results/selected
set -o pipefail
python3 run_bench.py \
  --gen \
  --sizes 100k,1m \
  --target-sec 60 \
  --output-dir results/selected \
  2>&1 | tee results/selected/run.log
```

This assumes the binaries were already built. Add `--build` if necessary.

### Optional: single-core benchmark without parallel algorithms

```bash
mkdir -p results/single_core
set -o pipefail
python3 run_bench.py \
  --build \
  --gen \
  --no-par \
  --target-sec 60 \
  --output-dir results/single_core \
  2>&1 | tee results/single_core/run.log
```

### Optional: ξ (component shrinkage) measurements only

```bash
mkdir -p results/xi_only
set -o pipefail
python3 run_bench.py \
  --build \
  --gen \
  --xi-only \
  --output-dir results/xi_only \
  2>&1 | tee results/xi_only/run.log
```

`--xi-only` does not run the multi-trial timing benchmark.

## 5. Save hardware and software information

Run inside the same compute-node allocation as the benchmarks, if possible:

```bash
mkdir -p results/full
{
  date -Is
  hostname
  uname -a
  lscpu
  free -h
  python3 --version
  g++ --version
  git rev-parse HEAD
  env | grep -E '^(OMP_|SLURM_)' || true
} > results/full/system_info.txt 2>&1
```

If `lscpu` or `free` is not available on the cluster, omit that command.

## 6. Check the output and package results

```bash
ls -lh results/full/
head -n 10 results/full/bench_timing.csv
head -n 10 results/full/bench_xi.csv
```

The Python driver produces:

| File | Description |
| --- | --- |
| `results/full/bench_timing.csv` | Per-algorithm runtimes and comparisons |
| `results/full/bench_xi.csv` | Component-shrinkage measurements |
| `results/full/bench_report.md` | Human-readable benchmark report |
| `results/full/run.log` | Captured console output from the driver |
| `results/full/system_info.txt` | Compute-node and build environment information (from §5) |

To archive these files for sharing:

```bash
tar -czf mst_benchmark_results.tar.gz -C results full
ls -lh mst_benchmark_results.tar.gz
```

## 7. Optional: submission to a Slurm-managed cluster

If the cluster uses Slurm, run the benchmark from an allocated compute node or submit a batch script. The partition, account, resource limits, and module names vary by cluster. The following is a **template**, not a guaranteed cluster-specific submission command.

Create `run_mst.sbatch` in the repository's `src/` directory:

```bash
#!/bin/bash
#SBATCH --job-name=mst-bench
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=mst-slurm-%j.out
#SBATCH --error=mst-slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"

# Load compiler/Python modules here if required by the cluster.
mkdir -p results/full

{
  date -Is
  hostname
  uname -a
  lscpu
  free -h
  python3 --version
  g++ --version
  git rev-parse HEAD
  env | grep -E '^(OMP_|SLURM_)' || true
} > results/full/system_info.txt 2>&1

python3 run_bench.py \
  --build --gen --target-sec 60 --par-threads 2,4 \
  --output-dir results/full \
  2>&1 | tee results/full/run.log
```

Submit from `MST/src`:

```bash
sbatch run_mst.sbatch
squeue -u "$USER"
```

Please adjust the `#SBATCH` options according to cluster policy and the actual job requirements. If the benchmark's graph instances do not fit in 32 GB, increase the memory request. For controlled performance comparisons, avoid simultaneous competing workloads and report the CPU/thread allocation.

## Notes on experimental reproducibility

- The default random seed is `42` (`--seed 42`). Record any change.
- `--n-seeds K` generates K graph instances per case (seeds `seed` … `seed+K-1`) and times every algorithm on each. Means pool all instances; the `σ_inst` column is the spread of per-instance means. The `--target-sec` budget applies per instance, so total run time grows roughly K-fold.
- The smoke test and the full benchmark have separate output directories to avoid overwriting output CSV files.
- The benchmark performs an MST-weight cross-check by default; it is a basic consistency check, not a full formal correctness proof.
- Because source code and compilation behavior may change, the commands should be tested on the target cluster before being treated as validated experimental procedure.
