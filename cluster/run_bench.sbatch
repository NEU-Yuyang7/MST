#!/bin/bash
#SBATCH --job-name=mst-bench
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --exclusive
#SBATCH --output=logs/bench-%j.out
#SBATCH --error=logs/bench-%j.err

# ---------------------------------------------------------------------------
# MST benchmark — Slurm batch job
#
# Submit from the repository root:
#     mkdir -p logs
#     sbatch cluster/run_bench.sbatch
#
# Watch it:   squeue --me
# Tail it:    tail -f logs/bench-<jobid>.out
# Cancel it:  scancel <jobid>
#
# Notes on the #SBATCH settings above:
#   --cpus-per-task=8  must be >= the largest --par-threads value (default 4).
#   --exclusive        gives you the whole node. Timing numbers are worthless
#                      if another job is competing for memory bandwidth, so
#                      keep this even though it costs more queue time.
#   --mem=32G          the 1m-node graphs are ~113 MB of text each and get
#                      parsed into memory; 32G is comfortable headroom.
#   --time=04:00:00    generous. Shorten it once you know the real runtime —
#                      shorter jobs are scheduled sooner.
# ---------------------------------------------------------------------------

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) ==="
echo "Started: $(date)"

# --- 1. Toolchain ----------------------------------------------------------
# The sources use <bits/stdc++.h> (a GCC-only header) and OpenMP, so a real
# GCC is required. Adjust the module name to whatever your cluster provides —
# `module avail gcc` will list the options.
module purge
module load gcc/12.2.0
module load python/3.10.13

echo "--- Toolchain ---"
g++ --version | head -1
python3 --version

# --- 2. Where to work ------------------------------------------------------
# Generated graphs total roughly 600 MB. If your home directory has a small
# quota, point SCRATCH at your scratch filesystem instead and the script will
# run there.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/src"

echo "--- Working in $(pwd) ---"
df -h . | tail -1

# --- 3. Pin threads --------------------------------------------------------
# run_bench.py sets OMP_NUM_THREADS per measurement, so do NOT set it here.
# These two stop OpenMP threads migrating between cores mid-run, which
# otherwise adds variance to the par_T* numbers.
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# --- 4. Build, generate, run ----------------------------------------------
# --target-sec 300: the trial count is (target_sec/k - parse) / algo, and
# parsing the 1m graph alone costs ~17.5 s. At the old default of 60 s the
# per-algorithm budget was exhausted before a single trial ran and the 1m case
# fell back to the 3-trial minimum. 300 s gives it ~45 trials; the smaller
# sizes hit the 60-trial cap either way.
python3 run_bench.py \
    --build \
    --gen \
    --target-sec 300 \
    --par-threads 2,4 \
    --xi-repeats 3 \
    --output-dir "results/job-${SLURM_JOB_ID}"

echo "Finished: $(date)"
echo "Results in: $REPO_ROOT/src/results/job-${SLURM_JOB_ID}"
