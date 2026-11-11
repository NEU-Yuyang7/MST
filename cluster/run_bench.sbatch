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
# Anything after the script name is passed straight through to run_bench.py,
# so a one-off run needs no edit to this file:
#     sbatch cluster/run_bench.sbatch --sizes 10k,100k,500k,1m --n-seeds 3
#
# Because argparse keeps the last occurrence of an option, a passed-through
# flag overrides the default set below — e.g. `--target-sec 600` wins over the
# `--target-sec 300` in section 4. Slurm's own options still belong before the
# script name: `sbatch --time=01:00:00 cluster/run_bench.sbatch --xi-only`.
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
# No modules needed on this cluster: the system compiler is GCC 11.5.0, which
# has C++17, <bits/stdc++.h> (GCC-only) and OpenMP. If you move this script to
# a cluster whose default compiler is older than GCC 9, or is Clang, you will
# need `module load` lines here — `module spider gcc` finds the candidates.
echo "--- Toolchain ---"
g++ --version | head -1
python3 --version

# Fail early and loudly rather than part-way through a 4-hour job.
# TMPDIR is not guaranteed to be set, and `set -u` above would abort on it.
PROBE_DIR="${TMPDIR:-/tmp}"
printf '#include <bits/stdc++.h>\n#include <omp.h>\nint main(){std::vector<int> v{1};return omp_get_max_threads()>0?0:1;}\n' > "$PROBE_DIR/probe.cpp"
if ! g++ -O2 -std=c++17 -fopenmp "$PROBE_DIR/probe.cpp" -o "$PROBE_DIR/probe"; then
    echo "ERROR: compiler cannot build C++17 + bits/stdc++.h + OpenMP" >&2
    exit 1
fi
echo "Toolchain OK"

# --- 2. Where to work ------------------------------------------------------
# Generated graphs total roughly 600 MB. If your home directory has a small
# quota, point SCRATCH at your scratch filesystem instead and the script will
# run there.
# Slurm copies the batch script to a spool directory before running it, so
# ${BASH_SOURCE[0]} points at /var/spool/.../slurm_script rather than this
# repository. SLURM_SUBMIT_DIR is the directory sbatch was run from, which is
# what we want; the BASH_SOURCE form is the fallback for running this script
# directly outside Slurm.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
    REPO_ROOT="$SLURM_SUBMIT_DIR"
else
    REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi

if [[ ! -d "$REPO_ROOT/src" ]]; then
    echo "ERROR: no src/ under '$REPO_ROOT'. Submit with 'sbatch cluster/run_bench.sbatch'" >&2
    echo "       from the repository root, so SLURM_SUBMIT_DIR points at it." >&2
    exit 1
fi

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
#
# "$@" forwards whatever followed the script name on the sbatch line. It comes
# last so those flags override the defaults above rather than the other way
# round. Echoed first so the log records what was actually run — the defaults
# below are no longer the whole story once this is used.
if [[ $# -gt 0 ]]; then
    echo "--- Extra arguments: $* ---"
fi

python3 run_bench.py \
    --build \
    --gen \
    --target-sec 300 \
    --par-threads 2,4 \
    --xi-repeats 3 \
    --output-dir "results/job-${SLURM_JOB_ID}" \
    "$@"

echo "Finished: $(date)"
echo "Results in: $REPO_ROOT/src/results/job-${SLURM_JOB_ID}"
