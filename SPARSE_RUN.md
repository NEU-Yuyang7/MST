# Re-running the sparse series with `--n-seeds 3`

The sparse series is Group A in `TEST_CASES`: `logn_10k`, `logn_30k`, `logn_100k`,
`logn_300k`, `logn_1m` (mean degree d̄ = 2 ln n). Job `11805997` already did a
3-seed run of the earlier 10k–1m series; `cluster/FINDINGS-io.md` Part 4 has the analysis.

## Optional: smoke test

Not on the login node, because its watchdog kills compiles. Grab a small interactive node first:

```bash
srun --partition=short --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
cd <your MST repo>/src
python3 run_bench.py --build --gen --sizes logn_10k --n-seeds 3 --target-sec 10 --no-par --xi-repeats 1 --output-dir results/smoke-seeds
```

Check for `Seeds   : [42, 43, 44]`, three `OK   logn_10k s4x` lines with different weights,
and `σ_inst=` in each timing line. Then `exit`.

## The run

From the repository root, using the argument passthrough in `cluster/run_bench.sbatch`:

```bash
mkdir -p logs
sbatch cluster/run_bench.sbatch --sizes logn_10k,logn_30k,logn_100k,logn_300k,logn_1m --n-seeds 3
squeue --me
tail -f logs/bench-<jobid>.out
```

Results land in `src/results/job-<jobid>/`. `--target-sec` applies per graph instance, so the
run takes roughly three times as long as a single-seed run of the same sizes. With `--quiet`,
parsing the largest graph costs about a second; the per-process overhead is mostly that parse
plus, for the algorithms that mutate their input, the per-repeat copy of the edge vector.

## Reading the results

Open `bench_report.md`, table **"t̂ and Timing Results"**.

1. For each size, ratio = boruvka Mean / bms Mean (the `vs bms` column on the boruvka row
   gives the inverse). Note `σ_inst` for both rows.
2. **Is a change in the ratio between two sizes real?** If the means move by more than about
   2× σ_inst, it is not graph-to-graph noise. If less, report mean ± σ_inst and don't read a
   trend into it. σ_inst from three seeds is itself rough, so treat the rule as indicative.
3. A large within-run σ points at the algorithm, not the graph: in job `11805997` the old 500k
   outlier came from bms's own 13% within-run variance, not from the instance.

Also check:
- **t̂ (convergence table)**: what matters is the shape across n, whether it flattens.
- **ξ table**: reports actual steps against the theory count P, and `ξ_full`, the geometric mean
  of the non-final steps, against 2^t. The final step is excluded because it always ends at one
  component; including it makes the mean exactly n^(1/steps), which says nothing about shrinkage.
  `xi_geomean_all_steps` in `bench_xi.csv` is that old value, kept for reference only.
