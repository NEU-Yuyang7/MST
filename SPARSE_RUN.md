# Running the sparse series with `--n-seeds 3`

## Optional: 2-minute smoke test

Not on the login node, because its watchdog kills compiles. Grab a small interactive node first:

```bash
srun --partition=short --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
cd <your MST repo>/src
python3 run_bench.py --build --gen --sizes 10k --n-seeds 3 --target-sec 10 --no-par --xi-repeats 1 --output-dir results/smoke-seeds
```

What to look for:
- `Seeds   : [42, 43, 44]` at the top.
- Three correctness lines, each with a different weight:
  `OK   10k s42`, `OK   10k s43`, `OK   10k s44`.
- `σ_inst=` in each timing line.

Then `exit` to give the node back.

## The real run

From the repository root:

```bash
mkdir -p logs
sbatch cluster/run_bench_sparse.sbatch
squeue --me                          # check it's queued or running
tail -f logs/sparse-<jobid>.out      # follow it live
```

**Expected run time: roughly 30–45 minutes.** This is an estimate, not a measurement. Most of it
is parsing the 500k and 1m graphs (about 17 s per process at 1m), and each graph is parsed by every
algorithm several times. The script asks for 2 hours to be safe; shorten it once you know the real figure.

What the script does differently from `run_bench.sbatch`:

| Option | Value | Why |
|---|---|---|
| `--sizes` | `10k,100k,500k,1m` | sparse series only |
| `--n-seeds` | `3` | seeds 42, 43, 44 |
| `--gen` | on | only creates the missing seed-43/44 graphs; existing files are reused |
| `--xi-repeats` | `1` | ξ traces are deterministic |
| `--max-trials` | `20` | within-run σ is already <1% |
| `--target-sec` | `300` | per graph; enough for 1m to reach 20 trials |
| `--output-dir` | `results/sparse-<jobid>` | kept apart from the old `job-*` results |

## Reading the results

Results land in `src/results/sparse-<jobid>/`. Open `bench_report.md`, table **"t̂ and Timing Results"**.

1. For each size, take the `bms` and `boruvka` rows. Ratio = boruvka Mean / bms Mean
   (the `vs bms` column on the boruvka row gives the inverse).
2. Note `σ_inst` for both rows.
3. **Is the wobble real?** Take two adjacent sizes, e.g. 100k (1.59× before) and 500k (1.34× before).
   - If the means move by much more than the σ_inst values (rule of thumb: the gap is over 2× σ_inst),
     the non-monotone ratio is real and needs explaining in the paper.
   - If the gap is within that range, the wobble is graph-to-graph noise. Report the ratio with error bars
     (mean ± σ_inst) and don't read a trend into it.
4. If a size is borderline, re-run just that size with more seeds, e.g. `--sizes 100k --n-seeds 8`.

Also check:
- **t̂ (convergence table at the bottom)**: bms values are now about 5× the old ones (θ₀ gained the `m` term).
  What matters is still the shape across n, whether it flattens.
- **`bench_xi.csv`**: has a `seed` column now. Note that bms `ξ_μ` will be the same on every seed
  (21.54 at 10k = 10000^(1/3)), because the per-step ξ values multiply out to n. That's the known
  definition problem, not a bug in this run.
