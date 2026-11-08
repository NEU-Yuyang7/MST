# Running the benchmarks on the cluster

Written for someone who doesn't work in C/C++ day to day. There is nothing
exotic here — the whole thing is eight small C++ programs and one Python
script that compiles them, feeds them graphs, and times them.

## How a C++ project works, in one minute

Unlike Java or Python, C++ has no project manifest, no `pom.xml`, no package
manager, and no build system in this repo. Each `.cpp` file is a complete,
standalone program with its own `main()`. You turn one into an executable with
a single command:

```bash
g++ -O2 -std=c++17 boruvka.cpp -o boruvka
```

`g++` is the compiler, `-O2` turns on optimisation (essential — without it the
timings are meaningless), `-std=c++17` picks the language version, and
`-o boruvka` names the output file. That produces a binary called `boruvka` in
the current directory, which you run as `./boruvka`.

The programs talk over stdin/stdout. They read a graph from standard input and
write the MST to standard output:

```bash
./gen 10000 50000 --seed 42 --connected > graph.txt   # make a graph
./boruvka < graph.txt | head -1                        # Total weight = 1209420330843
```

`src/run_bench.py` automates all of that: it compiles every `.cpp`, generates
the test graphs, runs each binary many times with a stopwatch around it, and
writes CSV + Markdown reports. You never have to invoke `g++` by hand.

## Step by step on the cluster

### 1. Log in and find a compiler

```bash
module avail gcc
```

The sources use `#include <bits/stdc++.h>`, which is a GCC-specific
convenience header, and `parallel_boruvka.cpp` needs OpenMP. Any reasonably
recent GCC has both. Load one:

```bash
module load gcc/12.2.0
```

Then check it worked:

```bash
g++ --version
```

If `gcc/12.2.0` doesn't exist on your cluster, pick whatever `module avail gcc`
listed and edit the `module load` line in `run_bench.sbatch` to match.

### 2. Don't benchmark on the login node

The login node is shared with everyone else on the cluster, so timings taken
there are noise. It is fine for a quick smoke test, though — this takes a few
seconds and confirms the toolchain works:

```bash
cd src
g++ -O2 -std=c++17 boruvka.cpp -o boruvka
g++ -O2 gen.cpp -o gen
./gen 10000 50000 --seed 42 --connected > /tmp/smoke.txt
./boruvka < /tmp/smoke.txt | head -1
```

You should see `Total weight = 1209420330843` — the same value this project
produces on any machine, since the generator is seeded.

### 3. Submit the real run

From the repository root:

```bash
mkdir -p logs && sbatch cluster/run_bench.sbatch
```

That queues the job. Check on it:

```bash
squeue --me
```

Once it starts, follow the output live (replace `<jobid>` with the number
`sbatch` printed):

```bash
tail -f logs/bench-<jobid>.out
```

Results land in `src/results/job-<jobid>/`: `bench_timing.csv`,
`bench_xi.csv`, and a human-readable `bench_report.md`.

### 4. If you'd rather watch it run interactively

Grab a compute node for an hour and run it by hand — useful the first time, so
you see failures immediately instead of in a log file:

```bash
srun --partition=short --cpus-per-task=8 --mem=32G --time=01:00:00 --pty bash
```

then

```bash
module load gcc/12.2.0 python/3.10.13
cd src
python3 run_bench.py --build --gen --sizes 10k,100k --target-sec 20
```

Start with the small sizes like that. Once it completes cleanly, submit the
full batch job.

## What you need to check before trusting the numbers

Three settings in `run_bench.sbatch` matter and are cluster-specific:

- **`--partition=short`** — the partition names are different on every cluster.
  `sinfo -s` lists yours.
- **`--exclusive`** — keep it. Sharing a node with another job makes memory
  bandwidth contention show up as algorithm differences.
- **`--cpus-per-task=8`** — must be at least the largest value you pass to
  `--par-threads`. If you want `par_T8` numbers, raise both.

You may also need an account/allocation flag (`#SBATCH --account=...`) if your
cluster requires one.

## Disk space

`--gen` writes about 600 MB of graph files into `src/data/`:

| File | Approx size |
|---|---|
| `test_10k.txt` | 1 MB |
| `test_100k.txt` | 11 MB |
| `test_500k.txt` | 56 MB |
| `test_1m.txt` | 113 MB |
| `test_dense_50k.txt` | 56 MB |
| `test_dense_100k.txt` | 113 MB |
| `test_1m_ties.txt` | 113 MB |
| `test_1m_small.txt` | 100 MB |

Generation is skipped for any file that already exists, so re-running the job
is cheap. If your home quota is tight, clone the repo onto scratch instead.

## Useful variations

```bash
# Just the sparse series the paper's convergence analysis uses
python3 run_bench.py --sizes 10k,100k,500k,1m --target-sec 60

# Shrinkage factor only, no timing — fast, and the part that is actually
# machine-independent
python3 run_bench.py --xi-only

# More thread counts (remember to raise --cpus-per-task to match)
python3 run_bench.py --par-threads 2,4,8,16
```

## What the timings mean now

Each binary times its own MST computation with `steady_clock`, after parsing
the graph and before printing the result, and reports it on stderr.
`run_bench.py` reads that number. So `mean_ms` is MST construction only —
reading the 113 MB input file is not in it.

This matters because it didn't used to be true. The harness previously timed
whole processes, which on the 1m case meant ~18 s of text I/O wrapped around
~0.7 s of algorithm. `cluster/FINDINGS-io.md` has the details and the
before/after numbers; read it before you use any of this in the paper, because
the headline comparison changes sign.

Two flags come out of that fix, both set automatically by `run_bench.py`:

- `--quiet` — compute the MST and print its total weight, but skip the
  per-edge dump. Useful by hand too.
- `--repeat K` — run the MST computation K times on a single parse, reporting
  each timing separately. This is how the harness gets 60 trials out of a graph
  that takes 1.6 s to read.

### Set `--target-sec` high enough at 1m

The trial count is `floor((target_sec/k − parse) / algo)`, where `k` is the
number of algorithms. Parsing the 1m graph takes ~17.5 s, so with the default
`--target-sec 60` and 6 algorithms the per-algorithm budget of 10 s is gone
before a single trial runs, and you fall back to the 3-trial minimum. For good
statistics at 1m use:

```bash
python3 run_bench.py --sizes 1m --target-sec 300
```

That gives a 50 s per-algorithm budget, ~32 s of which is usable, for roughly
45 trials. The smaller sizes already hit the 60-trial cap at the default.

## Still outstanding

**`--n-seeds` does nothing.** The flag is defined in `run_bench.py` and
documented as averaging over several independent graph instances, but the
parsed value is never used — every run uses seed 42 only. So the reported σ is
run-to-run variance on one fixed graph, not variance across graphs. If the
paper needs the latter, that flag has to be implemented first. I left it alone
because it changes what gets measured rather than fixing something broken.

## Running it on your Mac

You can't, with the stock toolchain, and it isn't worth fixing:

- `g++` on macOS is Apple Clang wearing a disguise, and it has no
  `bits/stdc++.h`, so every source fails to compile.
- Apple Clang also rejects `-fopenmp`, so `parallel_boruvka.cpp` won't build.

`brew install gcc` and then using `g++-14` explicitly would work, but a laptop
is the wrong place to take these measurements anyway. Use the cluster.
