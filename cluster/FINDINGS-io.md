# The benchmark was measuring text I/O, not MST construction

Found and fixed 2026-11-08. **Every wall-clock number in the Benchmark section
of the paper was produced by the old harness and needs regenerating.** The
reported speedup drops from ~2.5× to ~1.3–1.6× — but it now grows with `n`
instead of staying flat, which is the better result. See "What changes" below.

## The problem

`run_bench.py` used to time whole processes:

```python
t0 = time.perf_counter()
subprocess.run(cmd, stdin=fh, stdout=subprocess.DEVNULL, ...)
return (time.perf_counter() - t0) * 1000.0
```

So each measurement was process spawn + parse the graph from a text file +
compute the MST + format and write every MST edge to stdout. Only the third of
those is the algorithm. The paper's claim that "Program output was redirected
to /dev/null to exclude I/O from timings" doesn't hold: redirecting to
`/dev/null` discards the bytes only after `cout` has already formatted them,
and it does nothing at all about parsing a 113 MB input file.

To quantify it I built a `null_mst` binary that reads the input exactly as the
real programs do and prints `n-1` lines exactly as they do, but performs no MST
computation whatsoever. On the 1m sparse case (n=1,000,000, m=5,000,000):

| Binary | Wall clock | Minus I/O floor |
|---|---:|---:|
| `null_mst` (no algorithm at all) | 18,070 ms | — |
| `boruvka` | 19,560 ms | ~1,490 ms |
| `bms` | 19,130 ms | ~1,060 ms |
| `kruskal` | 18,370 ms | ~300 ms |
| `filter_kruskal` | 18,218 ms | ~150 ms |

92–99% of every published timing was text parsing and printing. (These were
measured on the pre-3bebedd binaries, which is why `bms` here is slower than in
the results further down.)

There was a second, compounding problem. `boruvka.cpp` and
`parallel_boruvka.cpp` were the only two sources *not* calling
`ios::sync_with_stdio(false)` — every binary they were compared against had it.
On libstdc++ that is routinely worth 3–5× on input-heavy programs. So the
baseline was paying full stream-synchronisation cost on a workload that was
almost entirely I/O, while the challengers weren't.

That is exactly the shape of the old results in `analysis/bench_timing.csv`:
Kruskal came out 3.10× faster than Borůvka at n=10k and 3.21× at n=100k.
Kruskal does a full comparison sort of all `m` edges — strictly more work than
Borůvka's `O(m log n)` — so a near-constant 3.2× advantage across an order of
magnitude in `n`, pointing against the asymptotics, was never an algorithmic
effect.

## The fix

**1. `src/bench_harness.h` (new).** A small shared header giving every binary
fast I/O, a `--quiet` flag, in-process timing, and `--repeat K`.

**2. Each binary times itself.** `std::chrono::steady_clock` wraps the MST
computation only — after parsing, before printing — and the result goes to
stderr as `{"type":"timing","ms":...}`. `run_bench.py` reads that number
instead of wrapping the process with a stopwatch. Parsing, printing and process
spawn are now outside the measurement by construction.

**3. `--quiet` suppresses the per-edge dump** during timing runs. The
`Total weight = ...` line is still printed, so the correctness check is
unaffected.

**4. `--repeat K` runs the MST computation K times on one parse**, reporting
each separately, with the first discarded as a warm-up. Parsing the 1m graph
costs ~17.5 s against an algorithm time of ~0.7 s, so paying it per trial
capped that case at the 3-trial minimum. Amortising it lifted the 100k case
from 4 trials to 60, and σ from ~2.6% to ~1%.

Entry points that mutate their input (`kruskal`, `filter_kruskal`,
`BMSBoruvka`) get a fresh copy per repeat, made outside the timed region.
`boruvka` and `parallel_boruvka` take `const vector<Edge>&` and need no copy.

**5. `boruvka.cpp` and `parallel_boruvka.cpp` now get fast I/O** like everything
else, via `bench::init()`. This no longer affects the comparison — I/O is
outside the timed region now — but it makes the wall-clock figures honest too.

Verified: all five binaries still produce `Total weight = 1209420330843` on the
seeded 10k graph, with and without `--quiet`, and identically across repeats.

## What changes

Measurements below are on the **current** `src/BMSBoruvka.cpp`, i.e. after
commit 3bebedd ("Fix BMS-Boruvka implementation"). That rewrite is a large
improvement on its own — roughly 2.2× faster than the previous implementation
at both 10k and 100k — so the numbers here reflect both changes together.

Clean timing, sparse series (macOS, Apple Clang; ratios are what matter, not
absolutes):

| n | boruvka | kruskal | filter_kruskal | bms | bms vs boruvka |
|---|---:|---:|---:|---:|---:|
| 10,000 | 2.29 ms | 2.58 ms | 1.46 ms | 1.76 ms | **1.30×** |
| 100,000 | 30.40 ms | 33.80 ms | 18.24 ms | 21.92 ms | **1.39×** |
| 1,000,000 | 641.55 ms | 390.71 ms | 243.94 ms | 399.38 ms | **1.61×** |

Two things are worth noting.

**BMSBoruvka does beat Borůvka, and the margin grows with n** — 1.30× → 1.39×
→ 1.61×. That is the signature the theory predicts: an advantage that widens
as `n` increases. It is a sounder result than the original 2.5×, which was
constant across sizes and came from a measurement dominated by text I/O.

**Kruskal's phantom 3.2× is gone.** It is now slower than Borůvka at 10k and
100k and faster at 1m — a ratio that varies with `n`, as it should, rather than
the suspiciously flat 3.10×/3.21× the old harness produced.

For reference, the old published figures in `analysis/bench_timing.csv` were
boruvka 38.88 / kruskal 12.53 / bms 15.52 ms at 10k and 419.92 / 130.64 /
163.95 at 100k — measured on a different machine, so compare the ratios.

## What this means for the paper

The direction of Section 6's claim survives; the magnitude does not, and the
reasoning behind it has to be restated.

- **The speedup figures need replacing.** 2.51–2.56× becomes roughly 1.3–1.6×,
  rising with `n`. The prose should make the trend the point, because the trend
  is the part that follows from the complexity analysis. A flat speedup across
  sizes — which is what the old table showed — is evidence *against* an
  asymptotic claim, not for it.

- **Report the crossover.** The margin is still widening at n=10⁶ and
  `TEST_CASES` stops there. Adding 2m/5m/10m would show how far it goes, and a
  measured growth curve is a far stronger result than a single ratio.

- **t̂ is not converging.** The convergence analysis wants t̂ to flatten as `n`
  grows. On clean data it falls: bms 31890 → 29699 → 18407, boruvka 289593 →
  273220 → 155339. With I/O removed this is measuring something real, and at
  10⁶ it is most likely cache and memory-bandwidth effects rather than the
  comparison count. Section 6.4's conclusion needs re-deriving.

## One discrepancy to resolve

`src/BMSBoruvka.cpp` now carries its own ξ instrumentation behind `#ifdef
BMS_XI`, described in the source as "same format as the old
bms_instrumented.cpp". The two disagree on the same input:

| source | steps_actual | ξ_μ |
|---|---:|---:|
| `bms_instrumented.cpp` (what run_bench.py uses) | 2 | 40.82 |
| `BMSBoruvka.cpp -DBMS_XI` | 3 | 21.54 |

I confirmed this is pre-existing, not something the merge introduced: the
upstream `BMSBoruvka.cpp` compiled with `-DBMS_XI`, with none of my changes,
gives the same 3 / 21.54.

This matters because `run_bench.py` builds `bms_xi` from `bms_instrumented.cpp`
but times `bms` from `BMSBoruvka.cpp`. So the ξ table and the timing table
would describe **two different implementations**. Whichever is right, they
should come from the same binary before the numbers go in the paper — easiest
by building `bms_xi` from `BMSBoruvka.cpp -DBMS_XI` and retiring
`bms_instrumented.cpp`. I left it alone because deciding which trace is correct
is a question about the algorithm, not the harness.

Note also that ξ (shrinkage factor) is unaffected by any of this — it is a
count of components per step, not a timing, so Table 4 and the ξ analysis stand
as published.

## Still outstanding

`run_bench.py --n-seeds` is parsed and documented but never used — `args.n_seeds`
appears nowhere after `parse_args()`. Every case uses seed 42 only, so the
reported σ is variance across repeated runs on one fixed graph, not across graph
instances. If the paper wants to claim results hold across random instances,
this needs implementing. I left it alone because it changes what gets measured,
which is your call rather than a bug fix.

One caveat on `--repeat`: in-process repeats run with warm caches and a warm
allocator. Discarding the first repeat is the conventional handling and the
remaining spread is small (σ well under 1% at 10k and 100k), but these are
warm-cache numbers. If you want cold-cache figures, run with `--repeat 1` and a
large `--target-sec`.
