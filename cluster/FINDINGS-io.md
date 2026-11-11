# Benchmark findings

Two things are documented here: a measurement defect that invalidated the
published timings, and what the numbers look like once it is fixed.

**Every wall-clock number in the Benchmark section of the paper was produced by
the old harness and needs regenerating.** The reported speedup drops from ~2.5×
to 1.27–1.78×, but a much stronger result emerges in its place — see
"What the data actually says".

Measurements below are from cluster job `11742193` on Explorer node `d0064`,
GCC 11.5.0, 8 cores, exclusive allocation, 22m33s, all eight sizes passing the
correctness check. Raw output is in `src/results/job-11742193/` and
`logs/bench-11742193.out`.

**That job predates `8783258`**, which fixed the ξ source, the t̂ formula and
`--n-seeds`. The timings in Part 2 are unaffected — that commit changed
instrumentation and a reporting formula, not any algorithm — but the ξ numbers
and the absolute t̂ values are from the old code. Part 3 says what changed and
what a re-run would buy.

---

## Part 1 — The measurement defect

### What was wrong

`run_bench.py` used to time whole processes:

```python
t0 = time.perf_counter()
subprocess.run(cmd, stdin=fh, stdout=subprocess.DEVNULL, ...)
return (time.perf_counter() - t0) * 1000.0
```

So each measurement was process spawn + parse the graph from a text file +
compute the MST + format and write every MST edge to stdout. Only the third is
the algorithm. The paper's claim that "Program output was redirected to
/dev/null to exclude I/O from timings" doesn't hold: redirecting to `/dev/null`
discards bytes only after `cout` has formatted them, and does nothing about
parsing a 113 MB input file.

A `null_mst` binary that reads the input exactly as the real programs do and
prints `n-1` lines exactly as they do, but performs no MST computation at all,
took **18.07 s** on the 1m case where `boruvka` took 19.56 s. **92–99% of every
published timing was text handling.**

A second, compounding problem: `boruvka.cpp` and `parallel_boruvka.cpp` were
the only sources without `ios::sync_with_stdio(false)`. On libstdc++ that is
worth several times on input-heavy programs. So the baseline paid full stream
synchronisation cost on an I/O-bound workload while every algorithm it was
compared against did not — which is why Kruskal appeared 3.10× faster at n=10k
and 3.21× at n=100k, a near-constant advantage across an order of magnitude,
pointing against the asymptotics. With clean timing Kruskal is **1.09×** at 10k
and 1.44× at 100k, varying with `n` as it should.

### What was changed

- **New `src/bench_harness.h`** — fast I/O, `--quiet`, in-process timing, `--repeat`.
- **Each binary times itself** with `steady_clock`, after parsing and before
  printing, reporting on stderr as JSON. `run_bench.py` reads that instead of
  wrapping the process.
- **`--quiet`** suppresses the per-edge dump; `Total weight = ...` still prints,
  so the correctness check is unaffected.
- **`--repeat K`** runs the computation K times on one parse, first discarded as
  warm-up. Entry points that mutate their input get a fresh copy per repeat,
  outside the timed region.
- **Fast I/O for `boruvka` and `parallel_boruvka`**, like everything else.

### A bug this surfaced

`1m_ties` had evidently never been run. `filter_kruskal` picks the median
weight as pivot and splits `<= pivot` into light, `> pivot` into heavy; when
more than half the remaining edges share the median weight, every edge lands in
light, heavy comes back empty, and it recurses on an identically-sized vector.
Unbounded recursion, stack overflow, SIGSEGV. The all-equal-weights graph hits
it on the first call.

Pre-existing — `f676330`'s `filter_kruskal` segfaults identically. Fixed by
falling back to the base case when the partition cannot make progress.

---

## Part 2 — What the data actually says

### BMSBoruvka beats Borůvka at every size

| n | boruvka | bms | ratio |
|---|---:|---:|---:|
| 10,000 | 3.63 ms | 2.82 ms | 1.29× |
| 100,000 | 58.58 ms | 36.82 ms | 1.59× |
| 500,000 | 449.76 ms | 335.97 ms | 1.34× |
| 1,000,000 | 834.25 ms | 528.40 ms | 1.58× |

The ratio is not monotone, so it should not be presented as a clean trend. But
scaling 500k → 1m, `bms` grows by 1.57× against Borůvka's 1.85×, so the
asymptotic advantage is visible in the scaling even where the ratio wobbles.

### The result the paper should be built around

`bms` is **not** the fastest algorithm overall — Filter-Kruskal is, on most
graphs. But look at *where* `bms` beats it:

| graph | weight distribution | bms vs filter_kruskal |
|---|---|---:|
| 10k | distinct random | 0.75× |
| 100k | distinct random | 0.67× |
| 500k | distinct random | 0.47× |
| 1m | distinct random | 0.75× |
| dense_50k | distinct random | 0.38× |
| dense_100k | distinct random | 0.50× |
| **1m_small** | **ints in [1,100]** | **1.05×** |
| **1m_ties** | **all weights equal** | **2.11×** |

`bms` wins on exactly the two low-entropy weight distributions and loses on
every distinct-random one. On `1m_ties` it is the fastest algorithm in the
suite by a wide margin:

| algo | 1m_ties (ms) |
|---|---:|
| **bms** | **587.13** |
| par_T4 | 653.98 |
| boruvka | 1046.57 |
| kruskal | 1066.99 |
| filter_kruskal | 1240.85 |

1.78× over Borůvka and 2.11× over Filter-Kruskal.

**This is the thesis, demonstrated.** The paper argues that comparison sorting
is the bottleneck and that selection-based activation avoids it. These numbers
say precisely that: the advantage appears exactly where sorting degenerates and
disappears where sorting is easy. That is a sharper, more falsifiable claim
than "faster than Borůvka", and it is supported by the data rather than
contradicted by it.

Section 6 should lead with the weight-distribution result, not a single
headline speedup.

### t̂ does not converge

The convergence analysis wants t̂ to flatten as `n` grows. It does not:

| n | t̂ (bms) | t̂ (boruvka) |
|---|---:|---:|
| 10,000 | 19908 | 182998 |
| 100,000 | 17681 | 141774 |
| 500,000 | 10571 | 105230 |
| 1,000,000 | 13912 | 119458 |

Both fall to 500k and rebound at 1m. With I/O removed this is measuring
something real — most plausibly cache and memory-bandwidth effects rather than
comparison counts. **Section 6.4's conclusion needs re-deriving.**

The bms column was computed with the old θ₀ = `n·(log₂ n)^(2/3)`. Since
`8783258` it is `m·(log₂ n)^(2/3)`, so on this series (m = 5n) the figures
rescale by 5. Job `11805997` re-measured them under the new formula:

| n | t̂ (bms) | t̂ (boruvka) |
|---|---:|---:|
| 10,000 | 100317 | 192022 |
| 100,000 | 87227 | 142759 |
| 500,000 | 67401 | 107199 |
| 1,000,000 | 69535 | 119677 |

Same shape: both fall to 500k and rebound at 1m. A constant factor cannot
create or remove convergence, and it didn't. **Section 6.4's conclusion still
needs re-deriving.** Quote these figures rather than the ones above.

### Parallel Borůvka underperforms

`par_T2` is *slower* than serial Borůvka nearly everywhere — 0.41× at 10k,
0.78× at 100k, 0.95× at 500k, 0.90× at 1m. `par_T4` reaches only 1.18–1.39× on
four threads, and beats serial only from 100k upward. Any claim of parallel
speedup needs restating.

---

## Part 3 — Problems found, and where they now stand

These four were open when job `11742193` ran. `8783258` addresses the first
three. That commit was authored by Claude in Yuyang's session and committed
under his account, so the diagnoses below are its reasoning, not a human
review of it — and none of the fixes has been exercised on the cluster yet.

### 1. ξ instrumentation was broken on dense graphs — fixed, unverified

Both dense cases reported `ξ_μ = 1.00` with an empty step count:

| size | algo | ξ_μ | ξ_theory | steps |
|---|---|---:|---:|---:|
| dense_50k | bms | 1.00 | 128.0 | *(blank)* |
| dense_100k | bms | 1.00 | 128.0 | *(blank)* |

`xi_per_step` came back empty, so the geometric mean fell back to 1.0.

Cause and fix in `8783258`: `bms_instrumented.cpp` jumped to `done` as soon as
the MST was complete and so never recorded the final super-step — under-counting
on sparse graphs and recording nothing at all on dense ones. That file is
deleted; `bms_xi` is now built from `BMSBoruvka.cpp -DBMS_XI`. A step count of
0 also no longer prints as blank, so a genuine zero is distinguishable from a
missing value.

**Still to confirm on a real run.** The dense path is the one that produced
nothing before, and it is the one least like what the fix was reasoned about.
Nobody has yet seen a non-empty dense ξ trace; the claim that there now is one
rests on the argument above and nothing else.

### 2. Two ξ implementations disagreed — fixed

`BMSBoruvka.cpp -DBMS_XI` and `bms_instrumented.cpp` gave different answers on
the same input:

| source | steps_actual | ξ_μ |
|---|---:|---:|
| `bms_instrumented.cpp` (what `run_bench.py` used) | 2 | 40.82 |
| `BMSBoruvka.cpp -DBMS_XI` | 3 | 21.54 |

`run_bench.py` built `bms_xi` from one and timed `bms` from the other, so the ξ
table and the timing table described two different implementations. Both now
come from `BMSBoruvka.cpp`, and the 3 / 21.54 trace is the one kept — consistent
with problem 1, where the discarded implementation was the one dropping a step.
The ξ rows in `src/results/job-11742193/bench_xi.csv` are therefore from the
rejected source and should not be quoted.

### 3. t̂ was not comparable across densities — fixed

θ₀ for `bms` was `n·(log₂ n)^(2/3)`, with no `m` term, so at m = 50n the formula
ignored almost all the work and the dense rows read 1940 and 1743 against
~10000–20000 on sparse — an artifact of the metric, not a performance collapse.
It is now `m·(log₂ n)^(2/3)`, matching the O(m log^{2/3} n) bound in the
abstract, so t̂ is comparable both across densities and against Borůvka. The
convergence table is restricted to an explicit `SPARSE_SERIES` list rather than
matching on tag substrings.

### 4. `--n-seeds` did nothing — implemented

Every case used seed 42 only, so the reported σ was variance across repeated
runs on one fixed graph. `--n-seeds K` now generates K instances per case
(seeds `seed` … `seed+K-1`), verifies and times every algorithm on each, and
reports `σ_inst`, the spread of the per-instance means, alongside the
within-run σ.

This is the one that most affects Part 2. The bms-vs-Borůvka ratio there is
non-monotone — 1.29×, 1.59×, 1.34×, 1.58× — and with a single graph per size
there is no way to tell a real effect from one unlucky instance. Answered by
job `11805997`; see Part 4.

### 5. `boruvka_xi.cpp` had the same bug as `bms_instrumented.cpp` — fixed here

Found while checking Part 4's ξ arithmetic. `if(mst_cnt==n-1)goto done;` fired
inside the union loop, before the round's `C_after` was read, ξ pushed, or
`round` incremented. So Borůvka's **final round was never recorded** — the same
defect, in the same place, as the one `8783258` deleted `bms_instrumented.cpp`
for. It survived because that commit replaced the bms instrumentation rather
than auditing the pattern.

Verified on a 1000-node graph, before and after:

| | rounds recorded | ξ_μ | implied final C |
|---|---:|---:|---:|
| before | 3 | 4.8075 | 9 |
| after | 4 | 5.6234 | 1 |

The discarded round was `C_before=9, C_after=1, xi=9.0000` — the largest single
shrinkage of the run. Total MST weight is unchanged, so only the trace was
affected.

The correction does not have a consistent direction: it adds one round, so ξ_μ
moves up or down according to whether that round's shrinkage was above or below
the mean of the others. On the 1000-node graph it rose, 4.81 → 5.62; on job
`11805997`'s 10k case it falls, 5.10 → 4.64. **Every Borůvka ξ_μ and round
count measured before this fix is wrong**, in both directions. The ξ ≥ 2 claim
survives comfortably either way.

---

## Part 4 — Job 11805997: three graph instances, sparse series

Explorer node `c3004`, 20m06s, `--sizes 10k,100k,500k,1m --n-seeds 3`, seeds
42–44. Raw output in `src/results/job-11805997/`. Dense cases were not run, so
Part 3 item 1 is unchanged by this job.

### Graph instance is not the explanation for the ratio spread

σ_inst — the spread of the per-instance means — is small against the effect
being measured:

| n | bms vs boruvka | σ_inst (bms) | σ_inst (boruvka) |
|---|---:|---:|---:|
| 10,000 | 1.24× | 1.8% | 9.2% |
| 100,000 | 1.56× | 1.5% | 1.8% |
| 500,000 | 1.68× | 4.8% | 1.7% |
| 1,000,000 | 1.58× | 3.4% | 2.0% |

The 10k ratio is genuinely below the rest. 500k and 1m differ by roughly one
σ_inst and should not be distinguished. Treat σ_inst as indicative only: it is
a standard deviation of three numbers, so its own uncertainty is large, and at
several points it is no bigger than the within-run σ propagated into the
instance means.

### bms at 500k is unstable, and that is what the old outlier was

Job `11742193` reported bms at 500k as 335.97 ms, giving a 1.34× ratio that
broke the trend. This job gives 263.47 ms and 1.68×. The cause is not the
graph, the node, or the code:

- `8783258`'s only change to `BMSBoruvka.cpp` is a comment, so the timed binary
  is behaviourally identical between the two jobs.
- Every other bms figure reproduces within 1.4% — 2.82→2.80, 36.82→37.32,
  528.40→528.60 — which rules out a machine difference, since that would move
  all of them.
- At σ_inst = 12.69 ms the old reading sits about 5.7σ out.

What is left is bms's own variance at that size: σ = 34.67 ms, 13% of its mean
and by far the largest in the table. Something there is bimodal. **Until that
is understood, no single-trial bms measurement at 500k should be trusted**, and
the mechanism is worth finding — it may well affect other sizes less visibly.

### ξ_μ carries no information beyond the step count

This is not a bug and no re-run will change it.

| n | ξ_μ (bms) | n^(1/3) |
|---|---:|---:|
| 10,000 | 21.54 | 21.5443 |
| 100,000 | 46.42 | 46.4159 |
| 500,000 | 79.37 | 79.3701 |
| 1,000,000 | 100.00 | 100.0000 |

Four for four, to every digit reported. It is forced by the definition:
ξ_i = C_i/C_{i+1} telescopes, so ∏ξ_i = C_0/C_final = n/1, and the geometric
mean over k steps is exactly n^(1/k). Any three-step run on n = 10⁴ gives
21.54, whether the shrinkage was balanced or wildly skewed.

Part 3 item 5 is the confirmation: completing Borůvka's trace made its ξ_μ
land on n^(1/rounds) exactly too (5.6234 against 1000^(1/4) = 5.6234). Before
the fix, neither algorithm telescoped, which is the only reason the identity
was not obvious — an incomplete trace stops short of C_final = 1 and so returns
something other than n^(1/k).

Consequences:

- **Comparing measured ξ_μ against ξ_theory = 2^t is a comparison of step
  counts**, rewritten. At 10k, 2^t = 64 implies about 2.2 super-steps; 3 were
  observed. That is the actual content, and it is a real result — it is simply
  not a measurement of shrinkage.
- **ξ_μ cannot support a claim about the shrinkage *pattern*.** The per-step
  `xi_per_step` values hold that, and the geometric mean discards it. If
  Section 6 argues that observed shrinkage matches theory, it needs the
  per-step distribution, not ξ_μ.
- A non-empty dense ξ trace will likewise report n^(1/steps). It will confirm
  that the instrumentation runs, not that the shrinkage is right.

---

## Notes on reading the results

**`overhead_ms` is not parse cost.** It is whole-process wall clock minus the
sum of the timed regions, so it includes spawning, loading the binary over the
networked filesystem, parsing, printing, and — for the algorithms that mutate
their input — the per-repeat copy of the edge vector. That is why it differs
between algorithms on the same input. It is excluded from every comparison.

**Repeats run warm.** In-process repeats have warm caches and a warm allocator.
Discarding the first is the conventional handling and the remaining spread is
small (σ typically well under 1%), but these are warm-cache numbers. For
cold-cache figures use `--repeat 1` with a large `--target-sec`.

**Borůvka is the noisiest.** σ of 39.11 ms at 1m (4.7%) and 14.28 ms on
`1m_small`, against well under 1% for most others. Worth understanding before
quoting its numbers to three significant figures.
