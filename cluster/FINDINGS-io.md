# Benchmark findings

Two things are documented here: a measurement defect that invalidated the
published timings, and what the numbers look like once it is fixed.

**Every wall-clock number in the Benchmark section of the paper was produced by
the old harness and needs regenerating.** The reported speedup drops from ~2.5×
to 1.27–1.78×, but a much stronger result emerges in its place — see
"What the data actually says".

Measurements below are from cluster job `11742193` on Discovery node `d0064`,
GCC 11.5.0, 8 cores, exclusive allocation, 22m33s, all eight sizes passing the
correctness check. Raw output is in `src/results/job-11742193/` and
`logs/bench-11742193.out`.

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

### Parallel Borůvka underperforms

`par_T2` is *slower* than serial Borůvka nearly everywhere — 0.41× at 10k,
0.78× at 100k, 0.95× at 500k, 0.90× at 1m. `par_T4` reaches only 1.18–1.39× on
four threads, and beats serial only from 100k upward. Any claim of parallel
speedup needs restating.

---

## Part 3 — Open problems

### 1. ξ instrumentation is broken on dense graphs

Both dense cases report `ξ_μ = 1.00` with an empty step count:

| size | algo | ξ_μ | ξ_theory | steps |
|---|---|---:|---:|---:|
| dense_50k | bms | 1.00 | 128.0 | *(blank)* |
| dense_100k | bms | 1.00 | 128.0 | *(blank)* |

That is `xi_per_step` coming back empty, so the geometric mean falls back to
1.0. Either dense inputs take a path the instrumentation doesn't cover, or the
trace is never populated there. **The dense ξ rows are currently unusable.**

### 2. Two ξ implementations disagree

`BMSBoruvka.cpp -DBMS_XI` and `bms_instrumented.cpp` disagree on the same input:

| source | steps_actual | ξ_μ |
|---|---:|---:|
| `bms_instrumented.cpp` (what `run_bench.py` uses) | 2 | 40.82 |
| `BMSBoruvka.cpp -DBMS_XI` | 3 | 21.54 |

Pre-existing: upstream `BMSBoruvka.cpp` with none of these changes gives the
same 3 / 21.54. `run_bench.py` builds `bms_xi` from one and times `bms` from
the other, so the ξ table and the timing table describe **two different
implementations**. Deciding which trace is correct is a question about the
algorithm, not the harness.

### 3. t̂ is not comparable across densities for bms

θ₀ for `bms` is `n·(log₂ n)^(2/3)` — no `m` term. At m = 50n the formula ignores
almost all the work, which is why the dense rows read 1940 and 1743 against
~10000–20000 on sparse. Those are an artifact of the metric, not a performance
collapse, and should not be presented as comparable.

### 4. `--n-seeds` does nothing

Parsed and documented, but `args.n_seeds` appears nowhere after `parse_args()`.
Every case uses seed 42 only, so the reported σ is variance across repeated
runs on one fixed graph, not across graph instances. Left alone because it
changes what gets measured.

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
