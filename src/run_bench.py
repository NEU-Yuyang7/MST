#!/usr/bin/env python3
"""
run_bench.py — MST Benchmark: ξ (shrinkage factor) + t̂ (normalised throughput)
=================================================================================

Two core metrics:

  ξ (xi)
    Per-step / per-round component shrinkage factor.
      ξ_i = C_i / C_{i+1}
    boruvka : theory lower bound  ξ ≥ 2     (each round halves components)
    BMS     : theory per super-step  ξ = 2^t
    ξ_μ = geometric mean over all steps — reported in Table 1.

  t̂ (t_hat) — normalised throughput
    t̂ = θ₀(n, m) / T_mean   [theoretical ops per millisecond]

    T_mean is MST CONSTRUCTION TIME ONLY. Each binary times itself with
    steady_clock after parsing the graph and before printing the MST, and
    reports the result on stderr; this script reads that number. Timing whole
    processes instead measured text I/O: on the 1m case ~18 s of the ~19 s
    total, which swamped every algorithmic difference. Most of that was
    *writing* the MST, which --quiet now suppresses; parsing alone is ~1.2 s
    for the 113 MB 1m graph.

    boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]
    bms:               θ₀ = m · (log₂ n)^(2/3)        [O(m log^{2/3} n)]

    t̂ → constant as n grows  ⟹  bound is empirically tight.

    Both θ₀ carry the m term, so t̂ is comparable across densities and
    between bms and Borůvka. (θ₀ for bms used to be n · (log₂ n)^(2/3); with
    no m term, dense graphs read ~10× lower than sparse ones purely as an
    artifact of the formula. On the sparse series, m = 5n, so the change
    only rescales t̂ by 5.) The convergence table uses the sparse series.

Graph instances (--n-seeds K):
    Each case is generated K times with seeds seed, seed+1, …, seed+K-1 and
    every algorithm is timed on every instance. Reported means pool all
    instances; σ_inst is the standard deviation of the per-instance means,
    i.e. variation across random graphs rather than across repeated runs.

Trials formula:
    trials = max(min_trials, min(max_trials, floor(T / (k · budget))))
    where T = target_ms, k = number of algorithms, budget = slowest probe time.
    Each algorithm gets an equal share of the total budget.

Usage:
  python3 run_bench.py [--build] [--gen] [--target-sec N] [--sizes S] [options]

Options:
  --build            Compile all binaries
  --gen              Generate test data
  --target-sec N     Total timing budget per size (seconds, default 30)
  --min-trials N     Minimum trials per algorithm (default 3)
  --max-trials N     Maximum trials per algorithm (default 60)
  --par-threads S    Threads for parallel_boruvka, comma-separated (default 2,4)
  --output-dir D     Results directory (default ./results)
  --sizes S          Only run specified size tags, comma-separated
  --skip-verify      Skip correctness check
  --xi-only          Collect ξ data only, skip multi-trial timing
  --xi-repeats N     Number of ξ collection repeats per test case (default 3)
  --seed N           Random seed for gen (default 42)
  --n-seeds K        Independent graph instances per case (default 1)
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).parent.resolve()
BIN_DIR    = SCRIPT_DIR
DATA_DIR   = SCRIPT_DIR / "data"

SOURCES = {
    "boruvka":        SCRIPT_DIR / "boruvka.cpp",
    "parallel":       SCRIPT_DIR / "parallel_boruvka.cpp",
    "bms":            SCRIPT_DIR / "BMSBoruvka.cpp",
    # ξ trace comes from the same source as the timed bms, compiled with
    # -DBMS_XI. The old bms_instrumented.cpp skipped the final super-step
    # (it jumped to `done` once the MST was complete), so it under-counted
    # steps on sparse graphs and recorded nothing at all on dense ones.
    "bms_xi":         SCRIPT_DIR / "BMSBoruvka.cpp",
    "boruvka_xi":     SCRIPT_DIR / "boruvka_xi.cpp",
    "kruskal":        SCRIPT_DIR / "kruskal.cpp",
    "filter_kruskal": SCRIPT_DIR / "filter_kruskal.cpp",
    "gen":            SCRIPT_DIR / "gen.cpp",
}

# ── Test cases ─────────────────────────────────────────────────────────────────
# (tag, n, m, extra_gen_args)
#
# Density is set by mean degree d̄ = 2m/n, at 2·ln n, √n/2 and 2·√n. The two
# reference points do NOT have equal standing, and it is worth being exact
# about which is which:
#
#   d̄ = ln n   A THEOREM. The Erdős–Rényi sharp threshold: below it a uniform
#              random graph is a.a.s. disconnected. Not negotiable.
#
#              It does double duty here. Fredman–Tarjan is O(m·β(m,n)) with
#              β(m,n) = min{i : log^(i) n ≤ m/n}, so once m/n ≥ log n — i.e.
#              d̄ ≥ 2 log n — β collapses to 1 and MST is already O(m) by a
#              1987 comparison-based algorithm. Group A sits within a constant
#              factor of that point (natural log puts d̄ = 27.6 at n = 10⁶
#              against 39.9 for base 2, so just under it).
#
#   d̄ = √n     A CHOICE, not a convention. There is no standard sparse/dense
#              boundary at √n. What is standard is coarser: sparse means
#              m = O(n) or O(n polylog n), dense means m = Θ(n²) or
#              m = Ω(n^(1+ε)). The real qualitative line behind those is
#              polylogarithmic vs polynomial growth in d̄ — no amount of
#              iterating logs reaches n^ε — and √n is one representative of
#              the polynomial side, chosen because m = Θ(n^1.5) is the
#              geometric mean of n and n² and is still reachable at n ≤ 10⁵.
#              n^(1/3) or n^(2/3) would be defensible too.
#
# Note that "beyond any polylog" is asymptotic and weak at these sizes: √n
# overtakes (log₂ n)² only at n ≈ 10^4.8 and (log₂ n)³ not until n ≈ 10^8.8, so
# across 10⁴–10⁶ it sits between the square and the cube of the log. The case
# for 2√n is that it is decisively past m = 50n (d̄ 632 against 100), not that
# it is qualitatively elsewhere.
#
# Previously the suite used m = 5n ("sparse") and m = 50n ("dense"). Neither
# dense case cleared √n — at n = 10⁵, d̄ = 100 against √n = 316 — so m = 50n is
# a constant factor denser, not asymptotically denser, and cannot show BMS
# degrading toward O(m log n). The m = 5n series sat *below* ln n from n = 10⁵
# up, so those graphs were connected only because gen imposed a Hamiltonian
# path on them.
#
# NOTE: a data file is reused whenever test_<tag>.txt exists, so changing the
# n or m of an existing tag silently reads the old graph. Every case whose
# size changed therefore has a new tag.
#
# Group A — convergence series, d̄ = 2 ln n (m = n ln n). The n-scaling series
# for ξ and t̂. Above the connectivity threshold, so these are true G(n, m).
#
# Group B — density triples at fixed n, spanning ln n → √n → above √n. Group A
# supplies each triple's lowest-density member, so only two new cases per n.
#
# Group C — weight distributions, left at the legacy m = 5n so they stay
# comparable with jobs 11742193 and 11805997. The 2.11× bms result on 1m_ties
# is measured against `1m`, which is why that case keeps its old density.
TEST_CASES = [
    # Group A — d̄ = 2 ln n
    ("logn_10k",    10_000,      92_103, []),
    ("logn_30k",    30_000,     309_268, []),
    ("logn_100k",  100_000,   1_151_292, []),
    ("logn_300k",  300_000,   3_783_461, []),
    ("logn_1m",  1_000_000,  13_815_510, []),
    # Group B — d̄ = √n/2 and 2√n
    ("rootlo_30k",  30_000,   1_299_038, []),
    ("roothi_30k",  30_000,   5_196_152, []),
    ("rootlo_100k", 100_000,  7_905_694, []),
    ("roothi_100k", 100_000, 31_622_776, []),
    # Group C — weight distributions, legacy density (d̄ = 10)
    ("1m",      1_000_000,   5_000_000, []),
    ("1m_ties",  1_000_000,   5_000_000, ["--weights", "uniform"]),
    ("1m_small", 1_000_000,   5_000_000, ["--weights", "small"]),
    # roadNet-CA: pre-converted from SNAP via convert_snap.py
    # n=1,965,206  m=2,766,607  (real-world sparse graph, m ≈ 1.4n)
    # Generate with:  python3 convert_snap.py roadNet-CA.txt --output data/test_roadnet_ca.txt
    # Then add to TEST_CASES by uncommenting the line below:
    # ("roadnet_ca", 1_965_206, 2_766_607, []),   # real-world; use --sizes roadnet_ca
]

# Group A: the n-scaling series used by the convergence table. One density
# regime (d̄ = 2 ln n), so t̂ across these rows varies with n alone.
SPARSE_SERIES = ("logn_10k", "logn_30k", "logn_100k", "logn_300k", "logn_1m")

# Cases generated with gen --connected. gen imposes a random Hamiltonian path
# before filling the rest uniformly, so a forced graph is
# `path ∪ G(n, m-n+1)`: every vertex has degree >= 2 and the degree
# distribution is roughly 2 + Poisson(d̄-2) rather than Poisson(d̄). That is
# only acceptable where it is unavoidable.
#
# Group C sits at d̄ = 10 against ln(10⁶) = 13.8, where a uniform graph is
# a.a.s. disconnected, so the path is doing structural work and must stay.
# Groups A and B are at or above 2 ln n, where P(disconnected) ≈ n·e^(-d̄)
# = 1/n, so they are generated unforced and are true uniform random graphs.
# A disconnected draw is not a failure — every algorithm computes the MSF and
# the weights still agree — it is merely unlikely.
FORCE_CONNECTED = frozenset({"1m", "1m_ties", "1m_small"})

# Graph instance seeds, set from --seed / --n-seeds in main. SEEDS[0] is the
# base seed; its data file keeps the original name test_<tag>.txt.
SEEDS = [42]

COMPILE_FLAGS = ["-O2", "-std=c++17"]

# ── Utilities ─────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] {level:5s}  {msg}", flush=True)


TIMEOUT_RC = 124   # conventional exit status for a timed-out command


def run_cmd(cmd, *, stdin=None, capture_out=False, capture_err=False,
            env=None, timeout=600):
    """
    Run cmd, returning (returncode, stdout, stderr).

    A timeout returns TIMEOUT_RC rather than raising. An uncaught
    TimeoutExpired here used to abort the entire benchmark with a traceback —
    one binary looping on one graph would destroy a multi-hour job after
    everything else had already succeeded. Callers decide what a timeout means.
    """
    fh = open(stdin) if isinstance(stdin, (str, Path)) else stdin
    try:
        r = subprocess.run(
            cmd,
            stdin=fh,
            stdout=subprocess.PIPE if capture_out  else subprocess.DEVNULL,
            stderr=subprocess.PIPE if capture_err  else subprocess.DEVNULL,
            env={**os.environ, **(env or {})},
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return TIMEOUT_RC, "", f"timed out after {timeout}s"
    finally:
        # subprocess.run does not close a handle we opened.
        if fh is not None and fh is not stdin:
            fh.close()
    return (
        r.returncode,
        r.stdout.decode() if capture_out  and r.stdout else "",
        r.stderr.decode() if capture_err  and r.stderr else "",
    )


def time_run(cmd, stdin, env=None, repeat=1):
    """
    Run cmd with stdin redirected. Returns (algo_ms_list, wall_ms).

      algo_ms_list  the binary's own measurements of MST construction only,
                    one per in-process repeat, taken with steady_clock after
                    parsing and before printing and reported on stderr as
                    {"type":"timing","ms":...}. Empty if the binary doesn't
                    report any (e.g. an old build without bench_harness.h).

      wall_ms       whole-process wall clock: spawn + parse + K× compute + print.

    algo_ms is the metric. wall_ms is kept only for reference: under the old
    print-everything regime ~18 s of the ~19 s at 1m was text I/O, which buried
    the differences between algorithms entirely.
    """
    e = {**os.environ, **(env or {})}
    full = list(cmd) + (["--repeat", str(repeat)] if repeat > 1 else [])
    with open(stdin) as fh:
        t0 = time.perf_counter()
        r = subprocess.run(
            full, stdin=fh,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            env=e, check=True,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0

    algo = []
    for line in r.stderr.decode(errors="replace").splitlines():
        try:
            obj = json.loads(line)
            if obj.get("type") == "timing":
                algo.append(float(obj["ms"]))
        except Exception:
            pass
    return algo, wall_ms


def primary_times(rec):
    """
    The series to report on: in-process algorithm times when the binaries
    provide them, otherwise whole-process wall clock.
    """
    algo = [x for x in rec.get("algo", []) if x is not None]
    return algo if algo else rec.get("wall", [])


def get_weight(cmd, stdin, env=None):
    """
    Run cmd and parse 'Total weight = ...' from stdout.

    Returns the weight as a string, or a marker describing what went wrong:
    "<timeout>" if it exceeded the limit, "<crash rc=N>" if it died (a stack
    overflow from unbounded recursion shows up here as rc=-11), or "<no output>"
    if it exited cleanly without printing a weight. These markers compare
    unequal to a real weight, so verify_all reports them as a mismatch and
    names the algorithm.
    """
    rc, out, _ = run_cmd(cmd, stdin=stdin, capture_out=True,
                         env={**os.environ, **(env or {})})
    if rc == TIMEOUT_RC:
        return "<timeout>"
    for line in out.splitlines():
        if line.startswith("Total weight"):
            return line.split("=")[1].strip()
    return f"<crash rc={rc}>" if rc != 0 else "<no output>"


def data_path(tag, seed=None):
    if seed is None or seed == SEEDS[0]:
        return DATA_DIR / f"test_{tag}.txt"
    return DATA_DIR / f"test_{tag}_s{seed}.txt"


def instances(tag):
    """(seed, path) for every graph instance of this case."""
    return [(s, data_path(tag, s)) for s in SEEDS]


def steps_of(r):
    """Steps (bms) or rounds (boruvka). 0 is a real value, not 'missing'."""
    v = r.get("steps_actual")
    if v is None:
        v = r.get("rounds")
    return "" if v is None else v


def is_sparse(tag):
    return tag in SPARSE_SERIES




# ── t̂ calculation ─────────────────────────────────────────────────────────────
def t_hat(mean_ms, n, m, label):
    """
    Normalised throughput = θ₀(n, m) / T_mean   [ops / ms]

    θ₀ definitions (matching the paper):
      boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n) bound]
      bms:               θ₀ = m · (log₂ n)^(2/3)        [O(m log^{2/3} n) bound]

    Note: the exponent 2/3 is applied to the entire log₂(n), i.e.
      (log₂ n)^(2/3)   NOT   log₂(n^(2/3))
    """
    log2n = math.log2(n) if n > 1 else 1.0
    if label == "bms":
        theta = m * (log2n ** (2.0 / 3.0))
    else:
        theta = m * log2n
    return theta / mean_ms if mean_ms > 0 else 0.0


# ── Build ─────────────────────────────────────────────────────────────────────
def build_all(par_threads):
    log("Compiling binaries …")
    jobs = [
        (["g++"] + COMPILE_FLAGS +
         [str(SOURCES["boruvka"]), "-o", str(BIN_DIR / "boruvka")],
         "boruvka"),
        (["g++"] + COMPILE_FLAGS + ["-fopenmp",
         str(SOURCES["parallel"]), "-o", str(BIN_DIR / "parallel_boruvka")],
         "parallel_boruvka"),
        (["g++"] + COMPILE_FLAGS +
         [str(SOURCES["bms"]),    "-o", str(BIN_DIR / "bms")],
         "bms"),
        (["g++"] + COMPILE_FLAGS +
         [str(SOURCES["kruskal"]), "-o", str(BIN_DIR / "kruskal")],
         "kruskal"),
        (["g++"] + COMPILE_FLAGS +
         [str(SOURCES["filter_kruskal"]), "-o", str(BIN_DIR / "filter_kruskal")],
         "filter_kruskal"),
        (["g++"] + COMPILE_FLAGS +
         ["-DBMS_XI", str(SOURCES["bms_xi"]), "-o", str(BIN_DIR / "bms_xi")],
         "bms_xi"),
        (["g++"] + COMPILE_FLAGS +
         [str(SOURCES["boruvka_xi"]), "-o", str(BIN_DIR / "boruvka_xi")],
         "boruvka_xi"),
        (["g++", "-O2",
         str(SOURCES["gen"]),    "-o", str(BIN_DIR / "gen")],
         "gen"),
    ]
    for cmd, name in jobs:
        log(f"  {name}")
        rc, _, err = run_cmd(cmd, capture_err=True)
        if rc != 0:
            log(f"  FAILED: {err}", "ERROR")
            sys.exit(1)
    log("Build complete.")


# ── Data generation ───────────────────────────────────────────────────────────
def generate_data(cases):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gen_bin = str(BIN_DIR / "gen")
    for tag, n, m, extra in cases:
        for seed, p in instances(tag):
            if p.exists():
                log(f"  {p.name} already exists, skipping")
                continue
            cmd = [gen_bin, str(n), str(m), "--seed", str(seed)]
            if tag in FORCE_CONNECTED:
                cmd.append("--connected")
            cmd += extra
            log(f"  Generating {p.name}  (n={n:,} m={m:,} d̄={2*m/n:.1f} "
                f"seed={seed}{' connected' if tag in FORCE_CONNECTED else ''})")
            with open(p, "w") as fout:
                if subprocess.run(cmd, stdout=fout, stderr=subprocess.DEVNULL).returncode:
                    log(f"  gen failed for {tag} seed={seed}", "ERROR")
                    p.unlink(missing_ok=True)
                    sys.exit(1)
    log("Data generation complete.")


# ── Algorithm list ─────────────────────────────────────────────────────────────
def make_timing_bins(par_threads, include_par=True):
    """
    Returns list of (label, cmd, env_overrides).
    par_T2, par_T4, … set OMP_NUM_THREADS accordingly.
    Set include_par=False to omit parallel Borůvka (e.g. on single-core machines).
    """
    # --quiet suppresses the per-edge MST dump. The "Total weight = ..." line
    # is still printed, so the correctness check is unaffected, but we no longer
    # pay to format n-1 lines of text on every single trial.
    bins = [
        ("boruvka",        [str(BIN_DIR / "boruvka"),        "--quiet"], {}),
        ("kruskal",        [str(BIN_DIR / "kruskal"),        "--quiet"], {}),
        ("filter_kruskal", [str(BIN_DIR / "filter_kruskal"), "--quiet"], {}),
        ("bms",            [str(BIN_DIR / "bms"),            "--quiet"], {}),
    ]
    if include_par:
        for t in par_threads:
            bins.append((
                f"par_T{t}",
                [str(BIN_DIR / "parallel_boruvka"), "--quiet"],
                {"OMP_NUM_THREADS": str(t)},
            ))
    return bins


# ── Correctness verification ───────────────────────────────────────────────────
def verify_all(cases, bins):
    log("Correctness check …")
    failures = 0
    every_instance = [(tag, seed, p) for tag, *_ in cases
                      for seed, p in instances(tag)]
    for tag, seed, p in every_instance:
        if not p.exists():
            continue
        name = tag if len(SEEDS) == 1 else f"{tag} s{seed}"
        weights = {label: get_weight(cmd, p, env) for label, cmd, env in bins}
        if len(set(weights.values())) == 1:
            log(f"  OK   {name:15s}  weight = {list(weights.values())[0]}")
        else:
            # Separate "this one broke" from "these disagree" — they need very
            # different investigation, and lumping them into one dict dump made
            # the cause hard to see.
            broken = {l: w for l, w in weights.items() if w and w.startswith("<")}
            agreed = {l: w for l, w in weights.items() if l not in broken}
            log(f"  FAIL {name}", "ERROR")
            for label, why in broken.items():
                log(f"         {label:15s} {why}", "ERROR")
            if len(set(agreed.values())) > 1:
                log(f"         disagreement among the rest: {agreed}", "ERROR")
            elif agreed:
                log(f"         the rest agree on {list(agreed.values())[0]}", "ERROR")
            failures += 1
    if failures:
        log(f"{failures} correctness failure(s). Aborting.", "ERROR")
        sys.exit(1)
    log("All algorithms agree.")


# ── Probe + trials calculation ─────────────────────────────────────────────────
def probe_once(cases, bins):
    """
    Run each (case, algorithm) pair twice; take the second run as the probe time.
    (First run discarded as warm-up.)
    """
    log("Probe runs (warm-up + one timed run each) …")
    probe = {}
    for tag, *_ in cases:
        p = data_path(tag)
        if not p.exists():
            continue
        for label, cmd, env in bins:
            time_run(cmd, p, env)                     # warm-up — discarded
            algo, wall_ms = time_run(cmd, p, env)     # probe
            # Budget on the algorithm time: that is what each extra in-process
            # repeat costs. Parsing is paid once per (case, algorithm), not per
            # trial, so it is accounted for separately in calc_trials.
            algo_ms = algo[-1] if algo else wall_ms
            probe[(tag, label)] = {"algo": algo_ms, "wall": wall_ms}
            log(f"  {tag:15s}  {label:10s}  wall={wall_ms:.0f} ms  algo={algo_ms:.2f} ms")
    return probe


def calc_trials(probe, cases, bins, target_ms, min_t, max_t):
    """
    trials = max(min_t, min(max_t, floor((T/k − parse) / algo)))

    T      = target_ms  (total budget for this size)
    k      = len(bins)  (number of algorithms)
    parse  = fixed per-process overhead (spawn + read the graph + print),
             paid once per (case, algorithm)
    algo   = slowest per-repeat algorithm time for this size

    Because repeats happen inside one process, the graph is parsed once and
    each extra trial costs only `algo`. Before --repeat existed, every trial
    paid the full parse, which capped the 1m case at the 3-trial minimum.
    """
    trials = {}
    k = len(bins)
    for tag, *_ in cases:
        if not data_path(tag).exists():
            continue
        entries = [probe[(tag, label)] for label, _, _ in bins
                   if (tag, label) in probe]
        if not entries:
            continue
        algo_budget  = max(e["algo"] for e in entries)
        # Per-process overhead ≈ whole process minus the single algorithm run
        # it contained (the probe uses repeat=1).
        overhead_budget = max(e["wall"] - e["algo"] for e in entries)

        per_algo_budget = target_ms / k
        usable = per_algo_budget - overhead_budget
        t = max(min_t, min(max_t, int(usable // algo_budget) if algo_budget > 0 else max_t))
        trials[tag] = t
        log(f"  {tag:15s}  overhead≈{overhead_budget:.0f} ms  slowest algo={algo_budget:.2f} ms"
            f"  per-algo budget={per_algo_budget:.0f} ms  → {t} trials")
    return trials


# ── Main timing loop ───────────────────────────────────────────────────────────
def run_timing(cases, bins, trials):
    """
    Time every algorithm on every graph instance of every case.

    Each (case, algorithm) record pools the timed repeats from all instances,
    so its mean is the mean over instances (every instance gets the same
    number of trials). `seed_means` keeps one mean per instance; their spread
    (σ_inst) is the variation across random graphs, which the within-run σ
    cannot show.
    """
    results = {}
    total = sum(1 for tag, *_ in cases if data_path(tag).exists())
    done  = 0
    for tag, n, m, _ in cases:
        if not data_path(tag).exists():
            continue
        done += 1
        t = trials.get(tag, 3)
        log(f"[{done}/{total}] {tag}  n={n:,} m={m:,}  trials={t}"
            f"  instances={len(SEEDS)}")
        for label, cmd, env in bins:
            rec = {"algo": [], "wall": [], "seed_means": [],
                   "overhead_ms": 0.0, "wall_total": 0.0}
            for seed, p in instances(tag):
                # One process, t+1 in-process repeats; the first is the
                # warm-up and is discarded.
                all_algo, wall_ms = time_run(cmd, p, env, repeat=t + 1)
                timed = all_algo[1:] if len(all_algo) > 1 else all_algo
                rec["algo"].extend(timed)
                # Fallback series for a binary built without bench_harness.h.
                if not timed:
                    rec["wall"].append(wall_ms)
                rec["seed_means"].append(
                    statistics.mean(timed) if timed else wall_ms)
                rec["wall_total"] += wall_ms / len(SEEDS)
                # Everything the process did that was NOT inside a timed
                # region: spawning, loading the binary, parsing the graph,
                # printing, and — for the algorithms that mutate their input —
                # the per-repeat copy of the edge vector. It is NOT parse cost
                # alone, so do not read it as one: it varies between algorithms
                # on the same input for exactly that reason. Averaged over
                # instances.
                rec["overhead_ms"] += (wall_ms - sum(all_algo)) / len(SEEDS)
            results[(tag, label)] = rec

            times = primary_times(rec)
            mu  = statistics.mean(times)
            sig = statistics.stdev(times) if len(times) > 1 else 0.0
            inst = (f"  σ_inst={sigma_inst(rec):5.2f}"
                    if len(SEEDS) > 1 else "")
            th  = t_hat(mu, n, m, label)
            log(f"    {label:10s}  {mu:8.2f} ms  σ={sig:5.2f}{inst}  t̂={th:9.1f} ops/ms"
                f"  [min={min(times):.2f} max={max(times):.2f}]"
                f"  (overhead {rec['overhead_ms']:.0f} ms, once per instance)")
    return results


def sigma_inst(rec):
    """Std dev of per-instance means; 0 with a single instance."""
    sm = rec.get("seed_means", [])
    return statistics.stdev(sm) if len(sm) > 1 else 0.0


# ── ξ data collection ──────────────────────────────────────────────────────────
def collect_xi(cases, n_repeats=3):
    """
    Run bms_xi and boruvka_xi, parse JSON from stderr.
    Returns list of row dicts.

    ξ_μ  (subscript μ) = geometric mean over all steps/rounds.
    (Note: do NOT use subscript t — 't' is the algorithm parameter.)
    """
    log("Collecting ξ_μ (shrinkage factor, geometric mean) …")
    rows = []
    bms_bin = str(BIN_DIR / "bms_xi")
    bor_bin = str(BIN_DIR / "boruvka_xi")

    for tag, n, m, _ in cases:
        if not data_path(tag).exists():
            continue
        log(f"  {tag}  (n={n:,}, instances={len(SEEDS)})")

        for seed, p in instances(tag):
            for rep in range(n_repeats):
                # ── BMS ──────────────────────────────────────────────────────────
                _, _, err = run_cmd([bms_bin, "--quiet"], stdin=p, capture_err=True, timeout=300)
                steps, summary = [], {}
                for line in err.splitlines():
                    try:
                        obj = json.loads(line)
                        if obj["type"] == "step":
                            steps.append(obj)
                        elif obj["type"] == "summary":
                            summary = obj
                    except Exception:
                        pass
                if summary:
                    rows.append({
                        "tag":          tag,
                        "n":            n,
                        "m":            m,
                        "algo":         "bms",
                        "seed":         seed,
                        "rep":          rep,
                        "P":            summary.get("P"),
                        "t":            summary.get("t"),
                        "steps_actual": summary.get("steps_actual"),
                        "xi_geomean":   summary.get("xi_geomean"),   # ξ_μ
                        "xi_theory":    summary.get("xi_theory"),     # 2^t
                        "C_trace":      summary.get("C_trace", []),
                        "M_trace":      summary.get("M_trace", []),
                        "xi_per_step":  [s["xi"] for s in steps],
                    })

                # ── Boruvka ───────────────────────────────────────────────────────
                _, _, err = run_cmd([bor_bin], stdin=p, capture_err=True, timeout=300)
                rounds, summary = [], {}
                for line in err.splitlines():
                    try:
                        obj = json.loads(line)
                        if obj["type"] == "round":
                            rounds.append(obj)
                        elif obj["type"] == "summary":
                            summary = obj
                    except Exception:
                        pass
                if summary:
                    rows.append({
                        "tag":           tag,
                        "n":             n,
                        "m":             m,
                        "algo":          "boruvka",
                        "seed":          seed,
                        "rep":           rep,
                        "rounds":        summary.get("rounds"),
                        "xi_geomean":    summary.get("xi_geomean"),  # ξ_μ
                        "xi_theory":     summary.get("xi_theory"),   # 2.0
                        "C_trace":       [r["C_before"] for r in rounds],
                        "xi_per_round":  [r["xi"]       for r in rounds],
                    })

        # Log averaged summary for this size
        for algo in ["bms", "boruvka"]:
            subset = [r for r in rows if r["tag"] == tag and r["algo"] == algo]
            if not subset:
                continue
            xi_vals = [r["xi_geomean"] for r in subset if r.get("xi_geomean")]
            if not xi_vals:
                continue
            xi_gm = statistics.mean(xi_vals)
            xi_th = subset[0].get("xi_theory", "?")
            if algo == "bms":
                extra = (f"  steps={subset[0].get('steps_actual', '?')}"
                         f"  t={subset[0].get('t', '?')}")
            else:
                extra = f"  rounds={subset[0].get('rounds', '?')}"
            log(f"    {algo:8s}  ξ_μ={xi_gm:.2f}  ξ_theory={xi_th}{extra}")
    return rows


# ── Output ─────────────────────────────────────────────────────────────────────
def write_outputs(cases, bins, timing_results, trials, xi_rows, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Timing CSV ────────────────────────────────────────────────────────────
    timing_csv = out_dir / "bench_timing.csv"
    with open(timing_csv, "w", newline="") as f:
        w = csv.writer(f)
        # mean_ms is MST construction only. wall_mean_ms is the whole process
        # including parsing the graph file — reported for reference, not for
        # comparing algorithms.
        w.writerow([
            "tag", "n", "m", "instances", "trials", "algo",
            "mean_ms", "min_ms", "max_ms", "sigma_ms", "sigma_inst_ms",
            "overhead_ms", "wall_total_ms",
            "t_hat_ops_per_ms", "vs_boruvka", "vs_bms",
        ])
        for tag, n, m, _ in cases:
            if not data_path(tag).exists():
                continue
            bor_t = primary_times(timing_results.get((tag, "boruvka"), {}))
            bms_t = primary_times(timing_results.get((tag, "bms"),     {}))
            bor_mu = statistics.mean(bor_t) if bor_t else None
            bms_mu = statistics.mean(bms_t) if bms_t else None
            t_val  = trials.get(tag, "?")
            for label, _, _ in bins:
                rec   = timing_results.get((tag, label), {})
                times = primary_times(rec)
                if not times:
                    continue
                mu  = statistics.mean(times)
                sig = statistics.stdev(times) if len(times) > 1 else 0.0
                th  = f"{t_hat(mu, n, m, label):.2f}"
                w.writerow([
                    tag, n, m, len(SEEDS), t_val, label,
                    f"{mu:.3f}", f"{min(times):.3f}", f"{max(times):.3f}",
                    f"{sig:.3f}", f"{sigma_inst(rec):.3f}",
                    f"{rec.get('overhead_ms', 0.0):.1f}",
                    f"{rec.get('wall_total', 0.0):.1f}",
                    th,
                    f"{bor_mu / mu:.4f}" if bor_mu and label != "boruvka" else "",
                    f"{bms_mu / mu:.4f}" if bms_mu and label != "bms"     else "",
                ])
    log(f"Timing CSV    → {timing_csv}")

    # ── ξ CSV ─────────────────────────────────────────────────────────────────
    xi_csv = out_dir / "bench_xi.csv"
    with open(xi_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "n", "m", "algo", "seed", "rep",
            "xi_geomean", "xi_theory",
            "steps_or_rounds", "t", "xi_sequence",
        ])
        for r in xi_rows:
            steps = steps_of(r)
            t_val = r.get("t", "")
            seq   = r.get("xi_per_step") or r.get("xi_per_round", [])
            w.writerow([
                r["tag"], r["n"], r["m"], r["algo"], r.get("seed", ""), r["rep"],
                f"{r['xi_geomean']:.4f}" if r.get("xi_geomean") else "",
                f"{r['xi_theory']:.2f}"  if r.get("xi_theory")  else "",
                steps, t_val,
                json.dumps([round(x, 4) for x in seq]),
            ])
    log(f"Xi CSV        → {xi_csv}")

    # ── Markdown report ────────────────────────────────────────────────────────
    md = []
    A = md.append

    A("# MST Benchmark Report\n")
    A(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    A("## Metric Definitions\n")
    A("### ξ_μ — Component Shrinkage Factor (geometric mean)\n")
    A("```")
    A("ξ_i = C_i / C_{i+1}  (ratio of component counts before/after step i)")
    A("")
    A("boruvka  per round:      theory lower bound  ξ ≥ 2")
    A("BMS      per super-step: theory value        ξ = 2^t")
    A("")
    A("ξ_μ = geometric mean over all steps (subscript μ = mean)")
    A("Note: subscript t is NOT used — 't' denotes the algorithm parameter.")
    A("```\n")

    A("### t̂ — Normalised Throughput (ops/ms)\n")
    A("```")
    A("t̂ = θ₀(n, m) / T_mean")
    A("")
    A("T_mean = MST construction time only, measured inside each binary with")
    A("         steady_clock after parsing and before printing. It does NOT")
    A("         include reading the graph file or writing the MST out.")
    A("")
    A("boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]")
    A("bms:               θ₀ = m · (log₂ n)^(2/3)        [O(m log^{2/3} n)]")
    A("")
    A("Note: (log₂ n)^(2/3) means the whole log₂(n) raised to power 2/3,")
    A("      NOT log₂(n^(2/3)).")
    A("")
    A("t̂ → constant as n increases  ⟹  bound is empirically tight.")
    A("")
    A("Both θ₀ carry the m term, so t̂ is comparable across densities.")
    A("```\n")

    A("### Trials formula\n")
    A("```")
    A("trials = max(min_trials, min(max_trials, floor((T/k − parse) / algo)))")
    A("")
    A("T     = total target time (ms) for this size")
    A("k     = number of algorithms compared")
    A("parse = fixed cost of reading the graph, paid ONCE per (size, algorithm)")
    A("algo  = slowest per-repeat MST construction time (ms) for this size")
    A("")
    A("Trials are in-process repeats (--repeat), so the graph is parsed once")
    A("and each extra trial costs only `algo`. The first repeat is discarded")
    A("as a warm-up.")
    A("Defaults: min_trials = 3, max_trials = 60")
    A("```\n")

    A("## ξ_μ Results (Shrinkage Factor)\n")
    A("| Size | n | m | algo | ξ_μ (measured) | ξ_theory | Steps/Rounds | t |")
    A("|------|---|---|------|----------------|---------|-------------|---|")
    xi_agg = defaultdict(list)
    for r in xi_rows:
        xi_agg[(r["tag"], r["algo"])].append(r.get("xi_geomean") or 0)

    for tag, n, m, _ in cases:
        if not data_path(tag).exists():
            continue
        for algo in ["bms", "boruvka"]:
            vals = xi_agg.get((tag, algo), [])
            if not vals:
                continue
            xi_gm = statistics.mean(vals)
            sample = next(
                (r for r in xi_rows if r["tag"] == tag and r["algo"] == algo), {}
            )
            xi_th  = sample.get("xi_theory", "")
            steps  = steps_of(sample)
            t_v    = sample.get("t", "")
            xi_th_s = f"{xi_th:.1f}" if isinstance(xi_th, (int, float)) else str(xi_th)
            A(f"| {tag} | {n:,} | {m:,} | {algo} | **{xi_gm:.2f}** "
              f"| {xi_th_s} | {steps} | {t_v} |")
        A("| | | | | | | | |")

    A("\n## t̂ and Timing Results\n")
    A("*Mean (ms) is MST construction only, measured inside each binary with "
      "steady_clock after parsing and before printing. Overhead (ms) is "
      "everything the process did outside the timed region — spawning, loading "
      "the binary, parsing the graph, printing, and for the algorithms that "
      "mutate their input, the per-repeat copy of the edge vector. It is paid "
      "once per (size, algorithm), is excluded from the comparison, and is NOT "
      "parse cost alone — which is why it differs between algorithms on the "
      "same input.*\n")
    A(f"*Graph instances per size: {len(SEEDS)} (seeds {SEEDS[0]}…{SEEDS[-1]}). "
      "Mean pools all instances. σ is the spread of individual runs; σ_inst is "
      "the standard deviation of the per-instance means, i.e. variation "
      "across random graphs (0 with a single instance).*\n")
    A("*vs boruvka = boruvka mean / algo mean; "
      "vs bms = bms mean / algo mean. Ratios > 1 indicate the algorithm is "
      "faster than the reference.*\n")
    A("| Size | n | m | trials | algo | Mean (ms) | σ | σ_inst | Overhead (ms) | t̂ (ops/ms) | vs boruvka | vs bms |")
    A("|------|---|---|--------|------|-----------|---|--------|---------------|-----------|-----------|--------|")
    for tag, n, m, _ in cases:
        if not data_path(tag).exists():
            continue
        bor_t = primary_times(timing_results.get((tag, "boruvka"), {}))
        bms_t = primary_times(timing_results.get((tag, "bms"),     {}))
        bor_mu = statistics.mean(bor_t) if bor_t else None
        bms_mu = statistics.mean(bms_t) if bms_t else None
        t_val  = trials.get(tag, "?")
        for label, _, _ in bins:
            rec   = timing_results.get((tag, label), {})
            times = primary_times(rec)
            if not times:
                continue
            mu  = statistics.mean(times)
            sig = statistics.stdev(times) if len(times) > 1 else 0.0
            th  = f"{t_hat(mu, n, m, label):.0f}"
            vs_bor = f"{bor_mu / mu:.2f}×" if bor_mu and label != "boruvka" else "—"
            vs_bms = f"{bms_mu / mu:.2f}×" if bms_mu and label != "bms"     else "—"
            A(f"| {tag} | {n:,} | {m:,} | {t_val} | {label} "
              f"| {mu:.2f} | {sig:.2f} | {sigma_inst(rec):.2f} "
              f"| {rec.get('overhead_ms', 0.0):.0f} "
              f"| **{th}** | {vs_bor} | {vs_bms} |")
        A("| | | | | | | | | | | | |")

    # Convergence table (sparse series only)
    A("\n## Convergence Analysis (Sparse graphs, m ≈ 5n)\n")
    A("ξ_μ stabilising → shrinkage rate matches theory. "
      "t̂ stabilising → complexity bound is tight.\n")
    A("| n | ξ_μ (bms) | ξ_theory (2^t) | ξ_μ (boruvka) | t̂_bms | t̂_boruvka |")
    A("|---|-----------|---------------|--------------|-------|----------|")
    for tag, n, m, _ in cases:
        if not is_sparse(tag):
            continue
        if not data_path(tag).exists():
            continue
        bms_xi_vals = xi_agg.get((tag, "bms"),     [])
        bor_xi_vals = xi_agg.get((tag, "boruvka"), [])
        bms_t_vals  = primary_times(timing_results.get((tag, "bms"),     {}))
        bor_t_vals  = primary_times(timing_results.get((tag, "boruvka"), {}))
        if not bms_xi_vals or not bor_xi_vals:
            continue
        xi_bms = statistics.mean(bms_xi_vals)
        xi_bor = statistics.mean(bor_xi_vals)
        sample = next(
            (r for r in xi_rows if r["tag"] == tag and r["algo"] == "bms"), {}
        )
        xi_th   = sample.get("xi_theory", "?")
        xi_th_s = f"{xi_th:.1f}" if isinstance(xi_th, (int, float)) else str(xi_th)
        th_bms  = t_hat(statistics.mean(bms_t_vals), n, m, "bms")     if bms_t_vals else 0
        th_bor  = t_hat(statistics.mean(bor_t_vals), n, m, "boruvka") if bor_t_vals else 0
        A(f"| {n:,} | {xi_bms:.2f} | {xi_th_s} | {xi_bor:.2f} "
          f"| {th_bms:.0f} | {th_bor:.0f} |")

    md_path = out_dir / "bench_report.md"
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    log(f"Report        → {md_path}")
    return timing_csv, xi_csv, md_path


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--build",        action="store_true",
                   help="Compile all binaries before running")
    p.add_argument("--gen",          action="store_true",
                   help="Generate test data before running")
    p.add_argument("--target-sec",   type=float, default=60.0,
                   help="Total timing budget per size in seconds (default 60)")
    p.add_argument("--min-trials",   type=int,   default=3,
                   help="Minimum timed trials per algorithm (default 3)")
    p.add_argument("--max-trials",   type=int,   default=60,
                   help="Maximum timed trials per algorithm (default 60)")
    p.add_argument("--par-threads",  type=str,   default="2,4",
                   help="Comma-separated thread counts for parallel_boruvka (default 2,4)")
    p.add_argument("--output-dir",   type=str,   default="results",
                   help="Output directory (default ./results)")
    p.add_argument("--sizes",        type=str,   default=None,
                   help="Comma-separated size tags to run (default: all)")
    p.add_argument("--skip-verify",  action="store_true",
                   help="Skip correctness verification")
    p.add_argument("--xi-only",      action="store_true",
                   help="Collect ξ data only; skip multi-trial timing")
    p.add_argument("--xi-repeats",   type=int,   default=3,
                   help="Number of ξ collection repeats per size (default 3)")
    p.add_argument("--seed",         type=int,   default=42,
                   help="Random seed for gen (default 42)")
    p.add_argument("--n-seeds",      type=int,   default=1,
                   help="Number of independent graph instances per (n,m) config (default 1). "
                        "Seeds used: seed, seed+1, ..., seed+n_seeds-1. "
                        "Every algorithm is timed on every instance; means pool all "
                        "instances and σ_inst reports the spread across them. "
                        "The --target-sec budget applies per instance.")
    p.add_argument("--no-par",       action="store_true",
                   help="Exclude parallel Borůvka from timing (useful on single-core machines)")
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    target_ms   = args.target_sec * 1000.0
    par_threads = [int(x) for x in args.par_threads.split(",")]
    out_dir     = Path(args.output_dir)
    if args.n_seeds < 1:
        log("--n-seeds must be at least 1", "ERROR")
        sys.exit(1)
    SEEDS[:] = [args.seed + i for i in range(args.n_seeds)]

    cases = TEST_CASES
    if args.sizes:
        wanted = set(args.sizes.split(","))
        cases  = [c for c in cases if c[0] in wanted]
        if not cases:
            log(f"No cases match --sizes {args.sizes}", "ERROR")
            sys.exit(1)

    bins = make_timing_bins(par_threads, include_par=not args.no_par)

    log("=" * 64)
    log("MST Benchmark  ─  ξ_μ (shrinkage)  +  t̂ (normalised throughput)")
    log(f"  Sizes   : {[c[0] for c in cases]}")
    log(f"  Algos   : {[l for l, _, _ in bins]}")
    log(f"  Seeds   : {SEEDS}")
    log(f"  Target  : {args.target_sec}s / size  "
        f"trials=[{args.min_trials}, {args.max_trials}]")
    log("=" * 64)

    if args.build:
        build_all(par_threads)

    if args.gen:
        generate_data(cases)

    missing = [p.name for c in cases for _, p in instances(c[0])
               if not p.exists()]
    if missing:
        log(f"Missing data files: {missing}. Re-run with --gen.", "ERROR")
        sys.exit(1)

    if not args.skip_verify and not args.xi_only:
        verify_all(cases, bins)

    xi_rows = collect_xi(cases, n_repeats=args.xi_repeats)

    timing_results, trials = {}, {}
    if not args.xi_only:
        probe_ms = probe_once(cases, bins)
        trials   = calc_trials(
            probe_ms, cases, bins,
            target_ms, args.min_trials, args.max_trials,
        )
        log("")
        log("Main timing …")
        log("")
        timing_results = run_timing(cases, bins, trials)

    write_outputs(cases, bins, timing_results, trials, xi_rows, out_dir)
    log("\nAll done.")
