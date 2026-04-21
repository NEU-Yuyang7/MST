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

    boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]
    bms:               θ₀ = n · (log₂ n)^(2/3)        [O(n log^{2/3} n)]

    t̂ → constant as n grows  ⟹  bound is empirically tight.

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
    "boruvka":    SCRIPT_DIR / "boruvka.cpp",
    "parallel":   SCRIPT_DIR / "parallel_boruvka.cpp",
    "bms":        SCRIPT_DIR / "BMSBoruvka.cpp",
    "bms_xi":     SCRIPT_DIR / "bms_instrumented.cpp",
    "boruvka_xi": SCRIPT_DIR / "boruvka_xi.cpp",
    "gen":        SCRIPT_DIR / "gen.cpp",
}

# ── Test cases ─────────────────────────────────────────────────────────────────
# (tag, n, m, extra_gen_args)
# Sparse series: m ≈ 5n  →  used for ξ and t̂ convergence analysis
# Dense series:  m ≈ 50n →  shows BMS degrading to O(m log n)
TEST_CASES = [
    ("10k",        10_000,      50_000, []),
    ("100k",      100_000,     500_000, []),
    ("500k",      500_000,   2_500_000, []),
    ("1m",      1_000_000,   5_000_000, []),
    ("dense_50k",   50_000,   2_500_000, []),
    ("dense_100k", 100_000,   5_000_000, []),
    ("1m_ties",  1_000_000,   5_000_000, ["--weights", "uniform"]),
    ("1m_small", 1_000_000,   5_000_000, ["--weights", "small"]),
]

COMPILE_FLAGS = ["-O2", "-std=c++17"]

# ── Utilities ─────────────────────────────────────────────────────────────────
def log(msg, level="INFO"):
    print(f"[{time.strftime('%H:%M:%S')}] {level:5s}  {msg}", flush=True)


def run_cmd(cmd, *, stdin=None, capture_out=False, capture_err=False,
            env=None, timeout=600):
    r = subprocess.run(
        cmd,
        stdin=open(stdin) if isinstance(stdin, (str, Path)) else stdin,
        stdout=subprocess.PIPE if capture_out  else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_err  else subprocess.DEVNULL,
        env={**os.environ, **(env or {})},
        timeout=timeout,
    )
    return (
        r.returncode,
        r.stdout.decode() if capture_out  and r.stdout else "",
        r.stderr.decode() if capture_err  and r.stderr else "",
    )


def time_run(cmd, stdin, env=None):
    """Run cmd with stdin redirected; return wall-clock ms."""
    e = {**os.environ, **(env or {})}
    with open(stdin) as fh:
        t0 = time.perf_counter()
        subprocess.run(
            cmd, stdin=fh,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            env=e, check=True,
        )
        return (time.perf_counter() - t0) * 1000.0


def get_weight(cmd, stdin, env=None):
    """Run cmd and parse 'Total weight = ...' from stdout."""
    _, out, _ = run_cmd(cmd, stdin=stdin, capture_out=True,
                        env={**os.environ, **(env or {})})
    for line in out.splitlines():
        if line.startswith("Total weight"):
            return line.split("=")[1].strip()
    return None


def data_path(tag):
    return DATA_DIR / f"test_{tag}.txt"


# ── t̂ calculation ─────────────────────────────────────────────────────────────
def t_hat(mean_ms, n, m, label):
    """
    Normalised throughput = θ₀(n, m) / T_mean   [ops / ms]

    θ₀ definitions (matching the paper):
      boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n) bound]
      bms:               θ₀ = n · (log₂ n)^(2/3)        [O(n log^{2/3} n) bound]

    Note: the exponent 2/3 is applied to the entire log₂(n), i.e.
      (log₂ n)^(2/3)   NOT   log₂(n^(2/3))
    """
    log2n = math.log2(n) if n > 1 else 1.0
    if label == "bms":
        theta = n * (log2n ** (2.0 / 3.0))
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
         [str(SOURCES["bms_xi"]), "-o", str(BIN_DIR / "bms_xi")],
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
def generate_data(cases, seed):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    gen_bin = str(BIN_DIR / "gen")
    for tag, n, m, extra in cases:
        p = data_path(tag)
        if p.exists():
            log(f"  {p.name} already exists, skipping")
            continue
        cmd = [gen_bin, str(n), str(m), "--seed", str(seed), "--connected"] + extra
        log(f"  Generating {p.name}  (n={n:,} m={m:,})")
        with open(p, "w") as fout:
            if subprocess.run(cmd, stdout=fout, stderr=subprocess.DEVNULL).returncode:
                log(f"  gen failed for {tag}", "ERROR")
                sys.exit(1)
    log("Data generation complete.")


# ── Algorithm list ─────────────────────────────────────────────────────────────
def make_timing_bins(par_threads):
    """
    Returns list of (label, cmd, env_overrides).
    par_T2, par_T4, … set OMP_NUM_THREADS accordingly.
    """
    bins = [
        ("boruvka", [str(BIN_DIR / "boruvka")], {}),
        ("bms",     [str(BIN_DIR / "bms")],     {}),
    ]
    for t in par_threads:
        bins.append((
            f"par_T{t}",
            [str(BIN_DIR / "parallel_boruvka")],
            {"OMP_NUM_THREADS": str(t)},
        ))
    return bins


# ── Correctness verification ───────────────────────────────────────────────────
def verify_all(cases, bins):
    log("Correctness check …")
    failures = 0
    for tag, *_ in cases:
        p = data_path(tag)
        if not p.exists():
            continue
        weights = {label: get_weight(cmd, p, env) for label, cmd, env in bins}
        if len(set(weights.values())) == 1:
            log(f"  OK   {tag:15s}  weight = {list(weights.values())[0]}")
        else:
            log(f"  FAIL {tag}: {weights}", "ERROR")
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
            time_run(cmd, p, env)          # warm-up — discarded
            ms = time_run(cmd, p, env)     # probe time
            probe[(tag, label)] = ms
            log(f"  {tag:15s}  {label:10s}  {ms:.0f} ms")
    return probe


def calc_trials(probe, cases, bins, target_ms, min_t, max_t):
    """
    trials = max(min_t, min(max_t, floor(T / (k · budget))))

    T       = target_ms  (total budget for this size)
    k       = len(bins)  (number of algorithms)
    budget  = slowest probe time across all algorithms for this size

    Each algorithm gets an equal share T/k of the budget.
    """
    trials = {}
    k = len(bins)
    for tag, *_ in cases:
        if not data_path(tag).exists():
            continue
        budget = max(
            (probe.get((tag, label), 1e9) for label, _, _ in bins),
            default=1000.0,
        )
        per_algo_budget = target_ms / k
        t = max(min_t, min(max_t, int(per_algo_budget // budget)))
        trials[tag] = t
        log(f"  {tag:15s}  slowest={budget:.0f} ms  per-algo budget={per_algo_budget:.0f} ms"
            f"  → {t} trials")
    return trials


# ── Main timing loop ───────────────────────────────────────────────────────────
def run_timing(cases, bins, trials):
    results = {}
    total = sum(1 for tag, *_ in cases if data_path(tag).exists())
    done  = 0
    for tag, n, m, _ in cases:
        p = data_path(tag)
        if not p.exists():
            continue
        done += 1
        t = trials.get(tag, 3)
        log(f"[{done}/{total}] {tag}  n={n:,} m={m:,}  trials={t}")
        for label, cmd, env in bins:
            time_run(cmd, p, env)                              # warm-up (discarded)
            times = [time_run(cmd, p, env) for _ in range(t)]
            results[(tag, label)] = times
            mu  = statistics.mean(times)
            sig = statistics.stdev(times) if len(times) > 1 else 0.0
            th  = t_hat(mu, n, m, label)
            log(f"    {label:10s}  {mu:8.1f} ms  σ={sig:5.1f}  "
                f"t̂={th:9.1f} ops/ms  [min={min(times):.0f} max={max(times):.0f}]")
    return results


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
        p = data_path(tag)
        if not p.exists():
            continue
        log(f"  {tag}  (n={n:,})")

        for rep in range(n_repeats):
            # ── BMS ──────────────────────────────────────────────────────────
            _, _, err = run_cmd([bms_bin], stdin=p, capture_err=True, timeout=300)
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
        w.writerow([
            "tag", "n", "m", "trials", "algo",
            "mean_ms", "min_ms", "max_ms", "sigma_ms",
            "t_hat_ops_per_ms", "vs_boruvka", "vs_bms",
        ])
        for tag, n, m, _ in cases:
            if not data_path(tag).exists():
                continue
            bor_times = timing_results.get((tag, "boruvka"), [])
            bms_times = timing_results.get((tag, "bms"),     [])
            bor_mu = statistics.mean(bor_times) if bor_times else None
            bms_mu = statistics.mean(bms_times) if bms_times else None
            t_val  = trials.get(tag, "?")
            for label, _, _ in bins:
                times = timing_results.get((tag, label), [])
                if not times:
                    continue
                mu  = statistics.mean(times)
                sig = statistics.stdev(times) if len(times) > 1 else 0.0
                th  = t_hat(mu, n, m, label)
                w.writerow([
                    tag, n, m, t_val, label,
                    f"{mu:.2f}", f"{min(times):.0f}", f"{max(times):.0f}",
                    f"{sig:.2f}", f"{th:.2f}",
                    f"{bor_mu / mu:.4f}" if bor_mu and label != "boruvka" else "",
                    f"{bms_mu / mu:.4f}" if bms_mu and label != "bms"     else "",
                ])
    log(f"Timing CSV    → {timing_csv}")

    # ── ξ CSV ─────────────────────────────────────────────────────────────────
    xi_csv = out_dir / "bench_xi.csv"
    with open(xi_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tag", "n", "m", "algo", "rep",
            "xi_geomean", "xi_theory",
            "steps_or_rounds", "t", "xi_sequence",
        ])
        for r in xi_rows:
            steps = r.get("steps_actual") or r.get("rounds", "")
            t_val = r.get("t", "")
            seq   = r.get("xi_per_step") or r.get("xi_per_round", [])
            w.writerow([
                r["tag"], r["n"], r["m"], r["algo"], r["rep"],
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
    A("boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]")
    A("bms:               θ₀ = n · (log₂ n)^(2/3)        [O(n log^{2/3} n)]")
    A("")
    A("Note: (log₂ n)^(2/3) means the whole log₂(n) raised to power 2/3,")
    A("      NOT log₂(n^(2/3)).")
    A("")
    A("t̂ → constant as n increases  ⟹  bound is empirically tight.")
    A("```\n")

    A("### Trials formula\n")
    A("```")
    A("trials = max(min_trials, min(max_trials, floor(T / (k · budget))))")
    A("")
    A("T      = total target time (ms) for this size")
    A("k      = number of algorithms compared")
    A("budget = slowest single-run probe time (ms) for this size")
    A("")
    A("Each algorithm receives an equal share T/k of the budget.")
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
            steps  = sample.get("steps_actual") or sample.get("rounds", "")
            t_v    = sample.get("t", "")
            xi_th_s = f"{xi_th:.1f}" if isinstance(xi_th, (int, float)) else str(xi_th)
            A(f"| {tag} | {n:,} | {m:,} | {algo} | **{xi_gm:.2f}** "
              f"| {xi_th_s} | {steps} | {t_v} |")
        A("| | | | | | | | |")

    A("\n## t̂ and Timing Results\n")
    A("*vs boruvka = boruvka mean / algo mean; "
      "vs bms = bms mean / algo mean. Ratios > 1 indicate the algorithm is "
      "faster than the reference.*\n")
    A("| Size | n | m | trials | algo | Mean (ms) | σ | t̂ (ops/ms) | vs boruvka | vs bms |")
    A("|------|---|---|--------|------|-----------|---|-----------|-----------|--------|")
    for tag, n, m, _ in cases:
        if not data_path(tag).exists():
            continue
        bor_times = timing_results.get((tag, "boruvka"), [])
        bms_times = timing_results.get((tag, "bms"),     [])
        bor_mu = statistics.mean(bor_times) if bor_times else None
        bms_mu = statistics.mean(bms_times) if bms_times else None
        t_val  = trials.get(tag, "?")
        for label, _, _ in bins:
            times = timing_results.get((tag, label), [])
            if not times:
                continue
            mu  = statistics.mean(times)
            sig = statistics.stdev(times) if len(times) > 1 else 0.0
            th  = t_hat(mu, n, m, label)
            vs_bor = f"{bor_mu / mu:.2f}×" if bor_mu and label != "boruvka" else "—"
            vs_bms = f"{bms_mu / mu:.2f}×" if bms_mu and label != "bms"     else "—"
            A(f"| {tag} | {n:,} | {m:,} | {t_val} | {label} "
              f"| {mu:.1f} | {sig:.1f} | **{th:.0f}** | {vs_bor} | {vs_bms} |")
        A("| | | | | | | | | | |")

    # Convergence table (sparse series only)
    A("\n## Convergence Analysis (Sparse graphs, m ≈ 5n)\n")
    A("ξ_μ stabilising → shrinkage rate matches theory. "
      "t̂ stabilising → complexity bound is tight.\n")
    A("| n | ξ_μ (bms) | ξ_theory (2^t) | ξ_μ (boruvka) | t̂_bms | t̂_boruvka |")
    A("|---|-----------|---------------|--------------|-------|----------|")
    for tag, n, m, _ in cases:
        if any(x in tag for x in ("dense", "ties", "small")):
            continue
        if not data_path(tag).exists():
            continue
        bms_xi_vals = xi_agg.get((tag, "bms"),     [])
        bor_xi_vals = xi_agg.get((tag, "boruvka"), [])
        bms_t_vals  = timing_results.get((tag, "bms"),     [])
        bor_t_vals  = timing_results.get((tag, "boruvka"), [])
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
    p.add_argument("--target-sec",   type=float, default=30.0,
                   help="Total timing budget per size in seconds (default 30)")
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
    return p.parse_args()


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    target_ms   = args.target_sec * 1000.0
    par_threads = [int(x) for x in args.par_threads.split(",")]
    out_dir     = Path(args.output_dir)

    cases = TEST_CASES
    if args.sizes:
        wanted = set(args.sizes.split(","))
        cases  = [c for c in cases if c[0] in wanted]
        if not cases:
            log(f"No cases match --sizes {args.sizes}", "ERROR")
            sys.exit(1)

    bins = make_timing_bins(par_threads)

    log("=" * 64)
    log("MST Benchmark  ─  ξ_μ (shrinkage)  +  t̂ (normalised throughput)")
    log(f"  Sizes   : {[c[0] for c in cases]}")
    log(f"  Algos   : {[l for l, _, _ in bins]}")
    log(f"  Target  : {args.target_sec}s / size  "
        f"trials=[{args.min_trials}, {args.max_trials}]")
    log("=" * 64)

    if args.build:
        build_all(par_threads)

    if args.gen:
        generate_data(cases, args.seed)

    missing = [c[0] for c in cases if not data_path(c[0]).exists()]
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
