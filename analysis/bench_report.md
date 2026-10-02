# MST Benchmark Report

Generated: 2026-10-02 22:44:15

## Metric Definitions

### ξ_μ — Component Shrinkage Factor (geometric mean)

```
ξ_i = C_i / C_{i+1}  (ratio of component counts before/after step i)

boruvka  per round:      theory lower bound  ξ ≥ 2
BMS      per super-step: theory value        ξ = 2^t

ξ_μ = geometric mean over all steps (subscript μ = mean)
Note: subscript t is NOT used — 't' denotes the algorithm parameter.
```

### t̂ — Normalised Throughput (ops/ms)

```
t̂ = θ₀(n, m) / T_mean

boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]
bms:               θ₀ = n · (log₂ n)^(2/3)        [O(n log^{2/3} n)]

Note: (log₂ n)^(2/3) means the whole log₂(n) raised to power 2/3,
      NOT log₂(n^(2/3)).

t̂ → constant as n increases  ⟹  bound is empirically tight.
```

### Trials formula

```
trials = max(min_trials, min(max_trials, floor(T / (k · budget))))

T      = total target time (ms) for this size
k      = number of algorithms compared
budget = slowest single-run probe time (ms) for this size

Each algorithm receives an equal share T/k of the budget.
Defaults: min_trials = 3, max_trials = 60
```

## ξ_μ Results (Shrinkage Factor)

| Size | n | m | algo | ξ_μ (measured) | ξ_theory | Steps/Rounds | t |
|------|---|---|------|----------------|---------|-------------|---|
| 10k | 10,000 | 50,000 | bms | **40.82** | 64.0 | 2 | 6 |
| 10k | 10,000 | 50,000 | boruvka | **5.07** | 2.0 | 5 |  |
| | | | | | | | |
| 100k | 100,000 | 500,000 | bms | **45.64** | 128.0 | 2 | 7 |
| 100k | 100,000 | 500,000 | boruvka | **5.67** | 2.0 | 6 |  |
| | | | | | | | |

## t̂ and Timing Results

*vs boruvka = boruvka mean / algo mean; vs bms = bms mean / algo mean. Ratios > 1 indicate the algorithm is faster than the reference.*

| Size | n | m | trials | algo | Mean (ms) | σ | t̂ (ops/ms) | vs boruvka | vs bms |
|------|---|---|--------|------|-----------|---|-----------|-----------|--------|
| 10k | 10,000 | 50,000 | 50 | boruvka | 38.9 | 1.6 | **17088** | — | 0.40× |
| 10k | 10,000 | 50,000 | 50 | kruskal | 12.5 | 0.9 | **53033** | 3.10× | 1.24× |
| 10k | 10,000 | 50,000 | 50 | filter_kruskal | 11.3 | 0.7 | **58786** | 3.44× | 1.37× |
| 10k | 10,000 | 50,000 | 50 | bms | 15.5 | 1.8 | **3616** | 2.51× | — |
| | | | | | | | | | |
| 100k | 100,000 | 500,000 | 4 | boruvka | 419.9 | 10.9 | **19777** | — | 0.39× |
| 100k | 100,000 | 500,000 | 4 | kruskal | 130.6 | 4.1 | **63570** | 3.21× | 1.25× |
| 100k | 100,000 | 500,000 | 4 | filter_kruskal | 106.8 | 2.4 | **77766** | 3.93× | 1.54× |
| 100k | 100,000 | 500,000 | 4 | bms | 164.0 | 2.9 | **3971** | 2.56× | — |
| | | | | | | | | | |

## Convergence Analysis (Sparse graphs, m ≈ 5n)

ξ_μ stabilising → shrinkage rate matches theory. t̂ stabilising → complexity bound is tight.

| n | ξ_μ (bms) | ξ_theory (2^t) | ξ_μ (boruvka) | t̂_bms | t̂_boruvka |
|---|-----------|---------------|--------------|-------|----------|
| 10,000 | 40.82 | 64.0 | 5.07 | 3616 | 17088 |
| 100,000 | 45.64 | 128.0 | 5.67 | 3971 | 19777 |
