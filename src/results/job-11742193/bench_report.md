# MST Benchmark Report

Generated: 2026-11-08 21:39:17

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

T_mean = MST construction time only, measured inside each binary with
         steady_clock after parsing and before printing. It does NOT
         include reading the graph file or writing the MST out.

boruvka / par_T*:  θ₀ = m · log₂(n)              [O(m log n)]
bms:               θ₀ = n · (log₂ n)^(2/3)        [O(n log^{2/3} n)]

Note: (log₂ n)^(2/3) means the whole log₂(n) raised to power 2/3,
      NOT log₂(n^(2/3)).

t̂ → constant as n increases  ⟹  bound is empirically tight.
```

### Trials formula

```
trials = max(min_trials, min(max_trials, floor((T/k − parse) / algo)))

T     = total target time (ms) for this size
k     = number of algorithms compared
parse = fixed cost of reading the graph, paid ONCE per (size, algorithm)
algo  = slowest per-repeat MST construction time (ms) for this size

Trials are in-process repeats (--repeat), so the graph is parsed once
and each extra trial costs only `algo`. The first repeat is discarded
as a warm-up.
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
| 500k | 500,000 | 2,500,000 | bms | **44.19** | 256.0 | 2 | 8 |
| 500k | 500,000 | 2,500,000 | boruvka | **5.35** | 2.0 | 7 |  |
| | | | | | | | |
| 1m | 1,000,000 | 5,000,000 | bms | **42.84** | 256.0 | 2 | 8 |
| 1m | 1,000,000 | 5,000,000 | boruvka | **5.90** | 2.0 | 7 |  |
| | | | | | | | |
| dense_50k | 50,000 | 2,500,000 | bms | **1.00** | 128.0 |  | 7 |
| dense_50k | 50,000 | 2,500,000 | boruvka | **5.41** | 2.0 | 6 |  |
| | | | | | | | |
| dense_100k | 100,000 | 5,000,000 | bms | **1.00** | 128.0 |  | 7 |
| dense_100k | 100,000 | 5,000,000 | boruvka | **5.21** | 2.0 | 6 |  |
| | | | | | | | |
| 1m_ties | 1,000,000 | 5,000,000 | bms | **42.41** | 256.0 | 2 | 8 |
| 1m_ties | 1,000,000 | 5,000,000 | boruvka | **5.90** | 2.0 | 7 |  |
| | | | | | | | |
| 1m_small | 1,000,000 | 5,000,000 | bms | **42.07** | 256.0 | 2 | 8 |
| 1m_small | 1,000,000 | 5,000,000 | boruvka | **6.15** | 2.0 | 7 |  |
| | | | | | | | |

## t̂ and Timing Results

*Mean (ms) is MST construction only, measured inside each binary with steady_clock after parsing and before printing. Parse (ms) is everything else the process did — reading the graph file and printing — paid once per (size, algorithm) rather than once per trial, and excluded from the comparison.*

*vs boruvka = boruvka mean / algo mean; vs bms = bms mean / algo mean. Ratios > 1 indicate the algorithm is faster than the reference.*

| Size | n | m | trials | algo | Mean (ms) | σ | Parse (ms) | t̂ (ops/ms) | vs boruvka | vs bms |
|------|---|---|--------|------|-----------|---|------------|-----------|-----------|--------|
| 10k | 10,000 | 50,000 | 60 | boruvka | 3.63 | 0.02 | 485 | **182998** | — | 0.78× |
| 10k | 10,000 | 50,000 | 60 | kruskal | 3.33 | 0.03 | 439 | **199457** | 1.09× | 0.85× |
| 10k | 10,000 | 50,000 | 60 | filter_kruskal | 2.10 | 0.01 | 581 | **317066** | 1.73× | 1.34× |
| 10k | 10,000 | 50,000 | 60 | bms | 2.82 | 0.07 | 468 | **19908** | 1.29× | — |
| 10k | 10,000 | 50,000 | 60 | par_T2 | 8.75 | 0.04 | 205 | **75931** | 0.41× | 0.32× |
| 10k | 10,000 | 50,000 | 60 | par_T4 | 5.75 | 0.06 | 259 | **115467** | 0.63× | 0.49× |
| | | | | | | | | | | |
| 100k | 100,000 | 500,000 | 60 | boruvka | 58.58 | 1.05 | 390 | **141774** | — | 0.63× |
| 100k | 100,000 | 500,000 | 60 | kruskal | 40.76 | 0.17 | 511 | **203747** | 1.44× | 0.90× |
| 100k | 100,000 | 500,000 | 60 | filter_kruskal | 24.61 | 0.16 | 536 | **337503** | 2.38× | 1.50× |
| 100k | 100,000 | 500,000 | 60 | bms | 36.82 | 0.93 | 374 | **17681** | 1.59× | — |
| 100k | 100,000 | 500,000 | 60 | par_T2 | 74.62 | 0.47 | 254 | **111292** | 0.78× | 0.49× |
| 100k | 100,000 | 500,000 | 60 | par_T4 | 49.26 | 0.41 | 355 | **168607** | 1.19× | 0.75× |
| | | | | | | | | | | |
| 500k | 500,000 | 2,500,000 | 60 | boruvka | 449.76 | 0.81 | 810 | **105230** | — | 0.75× |
| 500k | 500,000 | 2,500,000 | 60 | kruskal | 238.92 | 0.28 | 1630 | **198093** | 1.88× | 1.41× |
| 500k | 500,000 | 2,500,000 | 60 | filter_kruskal | 158.44 | 0.63 | 1835 | **298712** | 2.84× | 2.12× |
| 500k | 500,000 | 2,500,000 | 60 | bms | 335.97 | 5.05 | 1865 | **10571** | 1.34× | — |
| 500k | 500,000 | 2,500,000 | 60 | par_T2 | 471.47 | 3.26 | 936 | **100385** | 0.95× | 0.71× |
| 500k | 500,000 | 2,500,000 | 60 | par_T4 | 322.61 | 2.73 | 765 | **146708** | 1.39× | 1.04× |
| | | | | | | | | | | |
| 1m | 1,000,000 | 5,000,000 | 49 | boruvka | 834.25 | 39.11 | 1336 | **119458** | — | 0.63× |
| 1m | 1,000,000 | 5,000,000 | 49 | kruskal | 491.18 | 1.02 | 2930 | **202893** | 1.70× | 1.08× |
| 1m | 1,000,000 | 5,000,000 | 49 | filter_kruskal | 393.85 | 5.48 | 3722 | **253035** | 2.12× | 1.34× |
| 1m | 1,000,000 | 5,000,000 | 49 | bms | 528.40 | 0.92 | 3073 | **13912** | 1.58× | — |
| 1m | 1,000,000 | 5,000,000 | 49 | par_T2 | 931.36 | 14.39 | 1256 | **107003** | 0.90× | 0.57× |
| 1m | 1,000,000 | 5,000,000 | 49 | par_T4 | 705.38 | 10.71 | 1465 | **141283** | 1.18× | 0.75× |
| | | | | | | | | | | |
| dense_50k | 50,000 | 2,500,000 | 60 | boruvka | 203.73 | 0.40 | 840 | **191551** | — | 0.79× |
| dense_50k | 50,000 | 2,500,000 | 60 | kruskal | 206.23 | 0.19 | 1605 | **189224** | 0.99× | 0.78× |
| dense_50k | 50,000 | 2,500,000 | 60 | filter_kruskal | 61.90 | 0.23 | 1860 | **630409** | 3.29× | 2.60× |
| dense_50k | 50,000 | 2,500,000 | 60 | bms | 160.96 | 0.82 | 1690 | **1940** | 1.27× | — |
| dense_50k | 50,000 | 2,500,000 | 60 | par_T2 | 309.95 | 1.23 | 724 | **125905** | 0.66× | 0.52× |
| dense_50k | 50,000 | 2,500,000 | 60 | par_T4 | 181.99 | 0.85 | 969 | **214428** | 1.12× | 0.88× |
| | | | | | | | | | | |
| dense_100k | 100,000 | 5,000,000 | 60 | boruvka | 490.83 | 0.47 | 1372 | **169199** | — | 0.76× |
| dense_100k | 100,000 | 5,000,000 | 60 | kruskal | 437.09 | 0.47 | 3295 | **190003** | 1.12× | 0.85× |
| dense_100k | 100,000 | 5,000,000 | 60 | filter_kruskal | 187.22 | 3.36 | 3975 | **443583** | 2.62× | 2.00× |
| dense_100k | 100,000 | 5,000,000 | 60 | bms | 373.54 | 0.83 | 3160 | **1743** | 1.31× | — |
| dense_100k | 100,000 | 5,000,000 | 60 | par_T2 | 651.82 | 1.51 | 1621 | **127410** | 0.75× | 0.57× |
| dense_100k | 100,000 | 5,000,000 | 60 | par_T4 | 382.19 | 1.81 | 1474 | **217293** | 1.28× | 0.98× |
| | | | | | | | | | | |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | boruvka | 1046.57 | 7.01 | 1220 | **95223** | — | 0.56× |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | kruskal | 1066.99 | 0.59 | 2469 | **93401** | 0.98× | 0.55× |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | filter_kruskal | 1240.85 | 14.32 | 3200 | **80314** | 0.84× | 0.47× |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | bms | 587.13 | 1.52 | 2675 | **12521** | 1.78× | — |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | par_T2 | 897.72 | 17.49 | 1308 | **111012** | 1.17× | 0.65× |
| 1m_ties | 1,000,000 | 5,000,000 | 44 | par_T4 | 653.98 | 3.84 | 1107 | **152388** | 1.60× | 0.90× |
| | | | | | | | | | | |
| 1m_small | 1,000,000 | 5,000,000 | 42 | boruvka | 781.09 | 14.28 | 1156 | **127588** | — | 0.63× |
| 1m_small | 1,000,000 | 5,000,000 | 42 | kruskal | 880.06 | 0.52 | 2451 | **113240** | 0.89× | 0.56× |
| 1m_small | 1,000,000 | 5,000,000 | 42 | filter_kruskal | 516.39 | 17.72 | 3193 | **192990** | 1.51× | 0.95× |
| 1m_small | 1,000,000 | 5,000,000 | 42 | bms | 491.76 | 0.60 | 2519 | **14949** | 1.59× | — |
| 1m_small | 1,000,000 | 5,000,000 | 42 | par_T2 | 882.00 | 6.84 | 1295 | **112991** | 0.89× | 0.56× |
| 1m_small | 1,000,000 | 5,000,000 | 42 | par_T4 | 654.88 | 5.18 | 1342 | **152176** | 1.19× | 0.75× |
| | | | | | | | | | | |

## Convergence Analysis (Sparse graphs, m ≈ 5n)

ξ_μ stabilising → shrinkage rate matches theory. t̂ stabilising → complexity bound is tight.

| n | ξ_μ (bms) | ξ_theory (2^t) | ξ_μ (boruvka) | t̂_bms | t̂_boruvka |
|---|-----------|---------------|--------------|-------|----------|
| 10,000 | 40.82 | 64.0 | 5.07 | 19908 | 182998 |
| 100,000 | 45.64 | 128.0 | 5.67 | 17681 | 141774 |
| 500,000 | 44.19 | 256.0 | 5.35 | 10571 | 105230 |
| 1,000,000 | 42.84 | 256.0 | 5.90 | 13912 | 119458 |
