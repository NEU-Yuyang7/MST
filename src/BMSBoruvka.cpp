/*
 * BMSBoruvka.cpp  (v7)
 *
 * ── Complexity Summary (Sparse Graph Case: m = O(n)) ───────────────
 *
 *   Parameters:
 *       t = ⌈log^{2/3}(n)⌉
 *       P = ⌈log(n) / t⌉ = O(log^{1/3}(n))
 *       k = ⌊m / P⌋  edges activated per super-step
 *
 *   Phase A (activation, default mode "select"):
 *       One std::nth_element call per super-step on the unactivated
 *       suffix of the edge array (no global sort).
 *       Σ_{i=0}^{P-2} (m - i·k) = O(m · P) = O(m · log^{1/3}(n))
 *
 *   Component shrinkage:
 *       C_0 = n
 *       C_{i+1} ≤ C_i / 2^t
 *       (Each super-step performs t Borůvka rounds,
 *        and each round at least halves the number of components
 *        that have an activated outgoing edge.)
 *
 *   Sparse graph assumption:
 *       m_i ≤ C_i
 *       (Average degree per super-node remains O(1).)
 *
 *   Total edge volume across super-steps:
 *       Σ m_i ≤ Σ C_i
 *              = n · Σ_{i=0}^{P-1} 2^{-it}
 *              = n / (1 - 2^{-t})
 *              = O(n)
 *
 *   Phase B total cost:
 *       t · Σ m_i
 *       = O(t · n)
 *       = O(n · log^{2/3}(n))    ← dominant term
 *
 *   Phase C total cost:
 *       Σ m_i = O(n)
 *
 *   origToComp maintenance cost:
 *       O(n · P) = O(n · log^{1/3}(n))
 *
 *   Overall complexity:
 *       O(n · log^{2/3}(n))
 *
 * ── Changes in v7 ───────────────────────────────────────────────────
 *
 *   1. Phase A uses per-super-step std::nth_element (as described in
 *      the paper, Lemma 2) instead of a global sort in the constructor.
 *   2. Optional "--phase-a=radix" mode: one upfront O(m) LSD radix sort
 *      (replaces the old countingSortEdges, which silently fell back to
 *      std::sort when the weight range exceeded 5·10^7).
 *   3. boruvkaRound reuses its compBest buffer and resets only the
 *      entries it touched, instead of allocating O(N) every round.
 *      Every component's cheapest edge is now tried (previously a
 *      component merged earlier in the same loop was skipped).
 *   4. "--t=N" overrides the number of Borůvka rounds per super-step
 *      (e.g. --t=1 for the ablation).
 *   5. Correctness fix: if components remain after P super-steps (the
 *      early, partially-activated super-steps may shrink less than 2^t),
 *      extra super-steps with all edges activated run until no
 *      inter-component edge is left. Previously the result could be an
 *      incomplete spanning tree on a connected graph.
 *   6. Compiling with -DBMS_XI emits the ξ trace (JSON lines on stderr);
 *      run_bench.py builds bms_xi this way; bms_instrumented.cpp is retired
 *      (it skipped the final super-step, so it under-counted steps).
 *
 * Usage:  ./bms [--phase-a=select|radix] [--t=N] < graph.txt
 */

#include <bits/stdc++.h>
#include "bench_harness.h"
using namespace std;

// ════════════════════════════════════════════════════════════════
// Basic Types
// ════════════════════════════════════════════════════════════════
struct Edge { int u, v; long long w; };

struct CompEdge {
    int u, v;
    long long w;
    int orig_eidx;
};

static inline bool edgeLess(const CompEdge& A, const CompEdge& B) {
    if (A.w != B.w) return A.w < B.w;
    int au=min(A.u,A.v), av=max(A.u,A.v), bu=min(B.u,B.v), bv=max(B.u,B.v);
    if (au != bu) return au < bu;
    if (av != bv) return av < bv;
    return A.orig_eidx < B.orig_eidx;
}

// Order used by Phase A selection: weight, then (min(u,v), max(u,v)).
static inline bool origEdgeLess(const Edge& a, const Edge& b) {
    if (a.w != b.w) return a.w < b.w;
    int au = min(a.u, a.v), av = max(a.u, a.v);
    int bu = min(b.u, b.v), bv = max(b.u, b.v);
    if (au != bu) return au < bu;
    return av < bv;
}


// ════════════════════════════════════════════════════════════════
// LSD Radix Sort by weight (only used with --phase-a=radix)
// 4 passes of 16-bit digits over the 64-bit biased weight.
// Time Complexity: O(m), independent of the weight range.
// Stable; ties keep their input order. MST weight does not depend
// on how weight ties are broken.
// ════════════════════════════════════════════════════════════════
static void radixSortEdges(vector<Edge>& edges) {
    const size_t n = edges.size();
    if (n <= 1) return;

    // Bias to unsigned so ordering is preserved for negative weights too.
    vector<unsigned long long> key(n);
    for (size_t i = 0; i < n; i++) {
        key[i] = (unsigned long long)edges[i].w ^ 0x8000000000000000ULL;
    }

    vector<Edge> buf(n);
    vector<unsigned long long> keyBuf(n);
    constexpr int PASSES = 4;
    constexpr int BITS = 16;
    constexpr unsigned RADIX = 1u << BITS;

    vector<int> cnt(RADIX + 1);

    vector<Edge>* src = &edges;
    vector<Edge>* dst = &buf;
    vector<unsigned long long>* ksrc = &key;
    vector<unsigned long long>* kdst = &keyBuf;

    for (int pass = 0; pass < PASSES; pass++) {
        unsigned shift = pass * BITS;
        fill(cnt.begin(), cnt.end(), 0);
        for (size_t i = 0; i < n; i++) {
            unsigned digit = (unsigned)(((*ksrc)[i] >> shift) & (RADIX - 1));
            cnt[digit + 1]++;
        }
        for (unsigned d = 0; d < RADIX; d++) cnt[d + 1] += cnt[d];
        for (size_t i = 0; i < n; i++) {
            unsigned digit = (unsigned)(((*ksrc)[i] >> shift) & (RADIX - 1));
            int pos = cnt[digit]++;
            (*dst)[(size_t)pos] = (*src)[i];
            (*kdst)[(size_t)pos] = (*ksrc)[i];
        }
        swap(src, dst);
        swap(ksrc, kdst);
    }
    // PASSES is even, so src now points back to `edges` already in sorted order.
}

// ════════════════════════════════════════════════════════════════
// Disjoint Set Union (DSU)
// Optimized with path halving and union by rank
// Amortized time per operation: nearly O(α(n))
// ════════════════════════════════════════════════════════════════
struct DSU {
    vector<int> p, r;
    int comps = 0;

    void init(int n) {
        p.resize(n); r.assign(n, 0);
        iota(p.begin(), p.end(), 0);
        comps = n;
    }
    int find(int x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
        return x;
    }
    pair<int,int> unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return {-1, -1};
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        comps--;
        return {a, b};
    }
};

// ════════════════════════════════════════════════════════════════
// Graph Compressor  [Optimization 1 + Optimization 3]
//
// Optimization 1:
//   Eliminate sorting of compressed edges.
//   Since boruvkaRound performs a full scan and does not
//   rely on edge ordering, sorting is redundant.
//   Improves per-step cost from O(m_i log m_i) to O(m_i).
//
// Optimization 3:
//   Implement rootToNew as a reusable vector<int>.
//   Avoids reconstructing unordered_map<int,int> at each compression.
//   Achieves O(C_i) direct access with lower constant factors.
// ════════════════════════════════════════════════════════════════
struct GraphCompressor {
    vector<int> rootToNew;  // Maps oldRoot → new ID; unused entries are set to -1
    vector<int> usedRoots;  // Roots written during this compress call (for reset)

    struct Result {
        int n_nodes;
        vector<CompEdge> edges;
    };

    Result compress(DSU& uf, const vector<CompEdge>& curEdges) {
        const int oldSize = (int)uf.p.size();

        // Expand capacity (only grows, never shrinks; amortized cost O(1))
        if ((int)rootToNew.size() < oldSize) {
            rootToNew.assign(oldSize, -1);
        }
        usedRoots.clear();

        // Step 1: Enumerate active roots and build oldRoot → newID mapping
        //         (O(C_i) direct array writes)
        int newID = 0;
        for (int i = 0; i < oldSize; i++) {
            if (uf.find(i) == i) {
                rootToNew[i] = newID++;
                usedRoots.push_back(i);
            }
        }
        const int C = newID;

        // Step 2+3: Traverse current edges, remove internal edges,
        //           and keep the best edge for each pair (na, nb)
        unordered_map<uint64_t, CompEdge> best;
        best.reserve(curEdges.size() * 2);

        for (const auto& e : curEdges) {
            int ra = uf.find(e.u), rb = uf.find(e.v);
            if (ra == rb) continue;

            int na = rootToNew[ra], nb = rootToNew[rb];
            if (na > nb) swap(na, nb);

            uint64_t key = ((uint64_t)na << 32) | (uint32_t)nb;
            CompEdge ce{na, nb, e.w, e.orig_eidx};
            auto it = best.find(key);
            if (it == best.end() || edgeLess(ce, it->second)) {
                best[key] = ce;
            }
        }

        // Step 4: Collect compressed edges
        Result res;
        res.n_nodes = C;
        res.edges.reserve(best.size());
        for (auto& [k, e] : best) res.edges.push_back(e);
        return res;
    }

    // Must be called after origToComp update.
    // Resets all entries written during this compression to -1.
    // Time complexity: O(C_i).
    void resetRootToNew() {
        for (int r : usedRoots) rootToNew[r] = -1;
    }
};

// ════════════════════════════════════════════════════════════════
// One Borůvka round within a super-step
// Does not require sorted input; performs a full edge scan
// Time complexity: O(|edges|)
//
// compBest is kept across rounds (grow-only) and only the entries
// touched in this round are reset, so no O(N) allocation per round.
// ════════════════════════════════════════════════════════════════
struct BoruvkaWorkspace {
    vector<int> compBest;   // root → index of its cheapest edge, -1 if none
    vector<int> touched;    // roots written during this round (for reset)
};

static vector<CompEdge> boruvkaRound(
    const vector<CompEdge>& edges,
    DSU& uf,
    BoruvkaWorkspace& ws)
{
    const int N = (int)uf.p.size();
    if ((int)ws.compBest.size() < N) ws.compBest.resize(N, -1);
    vector<int>& compBest = ws.compBest;
    ws.touched.clear();

    for (int i = 0; i < (int)edges.size(); i++) {
        const auto& e = edges[i];
        int ca = uf.find(e.u), cb = uf.find(e.v);
        if (ca == cb) continue;
        if (compBest[ca] < 0) { ws.touched.push_back(ca); compBest[ca] = i; }
        else if (edgeLess(e, edges[compBest[ca]])) compBest[ca] = i;
        if (compBest[cb] < 0) { ws.touched.push_back(cb); compBest[cb] = i; }
        else if (edgeLess(e, edges[compBest[cb]])) compBest[cb] = i;
    }

    // Try the cheapest edge of every component that had one at scan time.
    // All of them are safe (cut property under the consistent edgeLess order),
    // and uf.unite rejects the ones that would close a cycle.
    vector<CompEdge> selected;
    for (int c : ws.touched) {
        const auto& e = edges[compBest[c]];
        compBest[c] = -1;
        auto [nr, dr] = uf.unite(e.u, e.v);
        if (nr != -1) selected.push_back(e);
    }
    return selected;
}

// ════════════════════════════════════════════════════════════════
// Parameters
// ════════════════════════════════════════════════════════════════
static int computeT(int n) {
    if (n <= 4) return 1;
    return max(1, (int)ceil(pow(log2((double)n), 2.0/3.0)));
}

static int computeP(int n, int t) {
    if (n <= 1) return 0;
    return max(1, (int)ceil(log2((double)n) / t));
}

enum class PhaseAMode { Select, Radix };

// ════════════════════════════════════════════════════════════════
// BMSBoruvka
// ════════════════════════════════════════════════════════════════
struct BMSBoruvka {
    int n_orig;
    vector<Edge> edges;
    PhaseAMode phaseA;

    long long mst_weight = 0;
    vector<int> mst_eidx;

    int t_rounds;
    int P_steps;

#ifdef BMS_XI
    vector<double> xi_per_step;
    vector<int> C_trace;   // component count at start of each super-step
    vector<int> M_trace;   // curEdges size entering Phase B each super-step
#endif

    explicit BMSBoruvka(int n_, vector<Edge> E,
                        PhaseAMode mode = PhaseAMode::Select, int t_override = 0)
        : n_orig(n_), edges(std::move(E)), phaseA(mode)
    {
        t_rounds = t_override > 0 ? t_override : computeT(n_orig);
        P_steps  = computeP(n_orig, t_rounds);
    }

    void run() {
        mst_weight = 0;
        mst_eidx.clear();
        mst_eidx.reserve(max(0, n_orig - 1));

        if (n_orig <= 1 || edges.empty()) return;

        const int m = (int)edges.size();

        if (phaseA == PhaseAMode::Radix) radixSortEdges(edges);

        DSU uf;
        uf.init(n_orig);

        vector<CompEdge> curEdges;
        curEdges.reserve(m);

        int curActivated = 0;
        const int edgesPerStep = max(1, m / max(1, P_steps));

        vector<int> origToComp(n_orig);
        iota(origToComp.begin(), origToComp.end(), 0);

        GraphCompressor compressor;
        BoruvkaWorkspace ws;

        // Super-steps 0 … P-1 as in the paper; the last one activates all
        // remaining edges. If components with outgoing edges remain after
        // that, further super-steps run (all edges already active) until
        // curEdges is empty.
        for (int step = 0; uf.comps > 1; step++) {
#ifdef BMS_XI
            const int C_before = uf.comps;
            C_trace.push_back(C_before);
#endif

            // Phase A: Activate the next k lightest edges
            // (weight-prefix activation preserves the cut property)
            const int activateEnd = (step >= P_steps - 1)
                ? m
                : min(m, curActivated + edgesPerStep);

            if (phaseA == PhaseAMode::Select && activateEnd < m && activateEnd > curActivated) {
                // After this call edges[curActivated, activateEnd) are the
                // (activateEnd - curActivated) lightest unactivated edges,
                // and no edge in [activateEnd, m) is lighter than any of them.
                nth_element(edges.begin() + curActivated,
                            edges.begin() + activateEnd,
                            edges.end(), origEdgeLess);
            }

            for (int i = curActivated; i < activateEnd; i++) {
                const auto& e = edges[i];
                // Direct array indexing (Optimization 2);
                // Component merges within the super-step are resolved via uf.find.
                int cu = uf.find(origToComp[e.u]);
                int cv = uf.find(origToComp[e.v]);
                if (cu == cv) continue;
                curEdges.push_back({cu, cv, e.w, i});
            }
            curActivated = activateEnd;

#ifdef BMS_XI
            M_trace.push_back((int)curEdges.size());
#endif

            if (curEdges.empty()) {
#ifdef BMS_XI
                xi_per_step.push_back(1.0);
#endif
                if (curActivated == m) break;   // nothing left to merge (MSF)
                continue;
            }

            // Phase B: Execute t Borůvka rounds
            // Each round runs in O(m_i) time
            for (int round = 0; round < t_rounds && uf.comps > 1; round++) {
                // Filter internalized edges and update endpoints in-place
                {
                    vector<CompEdge> active;
                    active.reserve(curEdges.size());
                    for (auto& e : curEdges) {
                        int ra = uf.find(e.u), rb = uf.find(e.v);
                        if (ra != rb) { e.u = ra; e.v = rb; active.push_back(e); }
                    }
                    curEdges = std::move(active);
                }
                if (curEdges.empty()) break;

                vector<CompEdge> selected = boruvkaRound(curEdges, uf, ws);
                for (const auto& e : selected) {
                    mst_eidx.push_back(e.orig_eidx);
                    mst_weight += e.w;
                }

                if ((int)mst_eidx.size() == n_orig - 1) break;
            }

#ifdef BMS_XI
            xi_per_step.push_back((double)C_before / max(1, uf.comps));
#endif

            if (uf.comps <= 1) break;
            if ((int)mst_eidx.size() == n_orig - 1) break;

            // Phase C: Graph compression
            {
                // Remove remaining intra-component edges from curEdges
                {
                    vector<CompEdge> active;
                    active.reserve(curEdges.size());
                    for (auto& e : curEdges) {
                        int ra = uf.find(e.u), rb = uf.find(e.v);
                        if (ra != rb) { e.u = ra; e.v = rb; active.push_back(e); }
                    }
                    curEdges = std::move(active);
                }

                if (curEdges.empty()) {
                    if (curActivated == m) break;   // nothing left to merge (MSF)
                    continue;
                }

                // compress O(m_i)
                auto res = compressor.compress(uf, curEdges);

                // rootToNew remains valid until resetRootToNew() is called
                for (int origNode = 0; origNode < n_orig; origNode++) {
                    const int oldComp = origToComp[origNode];
                    const int oldRoot = uf.find(oldComp);
                    const int newID   = compressor.rootToNew[oldRoot];
                    if (newID >= 0) origToComp[origNode] = newID;
                }

                // Reset rootToNew only after origToComp has been updated (O(C_i)).
                // This must be done here rather than inside compress(),
                // otherwise rootToNew would be cleared before the update completes.
                compressor.resetRootToNew();

                // Rebuild the DSU in the compressed ID space
                uf.init(res.n_nodes);

                // Update edge set (unordered; does not affect boruvkaRound)
                curEdges = std::move(res.edges);
            }
        }

        mst_weight = 0;
        for (int eidx : mst_eidx) mst_weight += edges[eidx].w;
    }
};

// ════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════
int main(int argc, char** argv) {
    // Also sets up fast I/O and parses --quiet / --repeat.
    bench::init(argc, argv);

    PhaseAMode mode = PhaseAMode::Select;
    int t_override = 0;
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        if (a == "--phase-a=select")      mode = PhaseAMode::Select;
        else if (a == "--phase-a=radix")  mode = PhaseAMode::Radix;
        else if (a.rfind("--t=", 0) == 0) t_override = max(1, atoi(a.c_str() + 4));
        // Consumed by bench::init above; skipped here so the usage check below
        // doesn't reject them.
        else if (a == "--quiet")          continue;
        else if (a == "--repeat")         { i++; continue; }
        else {
            cerr << "Usage: " << argv[0]
                 << " [--phase-a=select|radix] [--t=N] [--quiet] [--repeat K] < graph\n";
            return 1;
        }
    }

    int n, m;
    cin >> n >> m;

    vector<Edge> E;
    E.reserve(m);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        E.push_back({u, v, w});
    }

    // The solver takes ownership of the edge vector and mutates it, so each
    // repeat gets a fresh copy (made outside the timed region). The constructor
    // is inside the timed region: it does the initial setup, which is part of
    // the algorithm's cost.
    std::unique_ptr<BMSBoruvka> solver;
    for (int r = 0; r < bench::repeats(); r++) {
        vector<Edge> work = E;
        double t0 = bench::now_ms();
        solver = std::make_unique<BMSBoruvka>(n, std::move(work), mode, t_override);
        solver->run();
        bench::report_time(bench::now_ms() - t0);
    }

#ifdef BMS_XI
    // ξ trace on stderr (JSON lines), same format as the old bms_instrumented.cpp.
    // Emitted once, for the final repeat: the trace is deterministic given the
    // input, so repeats would only duplicate it.
    for (size_t i = 0; i < solver->xi_per_step.size(); i++) {
        fprintf(stderr, "{\"type\":\"step\",\"step\":%d,\"C_before\":%d,\"M_i\":%d,"
                "\"t\":%d,\"xi\":%.4f}\n", (int)i, solver->C_trace[i],
                solver->M_trace[i], solver->t_rounds, solver->xi_per_step[i]);
    }
    double log_sum = 0; int cnt = 0;
    for (double x : solver->xi_per_step) if (x > 1.0) { log_sum += log(x); cnt++; }
    double xi_geomean = cnt > 0 ? exp(log_sum / cnt) : 1.0;
    fprintf(stderr, "{\"type\":\"summary\",\"n\":%d,\"m\":%d,\"P\":%d,\"t\":%d,"
            "\"steps_actual\":%d,\"xi_geomean\":%.4f,\"xi_theory\":%.4f,\"C_trace\":[",
            n, m, solver->P_steps, solver->t_rounds, (int)solver->xi_per_step.size(),
            xi_geomean, pow(2.0, solver->t_rounds));
    for (size_t i = 0; i < solver->C_trace.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", solver->C_trace[i]);
    fprintf(stderr, "],\"M_trace\":[");
    for (size_t i = 0; i < solver->M_trace.size(); i++)
        fprintf(stderr, "%s%d", i ? "," : "", solver->M_trace[i]);
    fprintf(stderr, "]}\n");
#endif

    cout << "Total weight = " << solver->mst_weight << "\n";
    if (bench::print_edges()) {
        cout << "MST/MSF edges (" << solver->mst_eidx.size() << "):\n";
        for (int eidx : solver->mst_eidx) {
            const auto& e = solver->edges[eidx];
            cout << e.u << " " << e.v << " " << e.w << "\n";
        }
    }

    return 0;
}
