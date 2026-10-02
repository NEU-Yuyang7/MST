/*
 * BMSBoruvka.cpp
 *
 * ── Complexity Summary (Sparse Graph Case: m = O(n)) ───────────────
 *
 *   Parameters:
 *       t = ⌈log^{2/3}(n)⌉
 *       P = ⌈log(n) / t⌉ = O(log^{1/3}(n))
 *
 *   Component shrinkage:
 *       C_0 = n
 *       C_{i+1} ≤ C_i / 2^t
 *       (Each super-step performs t Borůvka rounds,
 *        and each round at least halves the number of components.)
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
 */

#include <bits/stdc++.h>
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


// ════════════════════════════════════════════════════════════════
// Integer Counting Sort
// Precondition: Edge weights are integers (long long) and
//               the range R = maxW - minW + 1 is reasonably bounded.
// Time Complexity: O(m + R).
// If R exceeds a predefined threshold, the implementation
// falls back to std::sort to avoid excessive memory usage.
// ════════════════════════════════════════════════════════════════
// Drop-in replacement: true LSD radix sort by weight, O(m) regardless of
// weight range (4 passes of 16-bit digits over the 64-bit biased weight).
// No tie-break on (u,v): MST total weight is invariant to how weight-ties
// are broken, so this is not required for correctness, only for byte-identical
// determinism with the original comparator (which this benchmark does not need).
static void countingSortEdges(std::vector<Edge>& edges) {
    const size_t n = edges.size();
    if (n <= 1) return;

    // Bias to unsigned so ordering is preserved for negative weights too.
    std::vector<unsigned long long> key(n);
    for (size_t i = 0; i < n; i++) {
        key[i] = (unsigned long long)edges[i].w ^ 0x8000000000000000ULL;
    }

    std::vector<Edge> buf(n);
    std::vector<unsigned long long> keyBuf(n);
    constexpr int PASSES = 4;
    constexpr int BITS = 16;
    constexpr unsigned RADIX = 1u << BITS;

    std::vector<int> cnt(RADIX + 1);

    std::vector<Edge>* src = &edges;
    std::vector<Edge>* dst = &buf;
    std::vector<unsigned long long>* ksrc = &key;
    std::vector<unsigned long long>* kdst = &keyBuf;

    for (int pass = 0; pass < PASSES; pass++) {
        unsigned shift = pass * BITS;
        std::fill(cnt.begin(), cnt.end(), 0);
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
        std::swap(src, dst);
        std::swap(ksrc, kdst);
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
// Time complexity: O(|edges| + C_i)
// ════════════════════════════════════════════════════════════════
static vector<CompEdge> boruvkaRound(
    const vector<CompEdge>& edges,
    DSU& uf)
{
    const int N = (int)uf.p.size();
    vector<int> compBest(N, -1);

    for (int i = 0; i < (int)edges.size(); i++) {
        const auto& e = edges[i];
        int ca = uf.find(e.u), cb = uf.find(e.v);
        if (ca == cb) continue;
        if (compBest[ca] < 0 || edgeLess(e, edges[compBest[ca]])) compBest[ca] = i;
        if (compBest[cb] < 0 || edgeLess(e, edges[compBest[cb]])) compBest[cb] = i;
    }

    vector<CompEdge> selected;
    for (int c = 0; c < N; c++) {
        if (uf.find(c) != c) continue;
        const int bi = compBest[c];
        if (bi < 0) continue;
        const auto& e = edges[bi];
        int ra = uf.find(e.u), rb = uf.find(e.v);
        if (ra == rb) continue;
        auto [nr, dr] = uf.unite(ra, rb);
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

// ════════════════════════════════════════════════════════════════
// BMSBoruvka
// ════════════════════════════════════════════════════════════════
struct BMSBoruvka {
    int n_orig;
    vector<Edge> edges;

    long long mst_weight = 0;
    vector<int> mst_eidx;

    int t_rounds;
    int P_steps;

    explicit BMSBoruvka(int n_, vector<Edge> E)
        : n_orig(n_), edges(std::move(E))
    {
        countingSortEdges(edges);
        t_rounds = computeT(n_orig);
        P_steps  = computeP(n_orig, t_rounds);
    }

    void run() {
        mst_weight = 0;
        mst_eidx.clear();
        mst_eidx.reserve(n_orig - 1);

        if (n_orig <= 1 || edges.empty()) return;

        const int m = (int)edges.size();

        DSU uf;
        uf.init(n_orig);

        vector<CompEdge> curEdges;
        curEdges.reserve(m);

        int curActivated = 0;
        const int edgesPerStep = max(1, m / max(1, P_steps));

        vector<int> origToComp(n_orig);
        iota(origToComp.begin(), origToComp.end(), 0);

        GraphCompressor compressor;

        for (int step = 0; step < P_steps && uf.comps > 1; step++) {

            // Phase A: Activate edges in nondecreasing weight order
            // (prefix-based activation to preserve the cut property)
            const int activateEnd = (step == P_steps - 1)
                ? m
                : min(m, curActivated + edgesPerStep);

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

            if (curEdges.empty()) continue;

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

                vector<CompEdge> selected = boruvkaRound(curEdges, uf);
                for (const auto& e : selected) {
                    mst_eidx.push_back(e.orig_eidx);
                    mst_weight += e.w;
                }

                if ((int)mst_eidx.size() == n_orig - 1) goto done;
            }

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

        done:;

        mst_weight = 0;
        for (int eidx : mst_eidx) mst_weight += edges[eidx].w;
    }
};

// ════════════════════════════════════════════════════════════════
// main
// ════════════════════════════════════════════════════════════════
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    cin >> n >> m;

    vector<Edge> E;
    E.reserve(m);
    for (int i = 0; i < m; i++) {
        int u, v; long long w;
        cin >> u >> v >> w;
        E.push_back({u, v, w});
    }

    BMSBoruvka solver(n, std::move(E));
    solver.run();

    cout << "Total weight = " << solver.mst_weight << "\n";
    cout << "MST/MSF edges (" << solver.mst_eidx.size() << "):\n";
    for (int eidx : solver.mst_eidx) {
        const auto& e = solver.edges[eidx];
        cout << e.u << " " << e.v << " " << e.w << "\n";
    }

    return 0;
}
