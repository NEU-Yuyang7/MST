/*
 * BMSBoruvka.cpp  —  BMS MST algorithm
 *
 * Implements the three-phase super-step structure described in the paper:
 *
 *   Phase A  (nth_element activation):
 *     Each super-step i partitions the unactivated suffix of edges[] around
 *     its (i·k)-th smallest edge via nth_element in O(remaining edges).
 *     The lightest k = floor(m / P) edges from that suffix become newly active.
 *     All super-steps together cost O(m + m/1 + m/2 + … + m/P) = O(m log^{1/3} n).
 *     No global sort is ever performed.
 *
 *   Phase B  (t Borůvka rounds):
 *     Run t standard Borůvka rounds on the currently active edge set.
 *     Each round finds the cheapest outgoing edge per component and unions.
 *     t = ceil((log2 n)^{2/3}) rounds reduce components by factor 2^t per step.
 *
 *   Phase C  (graph compression):
 *     Compress the active edge set: re-label endpoints to current component
 *     roots, discard intra-component edges, keep only the cheapest edge per
 *     (root_u, root_v) pair.  This bounds |active| going into the next step.
 *
 * Parameters (matching the paper):
 *   t = ceil((log2 n)^{2/3})
 *   P = ceil(log2(n) / t)   =  Theta(log^{1/3} n)  super-steps
 *   k = floor(m / P)        edges activated per super-step
 *
 * Complexity:
 *   Phase A aggregate:  O(m log^{1/3} n)  [nth_element per super-step, no global sort]
 *   Phase B aggregate:  O(t · Σ m_i) = O(n log^{2/3} n)  for sparse m=O(n)
 *   Phase C aggregate:  O(Σ m_i + P·n) = O(n log^{1/3} n)
 *   Total (sparse m=O(n)):  O(n log^{2/3} n)
 *
 * Fixes applied (see BMSBoruvka_Benchmark_Findings.docx):
 *
 *   [Finding 1]  Phase A now uses per-super-step nth_element on the
 *                unactivated suffix — no upfront global sort anywhere.
 *
 *   [Finding 3/5] boruvkaRound: replaced set<int> dedup (O(C log C)) and
 *                 per-call vector<int>(N,-1) allocation with a reusable
 *                 compBest vector that is selectively reset in O(C_i) via
 *                 a "touched" list — same pattern as GraphCompressor.
 *                 unite() return value used directly for dedup (no set).
 *
 *   [Phase C origToComp] Replaced set<int>+map<int,int> remap (O(C log C))
 *                 with GraphCompressor: a reusable vector<int> rootToNew
 *                 built/reset in O(C_i) per step, shared between compress
 *                 and the origToComp update pass.
 *
 * Input (stdin):  n m
 *                 u v w   (0-indexed, repeated m times)
 * Output (stdout): Total weight = <W>
 *                  Edges in MST/MSF (<k>):
 *                  u v w  ...
 */
#include <bits/stdc++.h>
using namespace std;
using LL = long long;

// ── Edge ──────────────────────────────────────────────────────────────────────
struct Edge { int u, v; LL w; };

// CompEdge: an edge whose endpoints have been mapped to component roots.
struct CompEdge {
    int u, v;   // component roots (in current compressed ID space)
    LL  w;
    int orig;   // index into the original edge array (for MST reconstruction)
};

// ── DSU ───────────────────────────────────────────────────────────────────────
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
    // Returns {newRoot, droppedRoot} on success, {-1,-1} if already same component.
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

// ── Comparator ────────────────────────────────────────────────────────────────
static inline bool edgeLess(const CompEdge& a, const CompEdge& b) {
    if (a.w != b.w) return a.w < b.w;
    return a.orig < b.orig;   // deterministic tie-break by original index
}
static inline bool rawWeightLess(const Edge& a, const Edge& b) { return a.w < b.w; }

// ── Parameter calculation ─────────────────────────────────────────────────────
static int computeT(int n) {
    // t = ceil((log2 n)^{2/3}),  minimum 1
    if (n <= 4) return 1;
    double lg = log2((double)n);
    return max(1, (int)ceil(pow(lg, 2.0 / 3.0)));
}
static int computeP(int n, int t) {
    // P = ceil(log2(n) / t),  minimum 1
    if (n <= 1) return 0;
    double lg = log2((double)n);
    return max(1, (int)ceil(lg / t));
}

// ── Graph Compressor ──────────────────────────────────────────────────────────
// Fix [Phase C origToComp]: reusable rootToNew vector replaces set+map per step.
// rootToNew[oldRoot] = newID during a compress call; reset to -1 in O(C_i).
// Also used by the main loop to update origToComp without rebuilding the mapping.
struct GraphCompressor {
    vector<int> rootToNew;  // oldRoot → newID; -1 for unused entries
    vector<int> usedRoots;  // entries written this call (for O(C_i) reset)

    struct Result {
        int newN;
        vector<CompEdge> edges;
    };

    // Build the oldRoot→newID mapping and compress curEdges.
    // rootToNew remains valid (not reset) until resetRootToNew() is called,
    // so the caller can use it for origToComp before calling reset.
    Result compress(DSU& uf, const vector<CompEdge>& curEdges) {
        const int oldSz = (int)uf.p.size();

        // Grow-only: never shrink; amortised O(1) per step
        if ((int)rootToNew.size() < oldSz)
            rootToNew.assign(oldSz, -1);
        usedRoots.clear();

        // Step 1: enumerate roots, assign compact IDs  O(oldSz)
        int newID = 0;
        for (int i = 0; i < oldSz; i++) {
            if (uf.find(i) == i) {
                rootToNew[i] = newID++;
                usedRoots.push_back(i);
            }
        }

        // Step 2: build compressed edge set — keep best edge per (na,nb) pair
        unordered_map<uint64_t, CompEdge> best;
        best.reserve(curEdges.size() * 2);
        for (const auto& e : curEdges) {
            int ra = uf.find(e.u), rb = uf.find(e.v);
            if (ra == rb) continue;
            int na = rootToNew[ra], nb = rootToNew[rb];
            if (na > nb) swap(na, nb);
            uint64_t key = ((uint64_t)na << 32) | (uint32_t)nb;
            CompEdge ce{na, nb, e.w, e.orig};
            auto it = best.find(key);
            if (it == best.end() || edgeLess(ce, it->second))
                best[key] = ce;
        }

        Result res;
        res.newN = newID;
        res.edges.reserve(best.size());
        for (auto& [k, ce] : best) res.edges.push_back(ce);
        return res;
    }

    // Reset written entries to -1.  Must be called after origToComp is updated.
    void resetRootToNew() {
        for (int r : usedRoots) rootToNew[r] = -1;
    }
};

// ── One Borůvka round on a CompEdge array ─────────────────────────────────────
// Fix [Finding 3/5]:
//   - compBest is passed in as a reusable vector (allocated once, size = max N).
//     Only touched entries are reset via the 'touched' list — O(C_i) instead of O(N).
//   - Dedup is done via unite() return value: only edges that cause a successful
//     union are added to MST — no set<int> needed.
//
// Returns selected MST edges.
static vector<CompEdge> boruvkaRound(
    const vector<CompEdge>& edges,
    DSU& uf,
    vector<int>& compBest,   // reusable; size >= uf.p.size(), all -1 on entry
    vector<int>& touched)    // reusable scratch list; empty on entry
{
    // Find best outgoing edge per component root
    for (int i = 0; i < (int)edges.size(); i++) {
        int ra = uf.find(edges[i].u);
        int rb = uf.find(edges[i].v);
        if (ra == rb) continue;
        auto upd = [&](int r) {
            if (compBest[r] < 0) { compBest[r] = i; touched.push_back(r); }
            else if (edgeLess(edges[i], edges[compBest[r]])) compBest[r] = i;
        };
        upd(ra); upd(rb);
    }

    // For each root that has a candidate, attempt union.
    // unite() is idempotent-safe: returns {-1,-1} if already merged this round.
    vector<CompEdge> chosen;
    for (int r : touched) {
        const int bi = compBest[r];
        if (bi < 0) continue;
        const auto& e = edges[bi];
        int ra = uf.find(e.u), rb = uf.find(e.v);
        if (ra == rb) continue;
        auto [nr, dr] = uf.unite(ra, rb);
        if (nr != -1) chosen.push_back(e);
    }

    // Reset only touched entries — O(C_i), not O(N)
    for (int r : touched) compBest[r] = -1;
    touched.clear();

    return chosen;
}

// ── Main algorithm ─────────────────────────────────────────────────────────────
pair<LL, vector<Edge>> bmsMST(int n, vector<Edge>& edges) {
    const int m = (int)edges.size();

    const int t = computeT(n);
    const int P = computeP(n, t);
    const int k = max(1, m / max(1, P));   // edges activated per super-step

    DSU uf;
    uf.init(n);

    LL totalWeight = 0;
    vector<int> mstOrig;
    mstOrig.reserve(n - 1);

    // origToComp[i] = current compact ID for original vertex i.
    // Updated in O(n) after each Phase C.
    vector<int> origToComp(n);
    iota(origToComp.begin(), origToComp.end(), 0);

    vector<CompEdge> curEdges;
    curEdges.reserve(m);

    int activated = 0;

    // Reusable scratch for boruvkaRound  (Fix Finding 5)
    vector<int> compBest(n, -1);   // grows only; entries always -1 between rounds
    vector<int> touched;
    touched.reserve(n);

    GraphCompressor compressor;

    for (int step = 0; step < P && uf.comps > 1; step++) {

        // ── Phase A: nth_element activation ──────────────────────────────────
        // Bring the k lightest unactivated edges to positions [activated, activateEnd).
        // Cost: O(m - activated) per step; aggregate O(m log^{1/3} n). No global sort.
        const int activateEnd = (step == P - 1) ? m : min(m, activated + k);

        if (activateEnd > activated) {
            nth_element(
                edges.begin() + activated,
                edges.begin() + activateEnd - 1,
                edges.end(),
                rawWeightLess
            );
        }

        // Map newly activated edges into the current compressed vertex space.
        for (int i = activated; i < activateEnd; i++) {
            int cu = uf.find(origToComp[edges[i].u]);
            int cv = uf.find(origToComp[edges[i].v]);
            if (cu != cv)
                curEdges.push_back({cu, cv, edges[i].w, i});
        }
        activated = activateEnd;

        if (curEdges.empty()) continue;

        // ── Phase B: t Borůvka rounds ─────────────────────────────────────────
        for (int round = 0; round < t && uf.comps > 1; round++) {
            // Refresh endpoints: previous round's unions may have merged roots.
            {
                vector<CompEdge> live;
                live.reserve(curEdges.size());
                for (auto& e : curEdges) {
                    e.u = uf.find(e.u);
                    e.v = uf.find(e.v);
                    if (e.u != e.v) live.push_back(e);
                }
                curEdges = move(live);
            }
            if (curEdges.empty()) break;

            // Grow compBest if the compressed vertex space shrank (it can only shrink)
            // — actually compBest was sized to n at construction; always sufficient.
            vector<CompEdge> chosen = boruvkaRound(curEdges, uf, compBest, touched);
            for (const auto& e : chosen) {
                mstOrig.push_back(e.orig);
                totalWeight += e.w;
            }
            if ((int)mstOrig.size() == n - 1) goto done;
        }

        if (uf.comps <= 1) break;

        // ── Phase C: graph compression ────────────────────────────────────────
        // 1. Flush stale intra-component edges from curEdges.
        {
            vector<CompEdge> live;
            live.reserve(curEdges.size());
            for (auto& e : curEdges) {
                e.u = uf.find(e.u);
                e.v = uf.find(e.v);
                if (e.u != e.v) live.push_back(e);
            }
            curEdges = move(live);
        }
        if (curEdges.empty()) continue;

        // 2. Compress: build oldRoot→newID mapping, deduplicate edges.
        {
            auto res = compressor.compress(uf, curEdges);

            // 3. Update origToComp using the mapping still live in compressor.
            //    O(n) — one array lookup per original vertex.
            for (int i = 0; i < n; i++) {
                const int oldRoot = uf.find(origToComp[i]);
                origToComp[i] = compressor.rootToNew[oldRoot];
            }

            // 4. Reset the mapping (O(C_i)); must follow origToComp update.
            compressor.resetRootToNew();

            // 5. Rebuild DSU on the smaller vertex set.
            uf.init(res.newN);

            // 6. Swap in the compressed edge set.
            curEdges = move(res.edges);
        }
    }

done:
    vector<Edge> mst;
    mst.reserve(mstOrig.size());
    for (int idx : mstOrig) mst.push_back(edges[idx]);
    return {totalWeight, mst};
}

// ── main ──────────────────────────────────────────────────────────────────────
int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    int n, m; cin >> n >> m;
    vector<Edge> edges(m);
    for (auto& e : edges) cin >> e.u >> e.v >> e.w;

    auto [total, mst] = bmsMST(n, edges);

    cout << "Total weight = " << total << "\n";
    cout << "Edges in MST/MSF (" << mst.size() << "):\n";
    for (auto& e : mst)
        cout << e.u << " " << e.v << " " << e.w << "\n";
    return 0;
}
