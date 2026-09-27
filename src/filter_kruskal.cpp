/*
 * filter_kruskal.cpp  —  Filter-Kruskal MST algorithm
 *
 * Osipov, Sanders & Singler, ALENEX 2009 [OSS09].
 *
 * Key idea: instead of sorting all m edges globally, partition around a pivot
 * weight (chosen via nth_element at the median) and filter out any edge that
 * would form a cycle with the current spanning forest *before* recursing on the
 * lighter half. For random edge weights this runs in O(m + n log n log(m/n)),
 * which is O(n log n) when m = O(n)  —  linear for non-sparse graphs.
 *
 * Implementation notes:
 *   - Uses DSU with union-by-rank + path halving (same as other benchmarks).
 *   - Falls back to plain sort + greedy when the edge set is small enough
 *     (threshold BASE_THRESHOLD) to avoid recursion overhead.
 *   - The filter step calls dsu.find() on both endpoints; edges where both
 *     endpoints are already in the same component are discarded.
 *   - Pivot is selected via nth_element at the median position, matching the
 *     paper's description of choosing a "random" pivot in expected O(m) time.
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

struct Edge { int u, v; LL w; };

// ── DSU ───────────────────────────────────────────────────────────────────────
struct DSU {
    vector<int> p, r;
    int comps;
    DSU(int n) : p(n), r(n, 0), comps(n) { iota(p.begin(), p.end(), 0); }
    int find(int x) {
        while (p[x] != x) { p[x] = p[p[x]]; x = p[x]; }
        return x;
    }
    bool unite(int a, int b) {
        a = find(a); b = find(b);
        if (a == b) return false;
        if (r[a] < r[b]) swap(a, b);
        p[b] = a;
        if (r[a] == r[b]) r[a]++;
        comps--;
        return true;
    }
    bool same(int a, int b) { return find(a) == find(b); }
};

// ── Globals ───────────────────────────────────────────────────────────────────
static const int BASE_THRESHOLD = 1024; // switch to plain sort below this size
static LL   g_total = 0;
static vector<Edge> g_mst;

// ── Filter: remove edges whose endpoints are already connected ────────────────
static void filter(vector<Edge>& E, DSU& dsu) {
    int w = 0;
    for (int i = 0; i < (int)E.size(); i++)
        if (!dsu.same(E[i].u, E[i].v))
            E[w++] = E[i];
    E.resize(w);
}

// ── Base case: sort + greedy ──────────────────────────────────────────────────
static void kruskal_base(vector<Edge>& E, DSU& dsu) {
    sort(E.begin(), E.end(), [](const Edge& a, const Edge& b) {
        if (a.w != b.w) return a.w < b.w;
        int au=min(a.u,a.v), av=max(a.u,a.v), bu=min(b.u,b.v), bv=max(b.u,b.v);
        return au!=bu ? au<bu : av<bv;
    });
    for (const auto& e : E) {
        if (dsu.unite(e.u, e.v)) {
            g_mst.push_back(e);
            g_total += e.w;
        }
    }
}

// ── Filter-Kruskal recursive step ─────────────────────────────────────────────
static void filter_kruskal(vector<Edge>& E, DSU& dsu) {
    if (dsu.comps == 1 || E.empty()) return;

    // Base case: small enough to sort directly
    if ((int)E.size() <= BASE_THRESHOLD) {
        filter(E, dsu);
        kruskal_base(E, dsu);
        return;
    }

    // Choose pivot = median weight via nth_element
    size_t mid = E.size() / 2;
    nth_element(E.begin(), E.begin() + (ptrdiff_t)mid, E.end(),
                [](const Edge& a, const Edge& b) { return a.w < b.w; });
    LL pivot = E[mid].w;

    // Partition: light (<= pivot) and heavy (> pivot)
    vector<Edge> light, heavy;
    light.reserve(mid + 1);
    heavy.reserve(E.size() - mid);
    for (auto& e : E) {
        if (e.w <= pivot) light.push_back(e);
        else              heavy.push_back(e);
    }
    E.clear(); E.shrink_to_fit();  // free memory

    // Recurse on light half
    filter_kruskal(light, dsu);
    light.clear(); light.shrink_to_fit();

    // Filter heavy half then recurse
    if (dsu.comps > 1) {
        filter(heavy, dsu);
        filter_kruskal(heavy, dsu);
    }
}

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);

    int n, m; cin >> n >> m;
    vector<Edge> E(m);
    for (auto& e : E) cin >> e.u >> e.v >> e.w;

    DSU dsu(n);
    g_mst.reserve(n - 1);
    g_total = 0;

    filter_kruskal(E, dsu);

    cout << "Total weight = " << g_total << "\n";
    cout << "Edges in MST/MSF (" << g_mst.size() << "):\n";
    for (const auto& e : g_mst)
        cout << e.u << " " << e.v << " " << e.w << "\n";
    return 0;
}
