/*
 * kruskal.cpp  —  Standard Kruskal's MST algorithm
 *
 * Sorts all edges by weight with std::sort, then greedily adds edges
 * using a DSU. Complexity: O(m log m) = O(m log n).
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

struct DSU {
    vector<int> p, r;
    DSU(int n) : p(n), r(n, 0) { iota(p.begin(), p.end(), 0); }
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
        return true;
    }
};

int main() {
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n, m; cin >> n >> m;
    vector<Edge> E(m);
    for (auto& e : E) cin >> e.u >> e.v >> e.w;

    // Sort by weight; break ties deterministically
    sort(E.begin(), E.end(), [](const Edge& a, const Edge& b) {
        if (a.w != b.w) return a.w < b.w;
        int au = min(a.u,a.v), av = max(a.u,a.v);
        int bu = min(b.u,b.v), bv = max(b.u,b.v);
        return au != bu ? au < bu : av < bv;
    });

    DSU dsu(n);
    LL total = 0;
    vector<Edge> mst;
    mst.reserve(n - 1);

    for (const auto& e : E) {
        if (dsu.unite(e.u, e.v)) {
            mst.push_back(e);
            total += e.w;
            if ((int)mst.size() == n - 1) break;
        }
    }

    cout << "Total weight = " << total << "\n";
    cout << "Edges in MST/MSF (" << mst.size() << "):\n";
    for (const auto& e : mst)
        cout << e.u << " " << e.v << " " << e.w << "\n";
    return 0;
}
