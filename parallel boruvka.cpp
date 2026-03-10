#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

using LL = long long;

struct Edge {
    int u, v;
    LL w;
};

struct DSU {
    vector<int> parent, rnk;
    int comps;

    DSU(int n = 0) { init(n); }

    void init(int n) {
        parent.resize(n);
        rnk.assign(n, 0);
        iota(parent.begin(), parent.end(), 0);
        comps = n;
    }

    int find(int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // path compression by halving
            x = parent[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if (a == b) return false;
        if (rnk[a] < rnk[b]) swap(a, b);
        parent[b] = a;
        if (rnk[a] == rnk[b]) rnk[a]++;
        comps--;
        return true;
    }
};

static inline bool betterEdge(const vector<Edge>& edges, int oldIdx, int newIdx) {
    if (newIdx == -1) return false;
    if (oldIdx == -1) return true;

    const auto& A = edges[oldIdx];
    const auto& B = edges[newIdx];

    if (B.w != A.w) return B.w < A.w;

    int au = min(A.u, A.v), av = max(A.u, A.v);
    int bu = min(B.u, B.v), bv = max(B.u, B.v);
    if (bu != au) return bu < au;
    return bv < av;
}

// Parallel Boruvka MST / MSF
pair<LL, vector<Edge>> boruvkaMST_parallel(int n, const vector<Edge>& edges) {
    DSU dsu(n);
    LL total = 0;
    vector<Edge> mst;
    mst.reserve(max(0, n - 1));

    vector<int> bestEdge(n, -1);

    while (true) {
        fill(bestEdge.begin(), bestEdge.end(), -1);

        int T = omp_get_max_threads();
        vector<vector<int>> localBest(T, vector<int>(n, -1));

        // Phase 1: parallel scan edges, thread-local best outgoing edge per component
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& myBest = localBest[tid];

            #pragma omp for schedule(static)
            for (int i = 0; i < (int)edges.size(); i++) {
                int u = edges[i].u;
                int v = edges[i].v;

                int ru = dsu.find(u);
                int rv = dsu.find(v);
                if (ru == rv) continue;

                if (betterEdge(edges, myBest[ru], i)) myBest[ru] = i;
                if (betterEdge(edges, myBest[rv], i)) myBest[rv] = i;
            }
        }

        // Phase 2: reduce thread-local best results
        for (int t = 0; t < T; t++) {
            for (int r = 0; r < n; r++) {
                if (betterEdge(edges, bestEdge[r], localBest[t][r])) {
                    bestEdge[r] = localBest[t][r];
                }
            }
        }

        // Optional dedup: collect candidate edges first
        vector<int> chosen;
        chosen.reserve(n);
        vector<char> used(edges.size(), 0);

        for (int r = 0; r < n; r++) {
            int idx = bestEdge[r];
            if (idx == -1) continue;
            if (!used[idx]) {
                used[idx] = 1;
                chosen.push_back(idx);
            }
        }

        // Phase 3: sequential union (safe and simple)
        bool progress = false;
        for (int idx : chosen) {
            const auto& e = edges[idx];
            if (dsu.unite(e.u, e.v)) {
                mst.push_back(e);
                total += e.w;
                progress = true;
                if ((int)mst.size() == n - 1) break;
            }
        }

        if (!progress || (int)mst.size() == n - 1) break;
    }

    return {total, mst};
}

int main() {
    int n, m;
    cin >> n >> m;

    vector<Edge> edges;
    edges.reserve(m);

    // Input format: u v w (0-indexed)
    for (int i = 0; i < m; i++) {
        int u, v;
        LL w;
        cin >> u >> v >> w;
        edges.push_back({u, v, w});
    }

    auto [total, mst] = boruvkaMST_parallel(n, edges);

    cout << "Total weight = " << total << "\n";
    cout << "Edges in MST/MSF (" << mst.size() << "):\n";
    for (const auto& e : mst) {
        cout << e.u << " " << e.v << " " << e.w << "\n";
    }

    return 0;
}
