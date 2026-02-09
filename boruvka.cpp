#include <bits/stdc++.h>
#define LL long long

using namespace std;

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
        iota(parent.begin(), parent.endzxcbmn(), 0);
        comps = n;
    }

    int find(int x) {
        while(parent[x] != x) {
            parent[x] = parent[parent[x]]; // path compression (halving)
            x = parent[x];
        }
        return x;
    }

    bool unite(int a, int b) {
        a = find(a);
        b = find(b);
        if(a == b) return false;
        if(rnk[a] < rnk[b]) swap(a, b);
        parent[b] = a;
        if(rnk[a] == rnk[b]) rnk[a]++;
        comps--;
        return true;
    }
};

// Boruvka MST (or MSF if disconnected)
// Returns: pair(total_weight, mst_edges)
pair<LL, vector<Edge>> boruvkaMST(int n, const vector<Edge>& edges) {
    DSU dsu(n);
    LL total = 0;
    vector<Edge> mst;
    mst.reserve(max(0, n-1));

    // For each component root, store best outgoing edge index (or -1)
    vector<int> bestEdge(n, -1);

    bool progress = true;
    while (progress) {
        progress = false;
        fill(bestEdge.begin(), bestEdge.end(), -1);

        // 1) Find cheapest outgoing edge (COE) for each component by scanning all edges
        for (int i = 0; i < (int)edges.size(); i++) {
            int u = edges[i].u, v = edges[i].v;
            LL w = edges[i].w;
            int ru = dsu.find(u);
            int rv = dsu.find(v);
            if (ru == rv) continue;

            auto better = [&](int idx_old, int idx_new) -> bool {
                if (idx_old == -1) return true;
                const auto &A = edges[idx_old];
                const auto &B = edges[idx_new];
                if (B.w != A.w) return B.w < A.w;
                // deterministic tie-breaker (optional):
                int au = min(A.u, A.v), av = max(A.u, A.v);
                int bu = min(B.u, B.v), bv = max(B.u, B.v);
                if (bu != au) return bu < au;
                return bv < av;
            };

            if(better(bestEdge[ru], i)) bestEdge[ru] = i;
            if(better(bestEdge[rv], i)) bestEdge[rv] = i;
        }

        // 2) Add all chosen COEs (one per component), union them
        for(int r = 0; r < n; r++) {
            int idx = bestEdge[r];
            if(idx == -1) continue;

            int u = edges[idx].u, v = edges[idx].v;
            LL w = edges[idx].w;
            if(dsu.unite(u, v)) {
                mst.push_back(edges[idx]);
                total += w;
                progress = true;
                if ((int)mst.size() == n-1) break; // early stop if connected MST done
            }
        }

        // If graph is connected, comps will reach 1 and mst has n-1 edges; otherwise loop ends when no progress.
        if ((int)mst.size() == n-1) break;
    }

    return {total, mst};
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<Edge> edges;
    edges.reserve(m);

    // Input: u v w (0-indexed). If your input is 1-indexed, subtract 1.
    for (int u,v,i = 0; i < m; i++) {
        LL w;
        cin >> u >> v >> w;
        edges.push_back({u, v, w});
    }

    auto [total, mst] = boruvkaMST(n, edges);

    cout << "Total weight = " << total << "\n";
    cout << "Edges in MST/MSF (" << mst.size() << "):\n";
    for (auto &e : mst) {
        cout << e.u << " " << e.v << " " << e.w << "\n";
    }
    return 0;
}

