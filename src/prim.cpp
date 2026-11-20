/*
 * prim.cpp  —  Prim's MST algorithm with a binary heap
 *
 * Lazy Prim: a std::priority_queue of candidate edges, keyed by weight; stale
 * entries (both endpoints already in the tree) are skipped when popped.
 * Complexity: O(m log m) = O(m log n).
 *
 * This is the textbook practical Prim. The O(m + n log n) bound in the paper's
 * related work needs a Fibonacci heap, whose constants make it slower in
 * practice; a binary heap is what implementations actually use.
 *
 * The adjacency structure (CSR) is built inside the timed region, because the
 * other binaries receive the same flat edge list and must do their own
 * preprocessing (sort, partition, activation) inside it too.
 *
 * Disconnected input: the outer loop restarts from every vertex not yet
 * reached, so the result is a minimum spanning forest, like the other binaries.
 *
 * Input (stdin):  n m
 *                 u v w   (0-indexed, repeated m times)
 * Output (stdout): Total weight = <W>
 *                  Edges in MST/MSF (<k>):
 *                  u v w  ...
 */
#include <bits/stdc++.h>
#include "bench_harness.h"
using namespace std;
using LL = long long;

struct Edge { int u, v; LL w; };

int main(int argc, char** argv) {
    bench::init(argc, argv);
    int n, m; cin >> n >> m;
    vector<Edge> E(m);
    for (auto& e : E) cin >> e.u >> e.v >> e.w;

    LL total = 0;
    vector<Edge> mst;

    for (int r = 0; r < bench::repeats(); r++) {
        double t0 = bench::now_ms();

        // CSR adjacency: off[v] .. off[v+1] index into adj (neighbour, edge id).
        vector<int> off(n + 1, 0);
        for (const auto& e : E) { off[e.u + 1]++; off[e.v + 1]++; }
        for (int v = 0; v < n; v++) off[v + 1] += off[v];
        vector<pair<int,int>> adj(2 * (size_t)m);
        {
            vector<int> pos(off.begin(), off.end() - 1);
            for (int i = 0; i < m; i++) {
                adj[pos[E[i].u]++] = {E[i].v, i};
                adj[pos[E[i].v]++] = {E[i].u, i};
            }
        }

        vector<char> in_tree(n, 0);
        // (weight, edge id, vertex the edge leads to); min-heap on weight.
        using Item = tuple<LL, int, int>;
        priority_queue<Item, vector<Item>, greater<Item>> pq;

        total = 0;
        mst.clear();
        mst.reserve(n > 0 ? n - 1 : 0);

        for (int s = 0; s < n && (int)mst.size() < n - 1; s++) {
            if (in_tree[s]) continue;
            in_tree[s] = 1;
            for (int k = off[s]; k < off[s + 1]; k++)
                if (!in_tree[adj[k].first])
                    pq.push({E[adj[k].second].w, adj[k].second, adj[k].first});

            while (!pq.empty()) {
                auto [w, id, v] = pq.top(); pq.pop();
                if (in_tree[v]) continue;            // stale entry
                in_tree[v] = 1;
                mst.push_back(E[id]);
                total += w;
                for (int k = off[v]; k < off[v + 1]; k++)
                    if (!in_tree[adj[k].first])
                        pq.push({E[adj[k].second].w, adj[k].second, adj[k].first});
            }
        }

        bench::report_time(bench::now_ms() - t0);
    }

    cout << "Total weight = " << total << "\n";
    if (bench::print_edges()) {
        cout << "Edges in MST/MSF (" << mst.size() << "):\n";
        for (const auto& e : mst)
            cout << e.u << " " << e.v << " " << e.w << "\n";
    }
    return 0;
}
