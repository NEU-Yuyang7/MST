#include<bits/stdc++.h>
#define ll long long 

using namespace std;

static const ll INF = (1LL<<62);

struct Edge {
    int to;
    ll w;
};

struct Graph {
    int n;
    vector<vector<Edge> > adj;
    Graph(int n=0): n(n), adj(n) {}
    void addEdge(int u, int v, ll w) { adj[u].push_back({v,w}); }
};

/*
  Global arrays to mimic the paper's labels:
  - d_hat[v] : current upper-bound estimate of shortest distance d(v)
  - pred[v]  : predecessor pointer maintaining a shortest-path tree wrt d_hat
*/
struct Labels {
    vector<ll> d_hat;
    vector<int> pred;
    Labels(int n=0): d_hat(n, INF), pred(n, -1) {}
};

/* -----------------------------
   Utility: relax edge (u -> v)
   Paper: if d_hat[u] + w_uv <= old d_hat[v], update and pred[v]=u
   ----------------------------- */
static inline bool relaxEdge(int u, int v, ll w, Labels &L) {
    if (L.d_hat[u] == INF) return false;
    ll cand = L.d_hat[u] + w;
    if (cand <= L.d_hat[v]) {
        L.d_hat[v] = cand;
        L.pred[v] = u;
        return true;
    }
    return false;
}

/* =========================================================
   Algorithm 2: BaseCase(B, S={x})
   - Truncated Dijkstra from x, only for nodes with d_hat < B
   - Stop after extracting k+1 nodes or heap empty
   - Return:
       if |U0| <= k : B'=B, U=U0
       else         : B' = max d_hat in U0, U = {v in U0: d_hat[v] < B'}
   ========================================================= */
struct BaseCaseResult {
    ll Bprime;
    vector<int> U;
};

BaseCaseResult BaseCase_Alg2(const Graph &g, ll B, int x, int k, Labels &L) {
    // requirement: S={x}, x is complete (assumed by caller)
    vector<int> U0;
    U0.reserve(k+1);
    U0.push_back(x);

    // binary heap over (dist, node)
    using P = pair<ll,int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    pq.push({L.d_hat[x], x});

    vector<char> inPQ(g.n, false);
    inPQ[x] = true;

    while (!pq.empty() && (int)U0.size() < k + 1) {
        auto [du, u] = pq.top();
        pq.pop();
        if (du != L.d_hat[u]) continue; // stale
        // In classic Dijkstra you'd mark visited; here we allow duplicates but keep simple
        // Add u if not already in U0
        // (paper adds each ExtractMin into U0; duplicates won't happen if you mark visited)
        // We'll mark visited to match paper behavior.
        // But we already pushed x before loop, so skip if u already in U0.
        // We'll maintain a visited flag:
        static vector<char> visited;
        if ((int)visited.size() != g.n) visited.assign(g.n, false);
        if (visited[u]) continue;
        visited[u] = true;
        if (u != x) U0.push_back(u);

        for (auto e : g.adj[u]) {
            int v = e.to;
            ll w = e.w;
            if (L.d_hat[u] == INF) continue;
            ll cand = L.d_hat[u] + w;
            if (cand <= L.d_hat[v] && cand < B) {
                L.d_hat[v] = cand;
                L.pred[v] = u;
                pq.push({L.d_hat[v], v});
            }
        }
    }

    BaseCaseResult res;
    if ((int)U0.size() <= k) {
        res.Bprime = B;
        res.U = U0;
        return res;
    } else {
        ll mx = 0;
        for (int v : U0) mx = max(mx, L.d_hat[v]);
        res.Bprime = mx;
        vector<int> U;
        for (int v : U0) if (L.d_hat[v] < res.Bprime) U.push_back(v);
        res.U = std::move(U);
        return res;
    }
}

/* =========================================================
   Algorithm 1: FindPivots(B, S)
   Simplified but aligned to pseudocode:

   - Run k steps of "frontier relax" starting from S:
       W0 = S
       for i=1..k:
         Wi = empty
         for u in Wi-1:
           for (u,v):
             if relax improves and d_hat[u]+w < B: put v into Wi
         W union Wi
       If |W| > k|S|: return P=S, W

   - Build forest F on W using pred pointers (approximation of paper's F):
       F contains edge pred[v] -> v if pred[v] in W and d_hat[v]==d_hat[pred[v]]+w(pred[v],v)
     Under unique-length assumption, it's a forest.

   - P = roots in S whose tree size >= k
   ========================================================= */
struct FindPivotsResult {
    vector<int> P; // pivots (subset of S)
    vector<int> W; // explored/collected
};

static inline ll edgeWeightIfExists(const Graph &g, int u, int v) {
    for (auto &e : g.adj[u]) if (e.to == v) return e.w;
    return INF;
}

FindPivotsResult FindPivots_Alg1(const Graph &g, ll B, const vector<int> &S, int k, Labels &L) {
    vector<int> W;
    W.reserve(k * (int)S.size() + 5);

    vector<int> Wi_prev = S;
    vector<char> inW(g.n, false);
    for (int x : S) {
        if (!inW[x]) { inW[x] = true; W.push_back(x); }
    }

    for (int i = 1; i <= k; i++) {
        vector<int> Wi;
        for (int u : Wi_prev) {
            for (auto e : g.adj[u]) {
                int v = e.to;
                ll w = e.w;
                // condition in paper: if d_hat[u] + w <= d_hat[v], update; if < B put into Wi
                ll old = L.d_hat[v];
                bool updated = relaxEdge(u, v, w, L);
                if (updated) {
                    ll cand = L.d_hat[u] + w;
                    if (cand < B) {
                        if (!inW[v]) {
                            inW[v] = true;
                            Wi.push_back(v);
                        }
                    }
                } else {
                    // paper says "even when equal" some places; for Alg1, we follow pseudocode
                    (void)old;
                }
            }
        }
        for (int v : Wi) W.push_back(v);

        if ((int)W.size() > k * (int)S.size()) {
            // early return: P = S
            FindPivotsResult r;
            r.P = S;
            r.W = W;
            return r;
        }
        Wi_prev = std::move(Wi);
        if (Wi_prev.empty()) break;
    }

    // Build forest F on W using pred pointers
    vector<char> isInW(g.n, false);
    for (int v : W) isInW[v] = true;

    vector<vector<int>> children(g.n);
    vector<int> roots; roots.reserve(S.size());

    for (int v : W) {
        int p = L.pred[v];
        if (p >= 0 && isInW[p]) {
            ll w = edgeWeightIfExists(g, p, v);
            if (w != INF && L.d_hat[p] + w == L.d_hat[v]) {
                children[p].push_back(v);
            } else {
                // treat as root if pred doesn't match a valid tight edge
                roots.push_back(v);
            }
        } else {
            roots.push_back(v);
        }
    }

    // Compute subtree sizes for roots that are in S (pivots candidates)
    vector<int> subSize(g.n, 0);
    vector<char> seen(g.n, false);

    function<int(int)> dfs = [&](int u)->int {
        seen[u] = true;
        int sz = 1;
        for (int v : children[u]) {
            if (!seen[v]) sz += dfs(v);
        }
        subSize[u] = sz;
        return sz;
    };

    for (int r0 : roots) if (!seen[r0] && isInW[r0]) dfs(r0);

    unordered_set<int> setS;
    setS.reserve(S.size() * 2);
    for (int s : S) setS.insert(s);

    vector<int> P;
    for (int s : S) {
        // "root of a tree with >= k vertices"
        // We'll interpret: s is a root in the forest AND subtree size >= k
        bool isRoot = (L.pred[s] < 0) || !isInW[L.pred[s]];
        if (isRoot && subSize[s] >= k) P.push_back(s);
    }

    FindPivotsResult r;
    r.P = std::move(P);
    r.W = std::move(W);
    return r;
}

/* =========================================================
   Lemma 3.3 Data structure D (SIMULATOR)
   - Supports Insert(key, value), BatchPrepend(list), Pull() returning (B_i, S_i)
   - This is a simplified "correctness demo" structure:
       - Keep best value per key
       - Store (value -> set of keys) in an ordered map
       - Pull returns up to M keys with smallest values and a boundary Bi = next smallest value (or B if none)
   ========================================================= */
struct PullResult {
    ll Bi;
    vector<int> Si;
};

struct DataStructureD {
    int M = 0;
    ll B = INF;

    unordered_map<int,ll> best;           // key -> current best value
    map<ll, vector<int>> buckets;         // value -> keys (may contain stale keys)

    void Initialize(int M_, ll B_) {
        M = M_;
        B = B_;
        best.clear();
        buckets.clear();
    }

    bool empty() const { return best.empty(); }

    void Insert(int key, ll value) {
        auto it = best.find(key);
        if (it == best.end() || value < it->second) {
            best[key] = value;
            buckets[value].push_back(key);
        }
    }

    void BatchPrepend(const vector<pair<int,ll>> &items) {
        // In the paper, all these items have values smaller than any current value.
        // We do not enforce that here; we just Insert with min-value semantics.
        for (auto &kv : items) Insert(kv.first, kv.second);
    }

    PullResult Pull() {
        PullResult res;
        res.Si.clear();
        res.Si.reserve(M);

        // Extract up to M keys with smallest values (deduplicate + skip stale)
        auto it = buckets.begin();
        while (it != buckets.end() && (int)res.Si.size() < M) {
            ll val = it->first;
            auto &vec = it->second;

            while (!vec.empty() && (int)res.Si.size() < M) {
                int key = vec.back();
                vec.pop_back();
                auto bit = best.find(key);
                if (bit == best.end()) continue;
                if (bit->second != val) continue; // stale record
                // accept
                res.Si.push_back(key);
                best.erase(bit);
            }

            if (vec.empty()) it = buckets.erase(it);
            else ++it;
        }

        // Compute boundary Bi: smallest remaining value or B if none
        if (buckets.empty()) res.Bi = B;
        else res.Bi = buckets.begin()->first;

        return res;
    }
};

/* =========================================================
   Algorithm 3: BMSSP(l, B, S)
   - Uses FindPivots, DataStructureD, recursion, relax, batch-prepend
   - Returns (B', U)
   NOTE: This mirrors the pseudocode structure, but uses simplified D.
   ========================================================= */
struct BMSSPResult {
    ll Bprime;
    vector<int> U;
};

struct Params {
    int k;
    int t;  // only used for M=2^{(l-1)t} and thresholds
};

BMSSPResult BMSSP_Alg3(const Graph &g, int l, ll B, const vector<int> &S, const Params &par, Labels &L) {
    int k = par.k;
    int t = par.t;

    if (l == 0) {
        // requirement: S is singleton
        int x = S.at(0);
        return { BaseCase_Alg2(g, B, x, k, L).Bprime,
                 BaseCase_Alg2(g, B, x, k, L).U };
    }

    // 1) Find pivots
    auto fp = FindPivots_Alg1(g, B, S, k, L);
    const vector<int> &P = fp.P;
    const vector<int> &W = fp.W;

    // 2) Init D with M = 2^{(l-1)t}
    DataStructureD D;
    // be careful with shift overflow
    int M = 1;
    {
        long long exp = 1LL * (l - 1) * t;
        if (exp >= 30) M = 1 << 30; // cap for demo
        else M = 1 << exp;
        if (M <= 0) M = 1;
    }
    D.Initialize(M, B);

    for (int x : P) D.Insert(x, L.d_hat[x]);

    ll Bprime0 = B;
    if (!P.empty()) {
        Bprime0 = INF;
        for (int x : P) Bprime0 = min(Bprime0, L.d_hat[x]);
    }

    vector<int> U;
    U.reserve(k * (1 << (l*t)) + 5); // rough

    // Main loop: while |U| < k*2^{lt} and D non-empty
    long long limitU = 1LL * k;
    {
        long long exp = 1LL * l * t;
        if (exp >= 30) limitU *= (1LL<<30);
        else limitU *= (1LL<<exp);
    }

    ll Bprime = Bprime0;
    while ((long long)U.size() < limitU && !D.empty()) {
        auto pr = D.Pull();
        ll Bi = pr.Bi;
        vector<int> Si = std::move(pr.Si);

        if (Si.empty()) break;

        // Recursive call
        auto child = BMSSP_Alg3(g, l-1, Bi, Si, par, L);
        ll Bip = child.Bprime;
        vector<int> Ui = std::move(child.U);

        // Merge Ui into U
        U.insert(U.end(), Ui.begin(), Ui.end());

        // Relax edges out of Ui and classify updates
        vector<pair<int,ll>> K; // for batch prepend
        for (int u : Ui) {
            for (auto e : g.adj[u]) {
                int v = e.to;
                ll w = e.w;
                if (L.d_hat[u] == INF) continue;
                ll cand = L.d_hat[u] + w;
                if (cand <= L.d_hat[v]) {
                    L.d_hat[v] = cand;
                    L.pred[v] = u;

                    if (cand >= Bi && cand < B) {
                        D.Insert(v, cand);
                    } else if (cand >= Bip && cand < Bi) {
                        K.push_back({v, cand});
                    }
                }
            }
        }

        // Batch prepend: K plus {x in Si with d_hat[x] in [Bip, Bi)}
        vector<pair<int,ll>> batch = std::move(K);
        for (int x : Si) {
            ll dx = L.d_hat[x];
            if (dx >= Bip && dx < Bi) batch.push_back({x, dx});
        }
        D.BatchPrepend(batch);

        // success condition: D empty -> return
        if (D.empty()) {
            Bprime = min(Bip, B);
            break;
        }

        // partial execution condition
        if ((long long)U.size() > limitU) {
            Bprime = Bi;
            break;
        }

        Bprime = min(Bip, B); // keep updated
    }

    // Final: add x in W with d_hat[x] < Bprime
    // (paper uses W from FindPivots)
    for (int x : W) {
        if (L.d_hat[x] < Bprime) U.push_back(x);
    }

    // Optionally: de-duplicate U
    sort(U.begin(), U.end());
    U.erase(unique(U.begin(), U.end()), U.end());

    return {Bprime, U};
}

/* ---------------------------
   Example usage (toy)
   --------------------------- */
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // Build a tiny sample directed graph
    int n = 6;
    Graph g(n);
    g.addEdge(0,1,2);
    g.addEdge(0,2,5);
    g.addEdge(1,2,1);
    g.addEdge(1,3,2);
    g.addEdge(2,3,1);
    g.addEdge(3,4,3);
    g.addEdge(2,5,10);
    g.addEdge(4,5,1);

    Labels L(n);
    int s = 0;
    L.d_hat[s] = 0;
    L.pred[s] = -1;

    // Parameters (toy)
    Params par;
    par.k = 2; // in paper k=floor(log^{1/3} n); here small demo
    par.t = 1; // in paper t=floor(log^{2/3} n); here small demo

    // Top-level call: BMSSP(l, B, S={s})
    int l = 2; // toy recursion depth
    ll B = INF;
    vector<int> S = {s};

    auto res = BMSSP_Alg3(g, l, B, S, par, L);

    cout << "B' = " << res.Bprime << "\n";
    cout << "U (complete set returned) = ";
    for (int v : res.U) cout << v << " ";
    cout << "\n";

    cout << "d_hat after run:\n";
    for (int i=0;i<n;i++) {
        cout << "  " << i << ": " << (L.d_hat[i]>=INF/2? -1: L.d_hat[i]) << "\n";
    }
    return 0;
}

