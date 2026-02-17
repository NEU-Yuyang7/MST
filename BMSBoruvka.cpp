#include <bits/stdc++.h>
using namespace std;

struct Edge {
    int u, v;
    long long w;
};

struct DSU {
    vector<int> p, r;
    int comps;

    DSU(int n=0){ init(n); }
    void init(int n){
        p.resize(n);
        r.assign(n,0);
        iota(p.begin(), p.end(), 0);
        comps = n;
    }
    int find(int x){
        while(p[x]!=x){
            p[x] = p[p[x]]; // path compression (halving)
            x = p[x];
        }
        return x;
    }
    bool unite(int a,int b){
        a = find(a); b = find(b);
        if(a==b) return false;
        if(r[a] < r[b]) swap(a,b);
        p[b]=a;
        if(r[a]==r[b]) r[a]++;
        comps--;
        return true;
    }
};

static inline bool betterEdge(const Edge& A, const Edge& B){
    // deterministic tie-breaker
    if(A.w != B.w) return A.w < B.w;
    int au = min(A.u, A.v), av = max(A.u, A.v);
    int bu = min(B.u, B.v), bv = max(B.u, B.v);
    if(au != bu) return au < bu;
    return av < bv;
}

/* ============================================================
   Candidate structure D:
   - Update(componentRoot, edgeIndex)
   - Pull(M): return up to M component roots with smallest candidate weights
   - BatchPrepend: just bulk Update (prototype)
   Implementation:
     - best[compRoot] = current best edge idx (may be stale after unions)
     - heap stores (weight, compRoot, edgeIdx, stamp)
     - lazy deletion: only accept heap top if matches current best stamp AND compRoot is still root.
   ============================================================ */
struct CandidateD {
    struct Item{
        long long w;
        int comp;
        int eidx;
        int stamp;
        bool operator>(const Item& o) const {
            if(w != o.w) return w > o.w;
            if(comp != o.comp) return comp > o.comp;
            return eidx > o.eidx;
        }
    };

    const vector<Edge>* edges = nullptr;
    DSU* dsu = nullptr;

    unordered_map<int,int> bestEdge;   // compRoot -> edge idx
    unordered_map<int,int> bestStamp;  // compRoot -> stamp
    int globalStamp = 1;

    priority_queue<Item, vector<Item>, greater<Item>> pq;

    void init(const vector<Edge>& E, DSU& uf){
        edges = &E;
        dsu = &uf;
        bestEdge.clear();
        bestStamp.clear();
        globalStamp = 1;
        while(!pq.empty()) pq.pop();
    }

    void update(int compRoot, int edgeIdx){
        if(!edges) return;
        // only meaningful if compRoot is still a root at time of update
        compRoot = dsu->find(compRoot);

        auto it = bestEdge.find(compRoot);
        if(it == bestEdge.end()){
            bestEdge[compRoot] = edgeIdx;
            bestStamp[compRoot] = globalStamp++;
            pq.push({(*edges)[edgeIdx].w, compRoot, edgeIdx, bestStamp[compRoot]});
            return;
        }
        int curIdx = it->second;
        const Edge& curE = (*edges)[curIdx];
        const Edge& newE = (*edges)[edgeIdx];

        if(betterEdge(newE, curE)){
            bestEdge[compRoot] = edgeIdx;
            bestStamp[compRoot] = globalStamp++;
            pq.push({newE.w, compRoot, edgeIdx, bestStamp[compRoot]});
        }
    }

    void batchPrepend(const vector<pair<int,int>>& compEdgePairs){
        // Prototype: just bulk update
        for(auto &p: compEdgePairs) update(p.first, p.second);
    }

    vector<int> pull(int M){
        vector<int> res;
        res.reserve(M);
        unordered_set<int> picked;
        picked.reserve(M*2 + 1);

        while((int)res.size() < M && !pq.empty()){
            auto top = pq.top(); pq.pop();
            int c = dsu->find(top.comp);

            // component root changed => stale
            if(c != top.comp) continue;

            // best stamp mismatch => stale
            auto itS = bestStamp.find(c);
            if(itS == bestStamp.end() || itS->second != top.stamp) continue;

            // avoid duplicates in one pull
            if(picked.count(c)) continue;

            res.push_back(c);
            picked.insert(c);
        }
        return res;
    }

    bool empty() const {
        return bestEdge.empty();
    }
};

/* ============================================================
   Threshold 而 (bucket / layer activation)
   We'll sort edges by weight.
   We activate edges in layers: each time we increase 而, we add a new chunk
   of edges (doubling chunk size). 而 is "weight of last activated edge".

   activeEnd is index in sorted edges: edges[0..activeEnd-1] are active.
   NewlyActivatedEdges are edges[oldEnd..newEnd-1].
   ============================================================ */
struct ThresholdActivator {
    const vector<Edge>* E = nullptr;
    int m = 0;
    int activeEnd = 0; // [0, activeEnd) active
    int layer = 0;     // chunk size grows with layer
    int chunkBase = 0; // initial chunk size

    void init(const vector<Edge>& edges, int initialChunk){
        E = &edges;
        m = (int)edges.size();
        activeEnd = 0;
        layer = 0;
        chunkBase = max(1, initialChunk);
    }

    // raise threshold: activate more edges; return range [oldEnd, newEnd)
    pair<int,int> raise(){
        int oldEnd = activeEnd;
        long long chunk = 1LL * chunkBase * (1LL << layer);
        layer++;
        activeEnd = (int)min<long long>(m, oldEnd + chunk);
        return {oldEnd, activeEnd};
    }

    bool allActivated() const { return activeEnd >= m; }
    long long tauValue() const {
        if(activeEnd <= 0) return LLONG_MIN;
        return (*E)[activeEnd-1].w;
    }
};

/* ============================================================
   ValidateCOE(C, 而, UF):
   - Fast path (lazy): try to use current best candidate edge from D by popping
     until we find a crossing edge.
   - If cannot find a valid candidate quickly, do a "refresh scan" over currently
     active edges to compute the true cheapest outgoing edge for this component,
     update D, and return it if exists.

   This keeps correctness for the active edge set, while being a practical prototype.
   ============================================================ */
struct BMSBoruvka {
    int n;
    vector<Edge> edges; // sorted by weight
    DSU uf;
    CandidateD D;
    ThresholdActivator activator;

    // progress control
    int M_pull = 64;              // batch size
    double progressFrac = 0.25;   // if merged < progressFrac * pulled, raise 而

    BMSBoruvka(int n_, vector<Edge> E): n(n_), edges(std::move(E)), uf(n_) {
        sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b){
            if(a.w != b.w) return a.w < b.w;
            int au=min(a.u,a.v), av=max(a.u,a.v);
            int bu=min(b.u,b.v), bv=max(b.u,b.v);
            if(au!=bu) return au<bu;
            return av<bv;
        });
        D.init(edges, uf);
        activator.init(edges, /*initialChunk*/ max(1, (int)edges.size()/64)); // heuristic
    }

    // Update D with newly activated edges
    void feedNewEdges(int oldEnd, int newEnd){
        for(int i=oldEnd;i<newEnd;i++){
            int u = edges[i].u, v = edges[i].v;
            int cu = uf.find(u), cv = uf.find(v);
            if(cu == cv) continue;
            D.update(cu, i);
            D.update(cv, i);
        }
    }

    // Refresh scan for component c over active edges: find true min crossing edge
    // among edges[0..activeEnd-1] (active edges only).
    int refreshCOE_byScanActive(int c){
        c = uf.find(c);
        int best = -1;
        for(int i=0;i<activator.activeEnd;i++){
            int u = edges[i].u, v = edges[i].v;
            int cu = uf.find(u), cv = uf.find(v);
            if(cu == cv) continue;
            if(cu != c && cv != c) continue; // not incident to component c
            if(best == -1 || betterEdge(edges[i], edges[best])) best = i;
        }
        if(best != -1) D.update(c, best);
        return best;
    }

    // Lazy validate: try current best edge for component c
    // Return edge index if found, else -1.
    int validateCOE(int c){
        c = uf.find(c);

        // Try to use the current best edge recorded in D (if exists)
        // We'll probe by repeatedly pulling candidates for this component from the heap indirectly:
        // easiest: do a small "mini-pull" and check.
        // But D.pull gives distinct comps; here we want just c.
        // So we do a bounded number of refresh attempts: first scan-based refresh if needed.

        // 1) If D has some record for c, test it (bestEdge map)
        auto it = D.bestEdge.find(c);
        if(it != D.bestEdge.end()){
            int eidx = it->second;
            int u = edges[eidx].u, v = edges[eidx].v;
            int cu = uf.find(u), cv = uf.find(v);
            if(cu != cv){
                // ensure this edge is incident to c
                if(cu == c || cv == c) return eidx;
            }
        }

        // 2) Otherwise do a refresh scan over active edges (correct for active set)
        int best = refreshCOE_byScanActive(c);
        return best;
    }

    pair<long long, vector<Edge>> run(){
        uf.init(n);
        D.init(edges, uf);

        long long total = 0;
        vector<Edge> mst;
        mst.reserve(max(0, n-1));

        // Initial 而 activation
        auto [oldEnd, newEnd] = activator.raise();
        feedNewEdges(oldEnd, newEnd);

        // Main loop
        while(uf.comps > 1){
            // if D is empty (no candidates), we must raise 而 (possibly disconnected graph)
            if(D.pq.empty()){
                if(activator.allActivated()) break; // cannot progress (disconnected)
                auto rng = activator.raise();
                feedNewEdges(rng.first, rng.second);
                continue;
            }

            // Pull a batch of components
            vector<int> S = D.pull(M_pull);
            if(S.empty()){
                // candidates were stale; raise 而 to inject more edges
                if(activator.allActivated()) break;
                auto rng = activator.raise();
                feedNewEdges(rng.first, rng.second);
                continue;
            }

            int merged = 0;
            vector<pair<int,int>> newly; // (componentRoot, edgeIdx) to batch prepend

            for(int c : S){
                c = uf.find(c);
                int eidx = validateCOE(c);
                if(eidx == -1) continue;

                int u = edges[eidx].u, v = edges[eidx].v;
                int cu = uf.find(u), cv = uf.find(v);
                if(cu == cv) continue;

                // union
                bool ok = uf.unite(cu, cv);
                if(!ok) continue;

                mst.push_back(edges[eidx]);
                total += edges[eidx].w;
                merged++;

                // Local relax (prototype): add candidates using edges adjacent to u and v among active edges
                // Here we cheaply "re-feed" a small window: just update using this edge for the new root
                int newRoot = uf.find(u);
                newly.push_back({newRoot, eidx});

                if((int)mst.size() == n-1) break;
            }

            // Batch prepend newly discovered candidates
            D.batchPrepend(newly);

            if((int)mst.size() == n-1) break;

            // Progress check: if not enough merges, raise 而 and activate more edges
            int threshold = max(1, (int)ceil(progressFrac * (double)S.size()));
            if(merged < threshold){
                if(!activator.allActivated()){
                    auto rng = activator.raise();
                    feedNewEdges(rng.first, rng.second);
                }else{
                    // all edges activated; if still low progress, we're likely disconnected or stuck
                    // We'll continue attempts, but may exit if no improvement.
                }
            }
        }

        return {total, mst};
    }
};

int main(){
    int n, m;
    cin >> n >> m;
    vector<Edge> E;
    E.reserve(m);

    // Input format: u v w (0-indexed). If 1-indexed, subtract 1.
    for(int i=0;i<m;i++){
        int u,v; long long w;
        cin >> u >> v >> w;
        E.push_back({u,v,w});
    }

    BMSBoruvka solver(n, E);
    auto [total, mst] = solver.run();

    cout << "Total weight = " << total << "\n";
    cout << "Edges in MST/MSF (" << mst.size() << "):\n";
    for(auto &e : mst){
        cout << e.u << " " << e.v << " " << e.w << "\n";
    }

    return 0;
}

