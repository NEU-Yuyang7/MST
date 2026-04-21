// BMSBoruvka with xi instrumentation
// Outputs to stderr (JSON lines) so stdout stays compatible with original format:
//   {"type":"step","step":0,"C_before":7,"C_after":3,"M_i":5,"t":2,"xi":2.333}
//   {"type":"summary","n":7,"m":11,"P":2,"t":2,"xi_geomean":2.646,"xi_theory":4.0}
#include <bits/stdc++.h>
using namespace std;

struct Edge { int u, v; long long w; };
struct CompEdge { int u, v; long long w; int orig_eidx; };

struct DSU {
    vector<int> p, r; int comps;
    void init(int n){ p.resize(n); r.assign(n,0); iota(p.begin(),p.end(),0); comps=n; }
    int find(int x){ while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x; }
    bool unite(int a,int b){
        a=find(a);b=find(b);if(a==b)return false;
        if(r[a]<r[b])swap(a,b);p[b]=a;if(r[a]==r[b])r[a]++;comps--;return true;
    }
};

bool edgeLess(const CompEdge&a,const CompEdge&b){
    return a.w!=b.w?a.w<b.w:a.orig_eidx<b.orig_eidx;}
bool edgeWeightLess(const Edge&a,const Edge&b){ return a.w<b.w; }
int computeT(int n){ return max(1,(int)ceil(pow(log2((double)n),2.0/3.0))); }
int computeP(int n,int t){ return max(1,(int)ceil(log2((double)n)/t)); }

vector<CompEdge> boruvkaRound(const vector<CompEdge>&edges, DSU&uf){
    int n=uf.p.size();
    vector<int> best(n,-1);
    for(int i=0;i<(int)edges.size();i++){
        int ra=uf.find(edges[i].u),rb=uf.find(edges[i].v);
        if(ra==rb)continue;
        auto upd=[&](int r){if(best[r]<0||edgeLess(edges[i],edges[best[r]]))best[r]=i;};
        upd(ra);upd(rb);
    }
    vector<CompEdge> sel;
    set<int>seen;
    for(int r=0;r<n;r++)if(best[r]>=0&&!seen.count(best[r])){seen.insert(best[r]);sel.push_back(edges[best[r]]);}
    for(auto&e:sel)uf.unite(e.u,e.v);
    return sel;
}

int main(){
    ios::sync_with_stdio(false); cin.tie(nullptr);
    int n,m; cin>>n>>m;
    vector<Edge> edges(m);
    for(int i=0;i<m;i++){cin>>edges[i].u>>edges[i].v>>edges[i].w; }

    int t=computeT(n), P=computeP(n,t);
    double xi_theory = pow(2.0, t);

    DSU uf; uf.init(n);
    vector<CompEdge> curEdges;
    int curActivated=0;
    int edgesPerStep=max(1,m/max(1,P));
    vector<int> o2c(n); iota(o2c.begin(),o2c.end(),0);

    long long mst_w=0;
    vector<int> mst_eids;

    // xi tracking
    vector<double> xi_per_step;
    vector<int> C_trace;   // component count at start of each superstep
    vector<int> M_trace;   // curEdges size entering Phase B each superstep

    for(int step=0;step<P&&uf.comps>1;step++){
        int C_before = uf.comps;
        C_trace.push_back(C_before);

        int activateEnd=(step==P-1)?m:min(m,curActivated+edgesPerStep);
        if(activateEnd<m)
            nth_element(edges.begin()+curActivated,
                        edges.begin()+activateEnd-1,
                        edges.end(),[](auto&a,auto&b){return a.w<b.w;});

        for(int i=curActivated;i<activateEnd;i++){
            auto&e=edges[i];
            int cu=uf.find(o2c[e.u]),cv=uf.find(o2c[e.v]);
            if(cu!=cv) curEdges.push_back({cu,cv,e.w,i});
        }
        curActivated=activateEnd;

        int M_i=(int)curEdges.size();
        M_trace.push_back(M_i);

        // Phase B
        for(int round=0;round<t&&uf.comps>1;round++){
            vector<CompEdge> active;
            for(auto&e:curEdges){int ra=uf.find(e.u),rb=uf.find(e.v);
                if(ra!=rb){e.u=ra;e.v=rb;active.push_back(e);}}
            curEdges=active;
            if(curEdges.empty())break;
            auto sel=boruvkaRound(curEdges,uf);
            for(auto&e:sel){mst_eids.push_back(e.orig_eidx);mst_w+=e.w;}
            if((int)mst_eids.size()==n-1)goto done;
        }

        if(uf.comps<=1)break;

        // Phase C compress
        {
            vector<CompEdge> active;
            for(auto&e:curEdges){int ra=uf.find(e.u),rb=uf.find(e.v);
                if(ra!=rb){e.u=ra;e.v=rb;active.push_back(e);}}
            curEdges=active;
            if(curEdges.empty()){
                // still emit step data
                int C_after=uf.comps;
                double xi_i=(C_before>0&&C_after>0)?(double)C_before/C_after:1.0;
                xi_per_step.push_back(xi_i);
                fprintf(stderr,"{\"type\":\"step\",\"step\":%d,\"C_before\":%d,\"C_after\":%d,"
                        "\"M_i\":%d,\"t\":%d,\"xi\":%.4f}\n",step,C_before,C_after,M_i,t,xi_i);
                continue;
            }
            // rebuild compressed graph
            int n_cur=uf.p.size();
            vector<int> rtn(n_cur,-1);
            int newID=0;
            for(int i=0;i<n_cur;i++)if(uf.find(i)==i)rtn[i]=newID++;
            unordered_map<uint64_t,CompEdge> best;
            for(auto&e:curEdges){
                int na=rtn[uf.find(e.u)],nb=rtn[uf.find(e.v)];
                if(na==nb)continue;
                if(na>nb)swap(na,nb);
                uint64_t key=((uint64_t)na<<32)|nb;
                auto it=best.find(key);
                if(it==best.end()||edgeLess(e,it->second))best[key]=e;
            }
            for(int i=0;i<n_cur;i++)if(rtn[uf.find(o2c[i])]>=0)o2c[i]=rtn[uf.find(o2c[i])];
            uf.init(newID);
            curEdges.clear();
            for(auto&[k,e]:best)curEdges.push_back(e);

            int C_after=uf.comps;
            double xi_i=(C_before>0&&C_after>0)?(double)C_before/C_after:1.0;
            xi_per_step.push_back(xi_i);
            fprintf(stderr,"{\"type\":\"step\",\"step\":%d,\"C_before\":%d,\"C_after\":%d,"
                    "\"M_i\":%d,\"t\":%d,\"xi\":%.4f}\n",step,C_before,C_after,M_i,t,xi_i);
        }
    }
    done:
    mst_w=0;
    for(int eid:mst_eids)mst_w+=edges[eid].w;

    // Geometric mean of xi
    double log_sum=0; int cnt=0;
    for(double x:xi_per_step){if(x>1.0){log_sum+=log(x);cnt++;}}
    double xi_geomean = cnt>0?exp(log_sum/cnt):1.0;

    fprintf(stderr,"{\"type\":\"summary\",\"n\":%d,\"m\":%d,\"P\":%d,\"t\":%d,"
            "\"steps_actual\":%d,\"xi_geomean\":%.4f,\"xi_theory\":%.4f,"
            "\"C_trace\":[",n,m,P,t,(int)xi_per_step.size(),xi_geomean,xi_theory);
    for(int i=0;i<(int)C_trace.size();i++)fprintf(stderr,"%s%d",i?",":"",C_trace[i]);
    fprintf(stderr,"],\"M_trace\":[");
    for(int i=0;i<(int)M_trace.size();i++)fprintf(stderr,"%s%d",i?",":"",M_trace[i]);
    fprintf(stderr,"]}\n");

    printf("Total weight = %lld\n",mst_w);
    printf("Edges in MST/MSF (%d):\n",(int)mst_eids.size());
    for(int eid:mst_eids)printf("%d %d %lld\n",edges[eid].u,edges[eid].v,edges[eid].w);
}
