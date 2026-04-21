// boruvka with xi instrumentation
// Outputs JSON to stderr: per-round component shrinkage
#include <bits/stdc++.h>
#define LL long long
using namespace std;
struct Edge{int u,v;LL w;};
struct DSU{
    vector<int> p,r;int comps;
    DSU(int n){p.resize(n);r.assign(n,0);iota(p.begin(),p.end(),0);comps=n;}
    int find(int x){while(p[x]!=x){p[x]=p[p[x]];x=p[x];}return x;}
    bool unite(int a,int b){a=find(a);b=find(b);if(a==b)return false;
        if(r[a]<r[b])swap(a,b);p[b]=a;if(r[a]==r[b])r[a]++;comps--;return true;}
};
int main(){
    ios::sync_with_stdio(false);cin.tie(nullptr);
    int n,m;cin>>n>>m;
    vector<Edge>E(m);for(auto&e:E)cin>>e.u>>e.v>>e.w;
    DSU dsu(n);
    vector<int>best(n,-1);
    LL total=0;int mst_cnt=0;
    bool prog=true;int round=0;
    vector<double> xi_rounds;
    vector<int> C_trace;
    while(prog){
        prog=false;fill(best.begin(),best.end(),-1);
        int C_before=dsu.comps;
        C_trace.push_back(C_before);
        for(int i=0;i<m;i++){
            int ru=dsu.find(E[i].u),rv=dsu.find(E[i].v);
            if(ru==rv)continue;
            auto bet=[&](int o,int ni)->bool{if(o<0)return true;return E[ni].w<E[o].w;};
            if(bet(best[ru],i))best[ru]=i;
            if(bet(best[rv],i))best[rv]=i;
        }
        for(int r=0;r<n;r++){int idx=best[r];if(idx<0)continue;
            if(dsu.unite(E[idx].u,E[idx].v)){total+=E[idx].w;prog=true;mst_cnt++;
                if(mst_cnt==n-1)goto done;}}
        if(mst_cnt==n-1)break;
        int C_after=dsu.comps;
        double xi_i=(C_after>0&&C_before>C_after)?(double)C_before/C_after:1.0;
        xi_rounds.push_back(xi_i);
        fprintf(stderr,"{\"type\":\"round\",\"round\":%d,\"C_before\":%d,\"C_after\":%d,\"xi\":%.4f}\n",
                round,C_before,C_after,xi_i);
        round++;
    }
    done:
    double log_sum=0;int cnt=0;
    for(double x:xi_rounds){if(x>1.0){log_sum+=log(x);cnt++;}}
    double xi_gm=cnt>0?exp(log_sum/cnt):1.0;
    fprintf(stderr,"{\"type\":\"summary\",\"n\":%d,\"m\":%d,\"rounds\":%d,"
            "\"xi_geomean\":%.4f,\"xi_theory\":2.0}\n",n,m,round,xi_gm);
    printf("Total weight = %lld\nEdges in MST/MSF (%d):\n",total,mst_cnt);
}
