/**
 * gen.cpp — Random graph generator for MST benchmarks
 *
 * Usage:
 *   ./gen <n> <m> [options]
 *
 * Options:
 *   --seed <s>       Random seed (default: 42)
 *   --weights <mode> Weight distribution:
 *                      random   (default) uniform random in [1, 1e9]
 *                      uniform  all edges have the same weight (1)
 *                      small    uniform in [1, 100]  (many ties)
 *                      negpos   uniform in [-1e9, 1e9]
 *   --connected      Guarantee connectivity by adding a spanning path first
 *
 * Output format (stdout):
 *   n m
 *   u v w    (0-indexed, repeated m times)
 *
 * Examples:
 *   ./gen 1000000 5000000                          # sparse, random weights
 *   ./gen 100000  5000000 --connected              # dense, guaranteed connected
 *   ./gen 1000000 5000000 --weights uniform        # all-same-weight stress test
 *   ./gen 1000000 5000000 --seed 123 --weights small
 */

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "Usage: ./gen <n> <m> [--seed S] [--weights random|uniform|small|negpos] [--connected]\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int m = atoi(argv[2]);

    if (n < 2) { fprintf(stderr, "n must be >= 2\n"); return 1; }
    if (m < 1) { fprintf(stderr, "m must be >= 1\n"); return 1; }

    // Parse options
    uint64_t seed       = 42;
    string   weight_mode = "random";
    bool     connected  = false;

    for (int i = 3; i < argc; i++) {
        string a = argv[i];
        if (a == "--seed" && i+1 < argc) {
            seed = (uint64_t)atoll(argv[++i]);
        } else if (a == "--weights" && i+1 < argc) {
            weight_mode = argv[++i];
        } else if (a == "--connected") {
            connected = true;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return 1;
        }
    }

    mt19937_64 rng(seed);

    // Weight generator
    auto gen_weight = [&]() -> long long {
        if      (weight_mode == "uniform")  return 1LL;
        else if (weight_mode == "small")    return (long long)(rng() % 100 + 1);
        else if (weight_mode == "negpos")   return (long long)(rng() % 2000000001LL) - 1000000000LL;
        else /* random */                   return (long long)(rng() % 1000000000LL + 1);
    };

    // Collect edges
    vector<tuple<int,int,long long>> edges;
    edges.reserve(m);

    // If --connected, first lay down a random spanning path to guarantee connectivity.
    // This uses (n-1) of the m edge slots.
    if (connected) {
        if (m < n - 1) {
            fprintf(stderr, "m must be >= n-1 for --connected\n");
            return 1;
        }
        // Random permutation of vertices, then chain them
        vector<int> perm(n);
        iota(perm.begin(), perm.end(), 0);
        for (int i = n-1; i > 0; i--) {
            int j = rng() % (i+1);
            swap(perm[i], perm[j]);
        }
        for (int i = 0; i < n-1; i++) {
            edges.emplace_back(perm[i], perm[i+1], gen_weight());
        }
    }

    // Fill remaining edges randomly
    int remaining = m - (int)edges.size();
    for (int i = 0; i < remaining; i++) {
        int u = (int)(rng() % n);
        int v = (int)(rng() % n);
        if (u == v) v = (v + 1) % n;   // avoid self-loops
        edges.emplace_back(u, v, gen_weight());
    }

    // Shuffle so the spanning path edges aren't always at the front
    for (int i = (int)edges.size()-1; i > 0; i--) {
        int j = rng() % (i+1);
        swap(edges[i], edges[j]);
    }

    // Output
    printf("%d %d\n", n, m);
    for (auto& [u, v, w] : edges)
        printf("%d %d %lld\n", u, v, w);

    return 0;
}
