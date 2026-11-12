// bench_harness.h — shared benchmarking support for the MST binaries.
//
// Why this exists
// ---------------
// The benchmark used to time whole processes with a stopwatch in Python, so
// every measurement included reading the graph from a text file and printing
// the MST back out. On the 1m case that was ~18 s of text I/O wrapped around
// ~1 s of actual MST construction, which buried the thing being measured.
//
// This header lets each binary time its own MST computation — after parsing,
// before printing — and report that number on stderr, where run_bench.py picks
// it up. It also provides --quiet, so timing runs can skip the edge dump.
//
// Usage in a main():
//
//     int main(int argc, char** argv) {
//         bench::init(argc, argv);          // fast I/O + parse --quiet
//         ... read the graph ...
//         double t0 = bench::now_ms();
//         ... compute the MST ...
//         bench::report_time(bench::now_ms() - t0);
//         cout << "Total weight = " << total << "\n";
//         if (bench::print_edges()) { ... dump edges ... }
//     }
//
// The "Total weight = ..." line is always printed: run_bench.py's correctness
// check parses it, and it costs nothing.

#ifndef BENCH_HARNESS_H
#define BENCH_HARNESS_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

namespace bench {

// Set by init() when --quiet is passed.
inline bool g_quiet = false;

// Set by init() from --repeat K. Run the MST computation K times on one parse
// of the input, reporting each timing separately. Parsing the 1m graph costs
// ~1.7 s against an algorithm time of ~30 ms, so paying it once per trial
// limited the harness to 3 trials inside a sensible budget. Amortising it over
// K in-process repeats buys 30+ trials for the same wall clock.
//
// Caveat: repeats run with warm caches and a warm allocator, so the first one
// is discarded by convention (run_bench.py asks for K+1 and drops the first).
inline int g_repeat = 1;

// Turn off C-stream synchronisation and the cin/cout tie. Without this, C++
// streams do unbuffered character-at-a-time I/O to stay in step with stdio,
// which on libstdc++ costs 3-5x on input-heavy programs. boruvka.cpp and
// parallel_boruvka.cpp were previously missing this while every other binary
// had it, which handicapped the baseline in every published comparison.
inline void init(int argc, char** argv) {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--quiet") == 0) {
            g_quiet = true;
        } else if (std::strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            int k = std::atoi(argv[++i]);
            if (k > 0) g_repeat = k;
        }
    }
}

// Number of in-process repeats of the MST computation.
inline int repeats() { return g_repeat; }

// Monotonic clock, in milliseconds. steady_clock rather than system_clock so a
// clock adjustment mid-run cannot produce a negative interval.
inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Emit the measured algorithm time on stderr as a JSON line, matching the
// convention the instrumented binaries already use. stderr keeps it out of the
// MST output on stdout, so correctness checking is unaffected.
inline void report_time(double ms) {
    std::fprintf(stderr, "{\"type\":\"timing\",\"ms\":%.6f}\n", ms);
    std::fflush(stderr);
}

// False when --quiet was passed: skip the per-edge dump during timing runs.
inline bool print_edges() { return !g_quiet; }

}  // namespace bench

#endif  // BENCH_HARNESS_H
