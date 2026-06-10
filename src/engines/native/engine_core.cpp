// engine_core.cpp  —  C++ graph-growth engine, C-ABI shared library.
//
// Speed/track counterpart to the Numba reference (engines/numba/kernel.py): SAME
// Hamiltonian (H = edge_cost·E + degree_penalty·Σd²) and SAME Metropolis move
// mix — local triangle move with probability lb, uniform global pair otherwise
// (lb is the runtime locality_bias parameter, NOT a hard-coded 90/10 split) —
// so the two tracks are statistically comparable at any lb.
//
// C ABI (shared with engine_core.rs; the Python loader in engines/native/loader.py
// binds exactly these symbols):
//     void*   engine_create(N, max_degree, seed)
//     int     engine_warm   (h, steps, mu,T,ec,lb,mode)      -> 0 / -1 capped
//     double  engine_measure(h, steps, mu,T,ec,lb,mode)      -> seconds / -1 capped
//     void    engine_stats  (h, &edges, &peak)
//     void    engine_export (h, nb_out, deg_out)             -> copy graph into NumPy-shaped buffers
//     void    engine_free   (h)
//     int64   growth_bytes  (N, max_degree)                  -> per-worker footprint
//     double  run_growth(...)  convenience: create+warm+measure+stats+free
//
// NOTE on how memory_benchmark.py actually uses this: it grows ONE graph from empty
// to equilibrium and times that whole growth EXTERNALLY (Python perf_counter around
// Engine.step(), which maps to engine_warm()). It deliberately does NOT use the
// internal warm-then-measure split — for a time-to-equilibrium measurement a warm
// pass would measure steady-state churn, not the one-time growth. engine_measure()
// and run_growth() are therefore convenience entry points that the benchmark does
// not call (engine_measure is bound by the loader but unused; run_growth is unused).
//
// mode 0 = tight scalar loop; mode 1 = + intra-step prefetch of the 2nd endpoint
// (overlaps the two endpoint cache-misses; bit-identical output to mode 0).
//
// Build (memory_benchmark.py / loader.py does this automatically when the source is
// newer than the .so; build.sh is the manual path):
//   g++ -O3 -march=native -funroll-loops -shared -fPIC engine_core.cpp -o engine_core.so

#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <chrono>
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace {
// splitmix64 (NOT xoshiro): the 0x9E3779B97F4A7C15 increment + the two
// multiply-shift finalizers are textbook splitmix64. Named to match engine_core.rs
// (which calls it splitmix64); same constants and SAME draw order, which is what
// makes the C++ and Rust tracks bit-identical for a given seed.
struct SplitMix64 {
    uint64_t s;
    inline uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    inline double uniform() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    // NB: `% n` carries the classic modulo bias. For n up to ~2^32 against a
    // 64-bit draw the bias is < 2^-32 per call — orders of magnitude below the
    // Monte-Carlo noise floor here — so it is accepted; kept identical in the
    // Rust track (engine_core.rs) because bit-identical draws are the contract.
    inline int32_t below(int32_t n) { return (int32_t)(next() % (uint64_t)n); }
};

struct Engine {
    int32_t  N;
    int64_t  MD;
    int32_t  max_degree;
    int32_t* nb;
    int32_t* deg;
    int32_t  peak;
    bool     capped;
    SplitMix64  rng;
};

inline int find_idx(const int32_t* __restrict nb, const int32_t* __restrict deg,
                    int64_t MD, int32_t u, int32_t v) {
    const int32_t* row = nb + (int64_t)u * MD;
    int d = deg[u];
    for (int i = 0; i < d; ++i) if (row[i] == v) return i;
    return -1;
}

// returns false if the max_degree cap was hit
bool sweep(Engine* e, int64_t steps, double mu, double T, double ec, double lb, int mode) {
    int32_t* __restrict nb  = e->nb;
    int32_t* __restrict deg = e->deg;
    const int64_t MD = e->MD;
    const int32_t N  = e->N;
    const int32_t MX = e->max_degree;
    SplitMix64& rng = e->rng;
    int32_t peak = e->peak;
    for (int64_t t = 0; t < steps; ++t) {
        int32_t u, v;
        if (rng.uniform() < lb) {
            int32_t k = rng.below(N);
            int cnt = deg[k];
            if (cnt < 2) continue;
            const int32_t* row = nb + (int64_t)k * MD;
            int i1 = rng.below(cnt);
            int i2 = rng.below(cnt - 1); if (i2 >= i1) ++i2;
            u = row[i1]; v = row[i2];
        } else {
            u = rng.below(N);
            v = rng.below(N);
            if (u == v) continue;
        }
        if (mode == 1) {
            __builtin_prefetch(nb + (int64_t)v * MD, 1, 1);
            __builtin_prefetch(deg + v, 1, 1);
        }
        int i_uv = find_idx(nb, deg, MD, u, v);
        bool exists = (i_uv != -1);
        int du = deg[u], dv = deg[v];
        if (!exists && (du >= MX || dv >= MX)) { e->peak = peak; e->capped = true; return false; }
        double dE = exists ? (-ec + mu * (2 - 2 * (du + dv)))
                           : ( ec + mu * (2 + 2 * (du + dv)));
        bool accept = (dE <= 0.0) || (T > 0.0 && rng.uniform() < std::exp(-dE / T));
        if (!accept) continue;
        if (exists) {
            int last = deg[u] - 1;
            nb[(int64_t)u * MD + i_uv] = nb[(int64_t)u * MD + last];
            nb[(int64_t)u * MD + last] = -1; deg[u] = last;
            int j = find_idx(nb, deg, MD, v, u);
            last = deg[v] - 1;
            nb[(int64_t)v * MD + j]    = nb[(int64_t)v * MD + last];
            nb[(int64_t)v * MD + last] = -1; deg[v] = last;
        } else {
            nb[(int64_t)u * MD + deg[u]] = v; ++deg[u];
            nb[(int64_t)v * MD + deg[v]] = u; ++deg[v];
            if (deg[u] > peak) peak = deg[u];
            if (deg[v] > peak) peak = deg[v];
        }
    }
    e->peak = peak;
    return true;
}
} // namespace

extern "C" {

void* engine_create(int32_t N, int32_t max_degree, uint64_t seed) {
    Engine* e = (Engine*)malloc(sizeof(Engine));
    if (!e) return nullptr;
    e->N = N; e->MD = max_degree; e->max_degree = max_degree;
    e->peak = 0; e->capped = false;
    e->rng.s = seed ? seed : 0x9E3779B97F4A7C15ull;
    size_t nb_bytes  = (size_t)N * max_degree * sizeof(int32_t);
    size_t deg_bytes = (size_t)N * sizeof(int32_t);
    e->nb  = (int32_t*)malloc(nb_bytes);
    e->deg = (int32_t*)malloc(deg_bytes);
    if (!e->nb || !e->deg) { free(e->nb); free(e->deg); free(e); return nullptr; }
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    madvise(e->nb, nb_bytes, MADV_HUGEPAGE);
#endif
    memset(e->nb, 0xFF, nb_bytes);   // int32 -1
    memset(e->deg, 0, deg_bytes);
    return e;
}

int engine_warm(void* h, int64_t steps, double mu, double T, double ec, double lb, int mode) {
    Engine* e = (Engine*)h;
    return sweep(e, steps, mu, T, ec, lb, mode) ? 0 : -1;
}

double engine_measure(void* h, int64_t steps, double mu, double T, double ec, double lb, int mode) {
    Engine* e = (Engine*)h;
    auto t0 = std::chrono::steady_clock::now();
    bool ok = sweep(e, steps, mu, T, ec, lb, mode);
    auto t1 = std::chrono::steady_clock::now();
    if (!ok) return -1.0;
    return std::chrono::duration<double>(t1 - t0).count();
}

void engine_stats(void* h, int64_t* out_edges, int32_t* out_peak) {
    Engine* e = (Engine*)h;
    int64_t edges = 0;
    for (int32_t i = 0; i < e->N; ++i) edges += e->deg[i];
    if (out_edges) *out_edges = edges / 2;
    if (out_peak)  *out_peak  = e->peak;
}

// Copy the grown graph into caller-provided buffers laid out EXACTLY like the
// Python engine's arrays: nb_out is N*max_degree int32 (row-major, -1 padded),
// deg_out is N int32. Lets the benchmark hand the C++/Rust-grown graph back to
// the rest of the pipeline in the identical construct the Numba engine produces.
void engine_export(void* h, int32_t* nb_out, int32_t* deg_out) {
    Engine* e = (Engine*)h;
    if (nb_out)  memcpy(nb_out,  e->nb,  (size_t)e->N * e->MD * sizeof(int32_t));
    if (deg_out) memcpy(deg_out, e->deg, (size_t)e->N * sizeof(int32_t));
}

void engine_free(void* h) {
    if (!h) return;
    Engine* e = (Engine*)h;
    free(e->nb); free(e->deg); free(e);
}

int64_t growth_bytes(int32_t N, int32_t max_degree) {
    return (int64_t)N * max_degree * (int64_t)sizeof(int32_t) + (int64_t)N * (int64_t)sizeof(int32_t);
}

double run_growth(int32_t N, int64_t warm_steps, int64_t measure_steps, uint64_t seed,
                  int32_t max_degree, double mu, double T, double ec, double lb,
                  int mode, int64_t* out_edges, int32_t* out_peak) {
    void* h = engine_create(N, max_degree, seed);
    if (!h) return -2.0;
    double dt = -1.0;
    if (engine_warm(h, warm_steps, mu, T, ec, lb, mode) == 0)
        dt = engine_measure(h, measure_steps, mu, T, ec, lb, mode);
    engine_stats(h, out_edges, out_peak);
    engine_free(h);
    return dt;
}

} // extern "C"
