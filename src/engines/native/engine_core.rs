// engine_core.rs  —  Rust graph-growth engine, identical C ABI to engine_core.cpp.
//
// Same Hamiltonian, same Metropolis moves, and the SAME splitmix64 RNG with the
// SAME draw order as the C++ engine — so for a given seed the Rust and C++ tracks
// produce BIT-IDENTICAL graphs, and both match the Numba reference's distribution.
//
// Build (memory_benchmark.py does this if `rustc` is present):
//   rustc -C opt-level=3 -C target-cpu=native --crate-type=cdylib \
//         engine_core.rs -o engine_core_rs.so
//
// Note: this is a faithful 1:1 port of engine_core.cpp. It was NOT compiled in the
// dev sandbox (no Rust toolchain there); compile it on the target box. The point
// of the Rust track is to confirm empirically that Rust ties C++ here — both lower
// to the same LLVM/native code and hit the same DRAM-latency wall — not to beat it.

use std::os::raw::c_void;
use std::time::Instant;

#[inline(always)]
fn sm_next(s: &mut u64) -> u64 {
    *s = s.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *s;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
#[inline(always)]
fn uniform(s: &mut u64) -> f64 {
    (sm_next(s) >> 11) as f64 * (1.0 / 9_007_199_254_740_992.0)
}
#[inline(always)]
// NB: `% n` carries the classic modulo bias. For n up to ~2^32 against a
// 64-bit draw the bias is < 2^-32 per call — far below the Monte-Carlo noise
// floor — so it is accepted; kept identical to engine_core.cpp because
// bit-identical draws between the two tracks are the contract.
fn below(s: &mut u64, n: i32) -> i32 {
    (sm_next(s) % (n as u64)) as i32
}

struct EngineImpl {
    n: usize,
    md: usize,
    max_degree: i32,
    nb: Vec<i32>,
    deg: Vec<i32>,
    peak: i32,
    capped: bool,
    s: u64,
}

#[inline(always)]
fn find_idx(nb: &[i32], deg: &[i32], md: usize, u: i32, v: i32) -> i32 {
    let d = deg[u as usize];
    let base = (u as usize) * md;
    let mut i = 0i32;
    while i < d {
        if nb[base + i as usize] == v {
            return i;
        }
        i += 1;
    }
    -1
}

// returns false if the max_degree cap was hit
fn sweep(e: &mut EngineImpl, steps: i64, mu: f64, t: f64, ec: f64, lb: f64, _mode: i32) -> bool {
    let md = e.md;
    let n = e.n as i32;
    let mx = e.max_degree;
    let mut peak = e.peak;
    let mut s = e.s;
    for _ in 0..steps {
        let (u, v);
        if uniform(&mut s) < lb {
            let k = below(&mut s, n);
            let cnt = e.deg[k as usize];
            if cnt < 2 {
                continue;
            }
            let base = (k as usize) * md;
            let i1 = below(&mut s, cnt);
            let mut i2 = below(&mut s, cnt - 1);
            if i2 >= i1 {
                i2 += 1;
            }
            u = e.nb[base + i1 as usize];
            v = e.nb[base + i2 as usize];
        } else {
            u = below(&mut s, n);
            v = below(&mut s, n);
            if u == v {
                continue;
            }
        }
        let i_uv = find_idx(&e.nb, &e.deg, md, u, v);
        let exists = i_uv != -1;
        let du = e.deg[u as usize];
        let dv = e.deg[v as usize];
        if !exists && (du >= mx || dv >= mx) {
            e.peak = peak;
            e.s = s;
            e.capped = true;
            return false;
        }
        let de = if exists {
            -ec + mu * ((2 - 2 * (du + dv)) as f64)
        } else {
            ec + mu * ((2 + 2 * (du + dv)) as f64)
        };
        let accept = de <= 0.0 || (t > 0.0 && uniform(&mut s) < (-de / t).exp());
        if !accept {
            continue;
        }
        if exists {
            let ub = (u as usize) * md;
            let last = e.deg[u as usize] - 1;
            e.nb[ub + i_uv as usize] = e.nb[ub + last as usize];
            e.nb[ub + last as usize] = -1;
            e.deg[u as usize] = last;
            let j = find_idx(&e.nb, &e.deg, md, v, u);
            let vb = (v as usize) * md;
            let last = e.deg[v as usize] - 1;
            e.nb[vb + j as usize] = e.nb[vb + last as usize];
            e.nb[vb + last as usize] = -1;
            e.deg[v as usize] = last;
        } else {
            let ub = (u as usize) * md;
            let du2 = e.deg[u as usize];
            e.nb[ub + du2 as usize] = v;
            e.deg[u as usize] = du2 + 1;
            let vb = (v as usize) * md;
            let dv2 = e.deg[v as usize];
            e.nb[vb + dv2 as usize] = u;
            e.deg[v as usize] = dv2 + 1;
            if e.deg[u as usize] > peak {
                peak = e.deg[u as usize];
            }
            if e.deg[v as usize] > peak {
                peak = e.deg[v as usize];
            }
        }
    }
    e.peak = peak;
    e.s = s;
    true
}

#[no_mangle]
pub extern "C" fn engine_create(n: i32, max_degree: i32, seed: u64) -> *mut c_void {
    let md = max_degree as usize;
    let e = Box::new(EngineImpl {
        n: n as usize,
        md,
        max_degree,
        nb: vec![-1i32; (n as usize) * md],
        deg: vec![0i32; n as usize],
        peak: 0,
        capped: false,
        s: if seed != 0 { seed } else { 0x9E37_79B9_7F4A_7C15 },
    });
    Box::into_raw(e) as *mut c_void
}

#[no_mangle]
pub extern "C" fn engine_warm(h: *mut c_void, steps: i64, mu: f64, t: f64, ec: f64, lb: f64, mode: i32) -> i32 {
    let e = unsafe { &mut *(h as *mut EngineImpl) };
    if sweep(e, steps, mu, t, ec, lb, mode) { 0 } else { -1 }
}

#[no_mangle]
pub extern "C" fn engine_measure(h: *mut c_void, steps: i64, mu: f64, t: f64, ec: f64, lb: f64, mode: i32) -> f64 {
    let e = unsafe { &mut *(h as *mut EngineImpl) };
    let t0 = Instant::now();
    let ok = sweep(e, steps, mu, t, ec, lb, mode);
    let dt = t0.elapsed().as_secs_f64();
    if ok { dt } else { -1.0 }
}

#[no_mangle]
pub extern "C" fn engine_stats(h: *mut c_void, out_edges: *mut i64, out_peak: *mut i32) {
    let e = unsafe { &mut *(h as *mut EngineImpl) };
    let mut edges: i64 = 0;
    for i in 0..e.n {
        edges += e.deg[i] as i64;
    }
    unsafe {
        if !out_edges.is_null() { *out_edges = edges / 2; }
        if !out_peak.is_null() { *out_peak = e.peak; }
    }
}

#[no_mangle]
pub extern "C" fn engine_export(h: *mut c_void, nb_out: *mut i32, deg_out: *mut i32) {
    let e = unsafe { &mut *(h as *mut EngineImpl) };
    unsafe {
        if !nb_out.is_null() {
            std::ptr::copy_nonoverlapping(e.nb.as_ptr(), nb_out, e.nb.len());
        }
        if !deg_out.is_null() {
            std::ptr::copy_nonoverlapping(e.deg.as_ptr(), deg_out, e.deg.len());
        }
    }
}

#[no_mangle]
pub extern "C" fn engine_free(h: *mut c_void) {
    if h.is_null() {
        return;
    }
    unsafe {
        drop(Box::from_raw(h as *mut EngineImpl));
    }
}

#[no_mangle]
pub extern "C" fn growth_bytes(n: i32, max_degree: i32) -> i64 {
    (n as i64) * (max_degree as i64) * 4 + (n as i64) * 4
}

#[no_mangle]
pub extern "C" fn run_growth(n: i32, warm_steps: i64, measure_steps: i64, seed: u64,
                             max_degree: i32, mu: f64, t: f64, ec: f64, lb: f64,
                             mode: i32, out_edges: *mut i64, out_peak: *mut i32) -> f64 {
    let h = engine_create(n, max_degree, seed);
    if h.is_null() {
        return -2.0;
    }
    let mut dt = -1.0;
    if engine_warm(h, warm_steps, mu, t, ec, lb, mode) == 0 {
        dt = engine_measure(h, measure_steps, mu, t, ec, lb, mode);
    }
    engine_stats(h, out_edges, out_peak);
    engine_free(h);
    dt
}
