import os
import glob
import numpy as np
import datetime
import argparse
import time
import re
import platform
import warnings

# --- FIX: Handle NumPy version differences for RankWarning ---
try:
    from numpy.exceptions import RankWarning as PolyRankWarning
    warnings.simplefilter('ignore', PolyRankWarning)
except ImportError:
    try:
        warnings.simplefilter('ignore', np.RankWarning)
    except AttributeError:
        pass

# --- CONFIG ---
RUNS_DIR = "data"           # Pointing to 'data' folder
MIN_TIMEOUT = 60.0          # Minimum timeout (seconds) even if updates are very fast
FALLBACK_EXP_SPEED = -2.0   # Assume speed drops as N^-2
FALLBACK_EXP_STEPS = 1.0    # Assume steps grow linearly

def get_cpu_name():
    try:
        if platform.system() == "Windows": return platform.processor()
        elif platform.system() == "Darwin": return "Apple Silicon / Intel"
        with open('/proc/cpuinfo') as f:
            for line in f:
                if "model name" in line: return line.split(':', 1)[1].strip()
    except: pass
    return "Unknown CPU"

def fmt_duration(seconds):
    if seconds is None or np.isnan(seconds) or seconds < 0: return "-"
    if seconds < 60: return f"{int(seconds)}s"
    if seconds < 3600: return f"{int(seconds//60)}m {int(seconds%60)}s"
    return f"{int(seconds//3600)}h {int((seconds%3600)//60)}m"

def fmt_ago(timestamp):
    delta = time.time() - timestamp
    if delta < 60: return "now"
    if delta < 3600: return f"+{int(delta//60)}m"
    return f"+{int(delta//3600)}h {int((delta%3600)//60)}m"

def fit_power_law(x_vals, y_vals):
    """Fits y = c * x^m via log-log regression. Returns (c, m)."""
    if len(x_vals) < 2: return None
    try:
        x_in, y_in = [], []
        for x, y in zip(x_vals, y_vals):
            if x > 0 and y > 0:
                x_in.append(x); y_in.append(y)
        if len(x_in) < 2: return None

        coeffs = np.polyfit(np.log(x_in), np.log(y_in), 1)
        return np.exp(coeffs[1]), coeffs[0]
    except: return None

def get_run_metrics(log_file):
    """Parses a SINGLE continuous log file (S{seed}_log.csv)."""
    if not os.path.exists(log_file): return None

    # 1. READ FILE
    try:
        with open(log_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
    except: return None

    if not lines: return None

    # 2. HEADER DETECTION
    idx = None
    # Default heuristic (step at 1, elapsed at 3, tri at 7, edges at 8)
    default_idx = {'step': 1, 'elapsed_sec': 3, 'triangles': 7, 'edges': 8, 'k_min': 4, 'k_avg': 5, 'k_max': 6}

    # Scan first few lines for header
    for line in lines[:5]:
        lower_line = line.lower()
        if ('step' in lower_line or 'iter' in lower_line) and 'elapsed' in lower_line:
            parts = [h.strip().lower() for h in line.split(',')]
            idx = {name: i for i, name in enumerate(parts)}

            # Map aliases
            if 'iter' in idx and 'step' not in idx:
                idx['step'] = idx['iter']
            break

    # Fallback / Heuristic for "Shifted" CSV
    if not idx:
        parts = lines[0].split(',')
        if len(parts) > 8 and parts[0].isdigit() and 'T' in parts[1]:
             # Shifted format
             idx = {'step': 2, 'elapsed_sec': 3, 'triangles': 8, 'edges': 9, 'k_min': 5, 'k_avg': 6, 'k_max': 7}
        else:
             idx = default_idx

    # 3. PARSE DATA
    data_lines = [l for l in lines if not l.startswith('#') and ',' in l and 'elapsed' not in l.lower()]
    if not data_lines: return None

    # --- INTELLIGENT COLUMN MAPPING ---
    if idx and 'triangles' in idx and 'edges' not in idx:
        tri_idx = idx['triangles']
        first_data = data_lines[0].split(',')
        if len(first_data) > tri_idx + 1:
            idx['edges'] = tri_idx + 1

    k_max_idx = idx.get('k_max', 6)
    edge_idx = idx.get('edges')
    tri_idx = idx.get('triangles')

    global_k_peak = 0.0
    global_edge_peak = 0.0
    global_tri_peak = 0.0
    is_done = False

    if any("COMPLETED" in l for l in lines[-5:]):
        is_done = True

    # Scan for Peaks
    for row_str in data_lines:
        try:
            parts = row_str.split(',')

            # K Peak
            if len(parts) > k_max_idx:
                val = float(parts[k_max_idx])
                if val > global_k_peak: global_k_peak = val

            # Edge Peak
            if edge_idx is not None and len(parts) > edge_idx:
                e_val = float(parts[edge_idx])
                if e_val > global_edge_peak: global_edge_peak = e_val

            # Triangle Peak
            if tri_idx is not None and len(parts) > tri_idx:
                t_val = float(parts[tri_idx])
                if t_val > global_tri_peak: global_tri_peak = t_val

        except: pass

    # Get Last State
    d_last = data_lines[-1].split(',')

    def get_val(row, name, default=0.0):
        i = idx.get(name)
        if i is not None and i < len(row):
            try: return float(row[i])
            except: pass
        return default

    t_curr = get_val(d_last, 'elapsed_sec')
    s_curr = int(get_val(d_last, 'step'))

    sps_avg = (s_curr / t_curr) if t_curr > 0.1 else 0.0

    # Instantaneous SPS
    sps_inst = sps_avg
    lookback = min(len(data_lines) - 1, 10)
    if lookback > 0:
        d_prev = data_lines[-(lookback+1)].split(',')
        t_prev = get_val(d_prev, 'elapsed_sec')
        s_prev = int(get_val(d_prev, 'step'))
        dt = t_curr - t_prev
        ds = s_curr - s_prev
        if dt > 1e-9: sps_inst = ds / dt

    k_stats = (get_val(d_last, 'k_min'), get_val(d_last, 'k_avg'), get_val(d_last, 'k_max'))

    tri_last = int(get_val(d_last, 'triangles'))
    edges_last = int(get_val(d_last, 'edges'))

    # Ensure peaks are at least the last value
    global_edge_peak = max(global_edge_peak, edges_last)
    global_tri_peak = max(global_tri_peak, tri_last)

    return {
        'iter': s_curr,
        'time': t_curr,
        'sps': sps_avg,
        'sps_inst': sps_inst,
        'k': k_stats,
        'k_peak': max(global_k_peak, k_stats[2]),
        'tri': tri_last,
        'tri_peak': global_tri_peak,
        'edges': edges_last,
        'edges_peak': global_edge_peak,
        'done': is_done,
        'mtime': os.path.getmtime(log_file),
        'avg_dt': 0
    }

def analyze_all(target_engine=None):
    if not os.path.exists(RUNS_DIR): return [], []

    engine_data = {}
    engines = [d for d in os.listdir(RUNS_DIR) if d.startswith('E')]
    if target_engine: engines = [target_engine]

    engines.sort(key=lambda x: int(x[1:].split('D')[0]) if x[1:].split('D')[0].isdigit() else 0)

    for eng in engines:
        engine_data[eng] = {}
        eng_path = os.path.join(RUNS_DIR, eng)

        for n_dir_name in os.listdir(eng_path):
            if not n_dir_name.startswith("N"): continue
            try:
                N = int(n_dir_name[1:])
                n_path = os.path.join(eng_path, n_dir_name)

                runs = []
                log_pattern = os.path.join(n_path, "S*_log.csv")
                for log_file in glob.glob(log_pattern):
                    r = get_run_metrics(log_file)
                    if r: runs.append(r)

                if runs: engine_data[eng][N] = runs
            except: continue

    models = {}
    for eng, n_map in engine_data.items():
        if not n_map: continue

        all_N, all_sps = [], []
        comp_N, comp_steps = [], []

        for N, runs in n_map.items():
            for r in runs:
                if r['sps'] > 0:
                    all_N.append(N); all_sps.append(r['sps'])
            for r in runs:
                if r['done']:
                    comp_N.append(N); comp_steps.append(r['iter'])

        speed_fit = fit_power_law(all_N, all_sps)
        if not speed_fit:
            if all_sps:
                c = np.mean(all_sps) * (np.mean(all_N) ** -FALLBACK_EXP_SPEED)
                speed_fit = (c, FALLBACK_EXP_SPEED)
            else:
                speed_fit = (1.0, FALLBACK_EXP_SPEED)

        step_fit = fit_power_law(comp_N, comp_steps)
        if not step_fit or step_fit[1] < 0:
            step_fit = (None, None)

        models[eng] = {'eng': eng, 'speed': speed_fit, 'steps': step_fit, 'max_n': max(n_map.keys())}

    rows = []
    for eng, n_map in engine_data.items():
        if not n_map: continue

        sorted_Ns = sorted(n_map.keys())
        for N in sorted_Ns:
            runs = n_map[N]

            speeds = [r['sps'] for r in runs if r['sps'] > 0]
            avg_sps = np.mean(speeds) if speeds else 0

            completed_runs = [r for r in runs if r['done']]
            latest_run = max(runs, key=lambda x: x['mtime'])

            timeout_thresh = 120.0
            time_since_activity = time.time() - latest_run['mtime']
            is_active_group = (not latest_run['done']) and (time_since_activity < timeout_thresh)

            if is_active_group:
                active_candidates = [r for r in runs if not r['done'] and (time.time()-r['mtime'] < timeout_thresh)]
                if active_candidates:
                    primary_run = max(active_candidates, key=lambda x: x['iter'])
                else:
                    primary_run = latest_run

                curr_iter = primary_run['iter']
                curr_sps = primary_run.get('sps_inst', primary_run['sps'])
                display_active_sps = curr_sps
            else:
                primary_run = latest_run
                curr_iter = primary_run['iter']
                curr_sps = 0.0
                display_active_sps = 0.0

            predicted_goal = None
            avg_goal = None
            model_goal = None

            if completed_runs:
                avg_goal = np.mean([r['iter'] for r in completed_runs])

            if eng in models:
                step_model = models[eng]['steps']
                if step_model[0] is not None:
                    c, d = step_model
                    model_goal = c * (N**d)

            if avg_goal:
                if curr_iter < avg_goal:
                    predicted_goal = avg_goal
                else:
                    if model_goal and model_goal > curr_iter:
                        predicted_goal = model_goal
                    else:
                        predicted_goal = avg_goal
            elif model_goal:
                predicted_goal = model_goal

            if is_active_group:
                if predicted_goal and curr_sps > 0:
                    pct = (curr_iter / predicted_goal) * 100
                    if pct < 100:
                        rem = predicted_goal - curr_iter
                        sec_rem = rem / curr_sps
                        status_str = f"~{fmt_duration(sec_rem)} ({int(pct)}%)"
                    else:
                        status_str = f"> Est ({int(pct)}%)"
                else:
                    status_str = "Active"
            else:
                status_str = f"seen {fmt_ago(latest_run['mtime'])}"

            if latest_run['done'] and len(completed_runs) == len(runs):
                 status_str = "Done"

            source_runs = completed_runs if completed_runs else runs
            phys_runs = [r for r in source_runs if r.get('k')]

            k_str, tri_str, edge_str = "-", "-", "-"
            if phys_runs:
                k_mins = [r['k'][0] for r in phys_runs]
                k_avgs = [r['k'][1] for r in phys_runs]
                k_ends = [r['k'][2] for r in phys_runs]
                k_peaks = [r.get('k_peak', r['k'][2]) for r in phys_runs]

                # Triangle Stats
                tris = [r['tri'] for r in phys_runs]
                tri_peaks = [r.get('tri_peak', r['tri']) for r in phys_runs]

                # Edge Stats
                edges_list = [r.get('edges', 0) for r in phys_runs]
                edges_peaks = [r.get('edges_peak', r.get('edges', 0)) for r in phys_runs]

                # K String
                peak_disp = max(k_peaks)
                k_str = f"{np.mean(k_mins):.0f}/{np.mean(k_avgs):.4f}/{np.mean(k_ends):.0f}/{peak_disp:.0f}"
                if len(k_peaks) > 1 and len(completed_runs) > 1:
                    k_str += f" \u00B1{int(np.std(k_peaks))}"

                # --- Edges String Construction ---
                edge_mean = int(np.mean(edges_list))
                edge_peak = int(max(edges_peaks))
                edge_str = f"{edge_mean:,}"

                # Always add SD if multiple runs exist
                if len(edges_list) > 1 and len(completed_runs) > 1:
                    edge_str += f" \u00B1{int(np.std(edges_list))}"

                # Append peak if it exceeds mean
                if edge_peak > edge_mean:
                     edge_str += f" / {edge_peak:,}"

                # --- Triangles String Construction ---
                tri_mean = int(np.mean(tris))
                tri_peak = int(max(tri_peaks))
                tri_str = f"{tri_mean:,}"

                # Always add SD if multiple runs exist
                if len(tris) > 1 and len(completed_runs) > 1:
                    tri_str += f" \u00B1{int(np.std(tris))}"

                # Append peak if it exceeds mean
                if tri_peak > tri_mean:
                    tri_str += f" / {tri_peak:,}"

            # "Avg Stability" expresses run-to-run variance of iteration counts.
            # Show relative standard deviation (percentage) rather than absolute +/-.
            avg_steps_str = "-"
            if completed_runs:
                steps_list = [r['iter'] for r in completed_runs]
                s_mean = float(np.mean(steps_list))
                avg_steps_str = f"{int(s_mean):,}"
                if len(steps_list) > 1 and s_mean > 0:
                    s_std = float(np.std(steps_list))
                    rsd_pct = (s_std / s_mean) * 100.0
                    avg_steps_str += f" \u00B1{rsd_pct:.1f}%"

            max_time = max(r['time'] for r in runs)

            rows.append({
                'eng': eng, 'N': N,
                'seeds': f"{len(completed_runs)}/{len(runs)}",
                'hz': f"{1/avg_sps:.4f}s" if avg_sps > 0 else "-",
                'sps_curr': display_active_sps,
                'sps_avg': avg_sps,
                'time_raw': max_time,
                'time_lbl': fmt_duration(max_time),
                'curr': f"{curr_iter:,}",
                'avg_steps': avg_steps_str,
                'k_metrics': k_str,
                'tri': tri_str,
                'edges': edge_str,
                'status': status_str
            })

    return rows, list(models.values())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", type=str, default=None)
    args = parser.parse_args()

    print(f"--- BENCHMARK REPORT | CPU: {get_cpu_name()} ---")
    rows, models = analyze_all(args.engine)

    if not rows:
        print(f"No run data found in '{RUNS_DIR}/'.")
        return

    #rows.sort(key=lambda x: (x['N'], -x['sps_avg']))
    rows.sort(key=lambda x: (x['N'], x['time_raw']))

    # Adjusted column widths for potentially longer strings (Avg + Std + Peak)
    # Column order tweak: put current iteration right after current i/s, then avg i/s.
    H_FMT = "{:<11} {:<6} {:<6} {:<10} {:<14} {:<10} {:<10} {:<26} {:<26} {:<22} {:<24} {:<18}"
    D_FMT = "{:<11} {:<6} {:<6} {:<10} {:<14} {:<10.0f} {:<10} {:<26} {:<26} {:<22} {:<24} {:<18}"

    print("-" * 185)
    print(H_FMT.format("N", "Conf", "Seeds", "Cur i/s", "Cur Iter", "Avg i/s", "Avg Time", "Avg Stability", "k(min/avg/max/peak)", "Edges (Avg/Peak)", "Triangles (Avg/Peak)", "Status"))
    print("-" * 185)

    curr_n = None
    for r in rows:
        if curr_n and r['N'] != curr_n: print("")
        cur_str = f"{r['sps_curr']:.0f}" if r['sps_curr'] > 0 else "-"
        print(D_FMT.format(
            f"N{r['N']:_}", r['eng'], r['seeds'],
            cur_str, r['curr'], r['sps_avg'],
            r['time_lbl'], r['avg_steps'],
            r['k_metrics'],
            r['edges'], r['tri'],
            r['status']
        ))
        curr_n = r['N']

    if not models: return

    global_max_N = max(m['max_n'] for m in models)
    targets = [global_max_N * 2, global_max_N * 4]

    print("\n" + "="*80)
    print(f" FORECAST (Based on Max N={global_max_N:,})")
    print("="*80)
    F_FMT = "{:<4} {:<6} {:<16} {:<15} {:<18} {:<15}"

    for T in targets:
        print(f"\n>> TARGET N = {T:,}")
        print("-" * 80)
        print(F_FMT.format("Rank", "Eng", "Est Time", "Speed (i/s)", "Total Iters", "Scaling (Iter)"))

        rank = []
        for m in models:
            if not m.get('speed'): continue
            a, b = m['speed']
            c, d = m['steps'] if m['steps'][0] else (None, None)
            pred_sps = a * (T**b)

            if c is not None:
                pred_steps = c * (T**d)
                total_sec = pred_steps / pred_sps
                rank.append((m['eng'], total_sec, pred_sps, pred_steps, d))

        rank.sort(key=lambda x: x[1])

        for i, (eng, sec, sps, steps, step_exp) in enumerate(rank):
            print(F_FMT.format(
                i+1, eng, fmt_duration(sec), f"{sps:.0f}", f"{int(steps):,}", f"O(N^{step_exp:.2f})"
            ))
        if not rank: print("   (Insufficient completion data to forecast)")

if __name__ == "__main__":
    main()
