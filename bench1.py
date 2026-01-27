import os
import glob
import numpy as np
import datetime
import re
import platform
import subprocess
import sys
import argparse
import time

# --- CONFIG ---
RUNS_DIR = "runs"
FALLBACK_COMPLEXITY_EXPONENT = 2.0  # O(N^2) assume for speed
FALLBACK_PHYSICS_EXPONENT = 1.5     # O(N^1.5) assume for steps if no data
ACTIVITY_THRESHOLD_SECONDS = 300    # 5 minutes to consider a run "Active"
DEFAULT_DISPLAY_TIMEOUT = 10.0      # 10 seconds inactivity -> switch to Last Active
STITCH_SAFETY_MARGIN = 2.0          # Multiplier for adaptive timeout

def get_cpu_name():
    """Attempts to retrieve the actual CPU model name."""
    try:
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r') as f:
                for line in f:
                    if "model name" in line:
                        return line.split(':', 1)[1].strip()
        elif sys.platform == "darwin":
            command = ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"]
            return subprocess.check_output(command).strip().decode()
        elif sys.platform == "win32":
            return platform.processor()
    except Exception:
        pass
    return platform.machine() or "Unknown CPU"

def get_available_engines():
    """Scans the runs directory for Engine versions (E*)."""
    if not os.path.exists(RUNS_DIR):
        return []

    entries = os.listdir(RUNS_DIR)
    engines = [e for e in entries if e.startswith('E') and os.path.isdir(os.path.join(RUNS_DIR, e))]

    def sort_key(x):
        try:
            return int(x[1:])
        except:
            return x

    return sorted(engines, key=sort_key)

def parse_metadata(line):
    meta = {}
    clean_line = line.lstrip('#').strip()
    parts = clean_line.split(',')
    for p in parts:
        if '=' in p:
            k, v = p.split('=', 1)
            meta[k.strip()] = v.strip()
    return meta

def get_fs_artifacts_stats(target_engine="*"):
    """Scans filesystem for rough estimates based on file timestamps/names."""
    artifacts = {}
    search_path = os.path.join(RUNS_DIR, target_engine, "*", "*", "data", "*_nodes.csv")
    files = glob.glob(search_path)

    run_groups = {}
    for fpath in files:
        dirname = os.path.dirname(fpath)
        if dirname not in run_groups: run_groups[dirname] = []
        run_groups[dirname].append(fpath)

    for rdir, fpaths in run_groups.items():
        if len(fpaths) < 2: continue

        try:
            parts = rdir.split(os.sep)
            n_part = [p for p in parts if p.startswith('N') and p[1:].isdigit()][0]
            N = int(n_part[1:])
            s_part = [p for p in parts if p.startswith('S') and p[1:].isdigit()][0]
            seed = int(s_part[1:])
        except:
            continue

        unique_key = f"N{N}_S{seed}"
        snapshots = []
        for fp in fpaths:
            try:
                fname = os.path.basename(fp)
                match = re.search(r"iter_(\d+)_", fname)
                if match:
                    step = int(match.group(1).replace('_', ''))
                    mtime = os.path.getmtime(fp)
                    snapshots.append((step, mtime))
            except: pass

        snapshots.sort(key=lambda x: x[0])
        if not snapshots: continue

        first_step, first_time = snapshots[0]
        last_step, last_time = snapshots[-1]
        delta_steps = last_step - first_step
        delta_time = last_time - first_time

        # Calculate update interval based on recent history
        update_interval = 0.0
        if len(snapshots) >= 2:
            s_last = snapshots[-1]
            s_prev = snapshots[-2]
            update_interval = s_last[1] - s_prev[1]

        if delta_steps > 0 and delta_time > 0.1:
            sec_per_step = delta_time / delta_steps
            artifacts[unique_key] = {
                'N': N,
                'seed': seed,
                'sec_per_step': sec_per_step,
                'max_step': last_step,
                'is_completed': False,
                'last_update': last_time,
                'total_time_elapsed': delta_time,
                'update_interval': update_interval,
                'source': 'filesystem',
                'physics': None
            }
    return artifacts

def get_log_stats(target_engine="*"):
    """Scans logs for precise timing, completion status, and physics metrics."""
    stats = {}
    search_path = os.path.join(RUNS_DIR, target_engine, "**", "logs", "*.csv")
    log_files = glob.glob(search_path, recursive=True)

    for lf in log_files:
        try:
            mtime = os.path.getmtime(lf)
            with open(lf, 'r') as f:
                lines = f.readlines()
            if len(lines) < 3: continue

            meta = parse_metadata(lines[0])
            if 'N' not in meta or 'seed' not in meta: continue
            N = int(meta['N'])
            seed = int(meta['seed'])
            unique_key = f"N{N}_S{seed}"

            is_completed = any(l.startswith("# COMPLETED") for l in lines[-3:])

            # Parse Header
            header = lines[1].strip().split(',')
            try:
                idx_step = header.index('step')
                idx_ts = header.index('elapsed_sec')
                idx_kmin = header.index('k_min')
                idx_kavg = header.index('k_avg')
                idx_kmax = header.index('k_max')
                idx_tri = header.index('triangles')
            except ValueError:
                idx_step = 1; idx_ts = 3; idx_kmin = 4; idx_kavg = 5; idx_kmax = 6; idx_tri = 7

            # Parse Data
            data_lines = [l for l in lines if not l.startswith('#') and ',' in l]
            if len(data_lines) < 2: continue

            first_data = data_lines[1].strip().split(',')
            last_data = data_lines[-1].strip().split(',')

            t_start = float(first_data[idx_ts])
            t_end = float(last_data[idx_ts])
            step_start = int(first_data[idx_step])
            step_end = int(last_data[idx_step])

            # Calculate Update Interval from last two entries
            update_interval = 0.0
            if len(data_lines) >= 3:
                prev_data = data_lines[-2].strip().split(',')
                try:
                    t_prev = float(prev_data[idx_ts])
                    update_interval = t_end - t_prev
                except: pass

            physics = {
                'k_min': int(last_data[idx_kmin]),
                'k_avg': float(last_data[idx_kavg]),
                'k_max': int(last_data[idx_kmax]),
                'tri': int(last_data[idx_tri])
            }

            delta_t = t_end - t_start
            delta_steps = step_end - step_start

            if delta_steps == 0:
                delta_t = t_end
                delta_steps = step_end

            if delta_steps > 0 and delta_t > 0:
                sec_per_step = delta_t / delta_steps
                stats[unique_key] = {
                    'N': N,
                    'seed': seed,
                    'sec_per_step': sec_per_step,
                    'max_step': step_end,
                    'is_completed': is_completed,
                    'last_update': mtime,
                    'total_time_elapsed': t_end,
                    'update_interval': update_interval,
                    'source': 'log',
                    'physics': physics
                }
        except Exception:
            continue
    return stats

def fit_power_law(x_vals, y_vals):
    """Fits y = c * x^m using log-log linear regression."""
    if len(x_vals) < 2: return None
    try:
        log_x = np.log(x_vals)
        log_y = np.log(y_vals)
        coeffs = np.polyfit(log_x, log_y, 1)
        exponent = coeffs[0]
        constant = np.exp(coeffs[1])
        return constant, exponent
    except:
        return None

def analyze_engine(target_engine):
    """
    Analyzes a single engine.
    Returns:
      current_rows: List of dicts for the Monitoring table.
      model_data: Dict containing regression parameters {speed: (a,b), steps: (c,d), max_n: int}
    """
    fs_data = get_fs_artifacts_stats(target_engine)
    log_data = get_log_stats(target_engine)
    merged = fs_data.copy()
    merged.update(log_data)

    if not merged:
        return [], None

    grouped = {}
    completed_runs_n = []
    completed_runs_steps = []
    all_n_vals = []
    all_speed_vals = []
    now = time.time()

    for k, d in merged.items():
        N = d['N']
        if N not in grouped:
            grouped[N] = {
                'runs': [],
                'k_mins': [], 'k_avgs': [], 'k_maxs': [], 'tris': []
            }

        grouped[N]['runs'].append(d)

        if d.get('physics'):
            p = d['physics']
            grouped[N]['k_mins'].append(p['k_min'])
            grouped[N]['k_avgs'].append(p['k_avg'])
            grouped[N]['k_maxs'].append(p['k_max'])
            grouped[N]['tris'].append(p['tri'])

        all_n_vals.append(N)
        all_speed_vals.append(d['sec_per_step'])

        if d.get('is_completed', False):
            completed_runs_n.append(N)
            completed_runs_steps.append(d['max_step'])

    sorted_Ns = sorted(grouped.keys())

    # --- MODELS ---
    # Speed Model: t_step = a * N^b
    speed_model = fit_power_law(all_n_vals, all_speed_vals)
    if not speed_model:
        # Fallback if fit fails
        default_a = np.mean(all_speed_vals)/(sorted_Ns[-1]**FALLBACK_COMPLEXITY_EXPONENT)
        speed_model = (default_a, FALLBACK_COMPLEXITY_EXPONENT)

    # Steps Model: steps = c * N^d
    unique_completed_n = sorted(list(set(completed_runs_n)))
    avg_completed_steps = []
    for uN in unique_completed_n:
        avg = np.mean([s for n, s in zip(completed_runs_n, completed_runs_steps) if n == uN])
        avg_completed_steps.append(avg)

    steps_model = fit_power_law(unique_completed_n, avg_completed_steps)
    if not steps_model:
        # We allow None here to signal "cannot predict completion"
        steps_model = (None, None)

    # Pack model data for global comparison
    model_data = {
        'engine': target_engine,
        'speed_params': speed_model, # (a, b)
        'steps_params': steps_model, # (c, d) or (None, None)
        'max_n': sorted_Ns[-1]
    }

    # --- ROW GENERATION ---
    current_rows = []
    c_steps, d_steps = steps_model

    for N in sorted_Ns:
        g = grouped[N]
        runs = g['runs']

        # 1. Basic Stats
        seed_count = len(runs)
        speeds = [r['sec_per_step'] for r in runs]
        avg_sps = np.mean(speeds)
        hz = 1.0 / avg_sps

        # 2. Physics Strings
        avg_tri = 0
        if g['k_avgs']:
            avg_k_min = np.mean(g['k_mins'])
            avg_k_avg = np.mean(g['k_avgs'])
            avg_k_max = np.mean(g['k_maxs'])

            avg_tri = np.mean(g['tris'])
            std_tri = np.std(g['tris'])

            str_k = f"{avg_k_min:<2.0f}/{avg_k_avg:<8.5f}/{avg_k_max:<2.0f}"
            if seed_count > 1:
                str_tri = f"{int(avg_tri):,} \u00B1 {int(std_tri):,}"
            else:
                str_tri = f"{int(avg_tri):,}"
        else:
            str_k = "-"
            str_tri = "-"

        # 3. Active / Stale / Completed sorting
        active_runs = [r for r in runs if not r['is_completed'] and (now - r['last_update']) < ACTIVITY_THRESHOLD_SECONDS]
        stale_runs = [r for r in runs if not r['is_completed'] and (now - r['last_update']) >= ACTIVITY_THRESHOLD_SECONDS]
        completed_runs = [r for r in runs if r['is_completed']]

        active_runs.sort(key=lambda x: x['last_update'], reverse=True)
        stale_runs.sort(key=lambda x: x['last_update'], reverse=True)
        completed_steps = [r['max_step'] for r in completed_runs]

        # 4. Completion & Stabilization Stats
        avg_time_done_str = "-"
        if completed_runs:
            times = [r['total_time_elapsed'] for r in completed_runs]
            avg_time_sec = np.mean(times)
            avg_time_done_str = str(datetime.timedelta(seconds=int(avg_time_sec)))
        elif active_runs:
            max_elapsed = max(r['total_time_elapsed'] for r in active_runs)
            avg_time_done_str = str(datetime.timedelta(seconds=int(max_elapsed)))

        # Stabilization Steps (Average Step)
        predicted_goal = None
        str_avg_steps = "?"

        max_active_step = 0
        if active_runs:
            max_active_step = max(r['max_step'] for r in active_runs)
        if stale_runs and not active_runs:
            max_active_step = max(r['max_step'] for r in stale_runs)

        if completed_steps:
            predicted_goal = int(np.mean(completed_steps))
            str_avg_steps = f"{predicted_goal:,}"
        elif c_steps is not None:
            model_pred = int(c_steps * (N ** d_steps))
            if max_active_step > model_pred:
                predicted_goal = max_active_step
                str_avg_steps = f"> {predicted_goal:,}"
            else:
                predicted_goal = model_pred
                str_avg_steps = f"{predicted_goal:,} (Est)"
        elif active_runs:
             predicted_goal = max_active_step
             str_avg_steps = f"> {predicted_goal:,}"

        # 5. EFFICIENCY METRIC
        raw_eff = 0.0
        if predicted_goal and predicted_goal > 0 and avg_tri > 0:
            eff_val = (avg_tri / predicted_goal) * 1_000_000
            raw_eff = eff_val
            str_eff = f"{eff_val:,.1f} ({target_engine})"
        else:
            str_eff = "-"

        # 6. Active / ETA Logic
        eta_str = "-"
        display_curr_step = 0

        def format_ts(ts):
            return datetime.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")

        if len(completed_runs) == seed_count:
            last_ts = max([r['last_update'] for r in completed_runs])
            eta_str = format_ts(last_ts)
            display_curr_step = max([r['max_step'] for r in completed_runs])

        elif active_runs:
            primary_run = active_runs[0]
            display_curr_step = primary_run['max_step']
            last_ts = primary_run['last_update']

            time_gap = now - last_ts
            run_interval = primary_run.get('update_interval', 0.0)
            adaptive_threshold = max(DEFAULT_DISPLAY_TIMEOUT, STITCH_SAFETY_MARGIN * run_interval)

            if time_gap > adaptive_threshold:
                eta_str = format_ts(last_ts)
            elif predicted_goal:
                remaining_steps = predicted_goal - display_curr_step
                if remaining_steps <= 0:
                     eta_str = format_ts(last_ts)
                else:
                    est_seconds_raw = remaining_steps * primary_run['sec_per_step']
                    real_eta_seconds = est_seconds_raw - time_gap

                    if real_eta_seconds < 0:
                        eta_str = format_ts(last_ts)
                    else:
                        eta_str = str(datetime.timedelta(seconds=int(real_eta_seconds)))
            else:
                eta_str = format_ts(last_ts)

        elif not active_runs and stale_runs:
            primary_run = stale_runs[0]
            display_curr_step = primary_run['max_step']
            last_activity = primary_run['last_update']
            eta_str = format_ts(last_activity)

        elif not active_runs and not stale_runs and completed_runs:
            last_ts = max([r['last_update'] for r in completed_runs])
            eta_str = format_ts(last_ts)
            display_curr_step = max([r['max_step'] for r in completed_runs])

        else:
            eta_str = "Pending"
            display_curr_step = 0

        current_rows.append({
            'Engine': target_engine,
            'N': N,
            'Seeds': f"{len(completed_runs)}/{seed_count}",  # CHANGED: Show Stable/Total
            'Hz': hz,
            'Avg T(Done)': avg_time_done_str,
            'Curr': f"{display_curr_step:,}",
            'Avg Steps': str_avg_steps,
            'K_Metrics': str_k,
            'Tri': str_tri,
            'Eff': str_eff,
            'raw_eff': raw_eff,
            'ETA': eta_str
        })

    return current_rows, model_data


def main():
    parser = argparse.ArgumentParser(description="Consolidated Performance Analysis Tool")
    parser.add_argument("-e", "--engine", type=str, help="Specific engine (e.g., 'E1'). Defaults to ALL.", default=None)
    args = parser.parse_args()

    print(f"--- CONSOLIDATED BENCHMARK REPORT ---")
    print(f"Host CPU: {get_cpu_name()}")

    engines_to_run = [args.engine] if args.engine else get_available_engines()
    if not engines_to_run: engines_to_run = ["*"]

    all_monitoring = []
    all_models = []

    # 1. Gather Data
    for eng in engines_to_run:
        mon, model = analyze_engine(eng)
        all_monitoring.extend(mon)
        if model:
            all_models.append(model)

    if not all_monitoring:
        print("No valid run data found.")
        return

    # --- TABLE 1: ACTIVE MONITORING ---
    def get_sort_key(row):
        val_n = row['N']
        val_eff = row.get('raw_eff', 0)
        return (val_n, -val_eff)

    all_monitoring.sort(key=get_sort_key)

    width_needed = 152
    print("\n" + "="*width_needed)
    print(f" ACTIVE MONITORING & SCALING METRICS (Sorted by N -> High Efficiency)")
    print("="*width_needed)

    ROW_FMT = "{:<4} {:<7} {:<6} {:<10.0f} {:<10} {:<16} {:<20} {:<15} {:<15} {:<16} {:<22}"

    header_data = {
        'Engine': 'Eng', 'N': 'N', 'Seeds': 'Seeds', 'Hz': 'Hz',
        'Avg T(Done)': 'Avg Time', 'Curr': 'Curr', 'Avg Steps': 'Avg Steps',
        'K_Metrics': 'k(m/avg/M)', 'Tri': 'Tri', 'Eff': 'Tri/1M', 'ETA': 'ETA / Last Active'
    }

    print(ROW_FMT.format(
        header_data['Engine'], header_data['N'], header_data['Seeds'], 444,
        header_data['Avg T(Done)'], header_data['Curr'], header_data['Avg Steps'],
        header_data['K_Metrics'], header_data['Tri'], header_data['Eff'], header_data['ETA']
    ).replace("444      ", "Hz        "))

    print("-" * width_needed)

    prev_n = None
    for row in all_monitoring:
        if prev_n is not None and row['N'] != prev_n:
            print("")
        print(ROW_FMT.format(
            row['Engine'], row['N'], row['Seeds'], row['Hz'],
            row['Avg T(Done)'], row['Curr'], row['Avg Steps'],
            row['K_Metrics'], row['Tri'], row['Eff'], row['ETA']
        ))
        prev_n = row['N']

    # --- FORECAST RANKING (RACE TO X) ---
    if not all_models:
        return

    # Determine Global Max N to set fair targets
    global_max_N = max(m['max_n'] for m in all_models)
    targets = [global_max_N * 4, global_max_N * 16]

    print("\n" + "="*80)
    print(f" FORECAST LEADERBOARD: Race to Target (Based on Global Max N={global_max_N})")
    print(f" Ranked by Estimated Total Time to Completion")
    print("="*80)

    forecast_fmt = "{:<4} {:<6} {:<16} {:<15} {:<18} {:<15}"

    for T in targets:
        print(f"\n>> TARGET N = {T:,} ({T//global_max_N}x)")
        print("-" * 80)
        print(forecast_fmt.format("Rank", "Eng", "Est Time", "Speed (s/it)", "Total Steps", "Scale Exp"))
        print("-" * 80)

        # Calculate prediction for each engine for this Target T
        rank_data = []
        for model in all_models:
            eng = model['engine']
            a, b = model['speed_params']
            c, d = model['steps_params']

            # Speed Prediction
            pred_sec_per_step = a * (T ** b)

            # Steps Prediction
            if c is not None:
                pred_total_steps = c * (T ** d)
                total_seconds = pred_sec_per_step * pred_total_steps
                rank_data.append({
                    'eng': eng,
                    'total_sec': total_seconds,
                    'sps': pred_sec_per_step,
                    'steps': pred_total_steps,
                    'b_exp': b
                })
            else:
                # Engine hasn't completed any runs, can't predict total steps
                pass

        # Sort by Total Time
        rank_data.sort(key=lambda x: x['total_sec'])

        if not rank_data:
            print("   (Insufficient completion data to forecast)")

        for i, r in enumerate(rank_data):
            time_str = str(datetime.timedelta(seconds=int(r['total_sec'])))
            sps_str = f"{r['sps']:.4f}"
            steps_str = f"{int(r['steps']):,}"

            # Annotate "Speed Exponent" to see scaling factor
            exp_str = f"O(N^{r['b_exp']:.2f})"

            print(forecast_fmt.format(
                i + 1,
                r['eng'],
                time_str,
                sps_str,
                steps_str,
                exp_str
            ))

    print("-" * 80)

if __name__ == "__main__":
    main()
