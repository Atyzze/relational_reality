import os
import glob
import numpy as np
import datetime
import re
import platform
import subprocess
import sys
import argparse

# --- CONFIG ---
RUNS_DIR = "runs"
FALLBACK_COMPLEXITY_EXPONENT = 2.0  # O(N^2) assume
FALLBACK_PHYSICS_EXPONENT = 1.5     # O(N^1.5) assume for steps if no data

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

        if delta_steps > 0 and delta_time > 0.1:
            sec_per_step = delta_time / delta_steps
            artifacts[unique_key] = {
                'N': N,
                'sec_per_step': sec_per_step,
                'max_step': last_step,
                'is_completed': False,
                'source': 'filesystem'
            }
    return artifacts

def get_log_stats(target_engine="*"):
    """Scans logs for precise timing and completion status."""
    stats = {}
    search_path = os.path.join(RUNS_DIR, target_engine, "**", "logs", "*.csv")
    log_files = glob.glob(search_path, recursive=True)

    for lf in log_files:
        try:
            with open(lf, 'r') as f:
                lines = f.readlines()
            if len(lines) < 3: continue

            meta = parse_metadata(lines[0])
            if 'N' not in meta or 'seed' not in meta: continue
            N = int(meta['N'])
            seed = int(meta['seed'])
            unique_key = f"N{N}_S{seed}"

            is_completed = any(l.startswith("# COMPLETED") for l in lines[-3:])

            # Parse timing
            first_data = lines[2].strip().split(',')
            last_data_line = [l for l in lines if not l.startswith('#')][-1]
            last_data = last_data_line.strip().split(',')

            t_start = float(first_data[3])
            t_end = float(last_data[3])
            step_start = int(first_data[1])
            step_end = int(last_data[1])

            delta_t = t_end - t_start
            delta_steps = step_end - step_start

            if delta_steps == 0:
                delta_t = t_end
                delta_steps = step_end

            if delta_steps > 0 and delta_t > 0:
                sec_per_step = delta_t / delta_steps
                stats[unique_key] = {
                    'N': N,
                    'sec_per_step': sec_per_step,
                    'max_step': step_end,
                    'is_completed': is_completed,
                    'source': 'log'
                }
        except:
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
    Analyzes a single engine and returns lists of dictionaries.
    """

    # 1. Gather Data
    fs_data = get_fs_artifacts_stats(target_engine)
    log_data = get_log_stats(target_engine)
    merged = fs_data.copy()
    merged.update(log_data)

    if not merged:
        return [], []

    # 2. Group Data
    grouped = {}
    completed_runs_n = []
    completed_runs_steps = []
    all_n_vals = []
    all_speed_vals = []

    for k, d in merged.items():
        N = d['N']
        if N not in grouped:
            grouped[N] = {'speeds': [], 'completed_steps': [], 'all_current_steps': []}

        grouped[N]['speeds'].append(d['sec_per_step'])
        grouped[N]['all_current_steps'].append(d['max_step'])
        all_n_vals.append(N)
        all_speed_vals.append(d['sec_per_step'])

        if d.get('is_completed', False):
            grouped[N]['completed_steps'].append(d['max_step'])
            completed_runs_n.append(N)
            completed_runs_steps.append(d['max_step'])

    sorted_Ns = sorted(grouped.keys())

    # 3. Build Models
    # Speed Model
    speed_model = fit_power_law(all_n_vals, all_speed_vals)
    if speed_model:
        a_speed, b_speed = speed_model
    else:
        a_speed, b_speed = (np.mean(all_speed_vals)/(sorted_Ns[-1]**FALLBACK_COMPLEXITY_EXPONENT), FALLBACK_COMPLEXITY_EXPONENT)

    # Physics Model
    unique_completed_n = sorted(list(set(completed_runs_n)))
    avg_completed_steps = []
    for uN in unique_completed_n:
        avg = np.mean([s for n, s in zip(completed_runs_n, completed_runs_steps) if n == uN])
        avg_completed_steps.append(avg)

    steps_model = fit_power_law(unique_completed_n, avg_completed_steps)
    if steps_model:
        c_steps, d_steps = steps_model
    else:
        c_steps, d_steps = (None, None)

    # 4. Generate Monitoring Rows
    current_rows = []
    for N in sorted_Ns:
        data = grouped[N]
        avg_sps = np.mean(data['speeds'])
        hz = 1.0 / avg_sps

        # Stats
        if data['completed_steps']:
            avg_stable_steps = int(np.mean(data['completed_steps']))
            str_stable = f"{avg_stable_steps:,}"
        else:
            avg_stable_steps = None
            str_stable = "-"

        max_curr_step = max(data['all_current_steps'])
        str_curr = f"{max_curr_step:,}"

        # Goal Prediction
        predicted_goal = None
        if avg_stable_steps:
            predicted_goal = avg_stable_steps
            str_goal = f"{predicted_goal:,} (Hist)"
        elif c_steps is not None:
            predicted_goal = int(c_steps * (N ** d_steps))
            str_goal = f"{predicted_goal:,} (Pred)"
        else:
            str_goal = "?"

        # ETA
        eta_str = "-"
        if predicted_goal:
            remaining = predicted_goal - max_curr_step
            if remaining < 0 and not data['completed_steps']:
                overdue_pct = abs(remaining) / predicted_goal * 100
                eta_str = f"+{overdue_pct:.0f}% Overdue"
            elif data['completed_steps'] and remaining <= 0:
                # Use theoretical run time for completed
                avg_total_sec = avg_stable_steps * avg_sps
                eta_str = str(datetime.timedelta(seconds=int(avg_total_sec)))
            elif remaining > 0:
                total_sec_left = remaining * avg_sps
                if total_sec_left < 60:
                    eta_str = "< 1 min"
                else:
                    eta_str = str(datetime.timedelta(seconds=int(total_sec_left)))

        current_rows.append({
            'Engine': target_engine,
            'N': N,
            'Hz': hz,
            'Sec/Step': avg_sps,
            'Stable': str_stable,
            'Curr': str_curr,
            'Goal': str_goal,
            'ETA': eta_str
        })

    # 5. Generate Forecast Rows
    forecast_rows = []
    last_N = sorted_Ns[-1]
    targets = [last_N * 2, last_N * 4, last_N * 8, last_N * 16]

    for t_N in targets:
        pred_sec = a_speed * (t_N ** b_speed)

        if c_steps is not None:
            pred_steps = c_steps * (t_N ** d_steps)
            steps_str = f"{int(pred_steps):,}"
            total_sec = pred_sec * pred_steps
            eta = str(datetime.timedelta(seconds=int(total_sec)))
        else:
            steps_str = "?"
            eta = "?"

        forecast_rows.append({
            'Engine': target_engine,
            'Next N': t_N,
            'Est Speed': pred_sec,
            'Est Steps': steps_str,
            'Est Time': eta
        })

    return current_rows, forecast_rows

def main():
    parser = argparse.ArgumentParser(description="Consolidated Performance Analysis Tool")
    parser.add_argument("-e", "--engine", type=str, help="Specific engine (e.g., 'E1'). Defaults to ALL.", default=None)
    args = parser.parse_args()

    cpu_info = get_cpu_name()
    print(f"--- CONSOLIDATED BENCHMARK REPORT ---")
    print(f"Host CPU: {cpu_info}")

    engines_to_run = []
    if args.engine:
        engines_to_run = [args.engine]
    else:
        engines_to_run = get_available_engines()
        if not engines_to_run:
            print("\n[INFO] No E* directories found. Scanning generic/legacy structure...")
            engines_to_run = ["*"]

    # Collect all data first
    all_monitoring = []
    all_forecast = []

    print(f"Analyzing {len(engines_to_run)} engine configurations...")

    for eng in engines_to_run:
        mon, forc = analyze_engine(eng)
        all_monitoring.extend(mon)
        all_forecast.extend(forc)

    if not all_monitoring:
        print("No valid run data found.")
        return

    # --- SORTING ---
    # Sort by N (Ascending), then by Engine name
    all_monitoring.sort(key=lambda x: (x['N'], x['Engine']))

    # Sort Forecast by Next N (Ascending), then by Engine name
    all_forecast.sort(key=lambda x: (x['Next N'], x['Engine']))

    # --- TABLE 1: ACTIVE MONITORING ---
    print("\n" + "="*110)
    print(f" ACTIVE MONITORING & SCALING METRICS (Sorted by N)")
    print("="*110)

    # Engine | N | Hz | Sec/Step | Stable(Avg) | Curr(Max) | Est. Goal | Active ETA
    header = "{:<8} {:<8} {:<8} {:<11} {:<14} {:<14} {:<16} {:<16}".format(
        "Engine", "N", "Hz", "Sec/Step", "Stable(Avg)", "Curr(Max)", "Est. Goal", "Active ETA"
    )
    print(header)
    print("-" * 110)

    row_fmt = "{:<8} {:<8} {:<8.2f} {:<11.6f} {:<14} {:<14} {:<16} {:<16}"

    for row in all_monitoring:
        print(row_fmt.format(
            row['Engine'],
            row['N'],
            row['Hz'],
            row['Sec/Step'],
            row['Stable'],
            row['Curr'],
            row['Goal'],
            row['ETA']
        ))

    # --- TABLE 2: FUTURE FORECAST ---
    print("\n" + "="*70)
    print(f" FORECAST (Sorted by Size)")
    print("="*70)

    # Engine | Next N | Est. Speed | Est. Steps | Est. Total Time
    f_header = "{:<8} {:<10} {:<15} {:<15} {:<15}".format(
        "Engine", "Next N", "Est. Speed", "Est. Steps", "Est. Time"
    )
    print(f_header)
    print("-" * 70)

    f_row_fmt = "{:<8} {:<10} {:<15.6f} {:<15} {:<15}"

    for row in all_forecast:
        print(f_row_fmt.format(
            row['Engine'],
            row['Next N'],
            row['Est Speed'],
            row['Est Steps'],
            row['Est Time']
        ))

    print("-" * 70)

if __name__ == "__main__":
    main()
