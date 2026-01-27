import threading
import os
import sys
import datetime

def run_cmd(cmd):
    # -u ensures output is unbuffered (appears instantly)
    full_cmd = f"python -u {cmd}"
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Launching: {cmd}")
    os.system(full_cmd)

commands = [

    "drive.py -N 200  -c 10",
    "drive.py -N 400  -c 10",
    "drive.py -N 800  -c 10",
    "drive.py -N 1600 -c 10",







]

# Capture extra arguments
extra_args = " ".join(sys.argv[1:])

threads = []
batch_start = datetime.datetime.now()

print(f"=== BATCH STARTED AT {batch_start.strftime('%Y-%m-%d %H:%M:%S')} ===")
print("-" * 60)

for cmd in commands:
    cmd_with_args = f"{cmd} {extra_args}" if extra_args else cmd
    t = threading.Thread(target=run_cmd, args=(cmd_with_args,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

batch_end = datetime.datetime.now()
duration = batch_end - batch_start

print("-" * 60)
print(f"All threads done. Batch Duration: {str(duration).split('.')[0]}")
