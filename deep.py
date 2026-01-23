import threading, os, sys

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    os.system(cmd)
    print(f"Finished: {cmd}")

commands = [
    "python drive.py -N 25600 -s 1000 -c 1",
    "python drive.py -N 12800 -s 1000 -c 2",
    "python drive.py -N 6400  -s 1000 -c 4",

    "python drive.py -N 25600 -s 2000 -c 1",
    "python drive.py -N 12800 -s 2000 -c 2",
    "python drive.py -N 6400  -s 2000 -c 4",







]
# Capture all arguments passed after the script name
extra_args = " ".join(sys.argv[1:])

threads = []
for cmd in commands:
    # Append the captured arguments to the base command
    # We add a space to ensure separation between the command and the new args
    full_cmd = f"{cmd} {extra_args}" if extra_args else cmd

    t = threading.Thread(target=run_cmd, args=(full_cmd,))
    t.start()
    threads.append(t)

for t in threads:
    t.join()

print("All threads done.")
