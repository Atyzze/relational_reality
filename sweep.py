import threading
import os

def run_cmd(cmd):
    print(f"Starting: {cmd}")
    os.system(cmd)
    print(f"Finished: {cmd}")

#use up to 20 max for optimal load on a 9950X3D
commands = [

    "python drive.py -N 3200 -s 1000 -c 10",
    "python drive.py -N 1600 -s 1000 -c 10",
    "python drive.py -N 800 -s 1000 -c 10",
    "python drive.py -N 400 -s 1000 -c 10",
    "python drive.py -N 200 -s 1000 -c 10",
    "python drive.py -N 100 -s 1000 -c 10",

    "python drive.py -N 3200 -s 2000 -c 10",
    "python drive.py -N 1600 -s 2000 -c 10",
    "python drive.py -N 800 -s 2000 -c 10",
    "python drive.py -N 400 -s 2000 -c 10",
    "python drive.py -N 200 -s 2000 -c 10",
    "python drive.py -N 100 -s 2000 -c 10",

]

# 2. Loop to create and start threads
threads = []
for cmd in commands:
    t = threading.Thread(target=run_cmd, args=(cmd,))
    t.start()
    threads.append(t)

# 3. Loop to wait for all of them to finish
for t in threads:
    t.join()

print("All threads done.")


