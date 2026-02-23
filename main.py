import os
import sys
import argparse
import re
import shutil
import filecmp
import subprocess
import difflib

# --- CONFIG ---
DATA_DIR = "data"
ENGINE_FILE = "engine.py"
DRIVE_FILE = "drive.py"

def parse_existing_versions(data_dir):
    """
    Scans data/ folder to build a global catalog of versions.
    Returns:
        versions: list of (e_id, d_id, folder_path)
        max_e: integer
        max_d: integer
    """
    if not os.path.exists(data_dir): return [], 0, 0

    versions = []
    max_e = 0
    max_d = 0

    pattern = re.compile(r"^E(\d+)D(\d+)$")

    for d_name in os.listdir(data_dir):
        match = pattern.match(d_name)
        if match:
            e_id = int(match.group(1))
            d_id = int(match.group(2))
            full_path = os.path.join(data_dir, d_name)
            versions.append((e_id, d_id, full_path))

            if e_id > max_e: max_e = e_id
            if d_id > max_d: max_d = d_id

    return versions, max_e, max_d

def identify_file_version(current_file, filename_in_archive, versions, id_index):
    """
    Checks if 'current_file' matches any existing version in the archives.
    id_index: 0 for Engine ID (E), 1 for Drive ID (D) inside the 'versions' tuple.
    Returns: matched_id (int) or None
    """
    for v in versions:
        e_id, d_id, folder_path = v
        archived_file = os.path.join(folder_path, filename_in_archive)

        # If the archived file exists and matches current content
        if os.path.exists(archived_file) and filecmp.cmp(current_file, archived_file, shallow=False):
            return v[id_index] # Return the E_ID or D_ID found
    return None

def generate_diff(new_file_path, old_file_path, diff_output_path):
    """Generates a unified diff between an old file and a new file."""
    if not os.path.exists(old_file_path):
        return

    with open(old_file_path, 'r') as f_old, open(new_file_path, 'r') as f_new:
        old_lines = f_old.readlines()
        new_lines = f_new.readlines()

    diff = difflib.unified_diff(
        old_lines, new_lines,
        fromfile=os.path.basename(old_file_path) + " (Previous)",
        tofile=os.path.basename(new_file_path) + " (New)",
        n=3 # Number of context lines
    )

    with open(diff_output_path, 'w') as f_out:
        f_out.writelines(diff)

def detect_and_archive(data_dir, current_engine, current_drive):
    """
    Determines the correct E and D IDs for the current root files.
    Creates a new folder if this specific combination doesn't exist,
    and generates a diff if the files are new.
    """
    existing_versions, max_e, max_d = parse_existing_versions(data_dir)

    # 1. Identify ENGINE Version
    target_e = identify_file_version(current_engine, ENGINE_FILE, existing_versions, 0)
    is_new_e = False

    if target_e is None:
        target_e = max_e + 1
        is_new_e = True
        print(f"🆕 New Engine detected (will be E{target_e})")
    else:
        print(f"✅ Engine matches existing version E{target_e}")

    # 2. Identify DRIVE Version
    target_d = identify_file_version(current_drive, DRIVE_FILE, existing_versions, 1)
    is_new_d = False

    if target_d is None:
        target_d = max_d + 1
        is_new_d = True
        print(f"🆕 New Drive detected (will be D{target_d})")
    else:
        print(f"✅ Drive matches existing version D{target_d}")

    # 3. Construct the Target Folder for this Combo
    version_str = f"E{target_e}D{target_d}"
    new_folder = os.path.join(data_dir, version_str)

    if os.path.exists(new_folder):
        print(f"📂 Using existing configuration: {version_str}")
    else:
        print(f"🔨 Creating new configuration: {version_str}")
        os.makedirs(new_folder, exist_ok=True)

        shutil.copy2(current_engine, os.path.join(new_folder, ENGINE_FILE))
        shutil.copy2(current_drive, os.path.join(new_folder, DRIVE_FILE))

        # --- Generate Diffs for New Files ---
        if is_new_e and max_e > 0:
            # Find any previous engine version to diff against
            for v in existing_versions:
                if v[0] == max_e:
                    old_e_path = os.path.join(v[2], ENGINE_FILE)
                    diff_path = os.path.join(new_folder, "engine_diff.txt")
                    generate_diff(current_engine, old_e_path, diff_path)
                    print(f"   📝 Generated engine_diff.txt (compared against E{max_e})")
                    break

        if is_new_d and max_d > 0:
            # Find any previous drive version to diff against
            for v in existing_versions:
                if v[1] == max_d:
                    old_d_path = os.path.join(v[2], DRIVE_FILE)
                    diff_path = os.path.join(new_folder, "drive_diff.txt")
                    generate_diff(current_drive, old_d_path, diff_path)
                    print(f"   📝 Generated drive_diff.txt (compared against D{max_d})")
                    break

    return new_folder

def main():
    # 1. Parse ONLY the target-version flag if present, let the rest pass through
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-t", "--target-version", type=str)
    args, unknown_args = parser.parse_known_args()

    os.makedirs(DATA_DIR, exist_ok=True)

    target_folder = ""

    # --- STRATEGY A: FORCE SPECIFIC VERSION ---
    if args.target_version:
        v_tag = args.target_version.upper()
        target_folder = os.path.join(DATA_DIR, v_tag)
        if not os.path.exists(target_folder):
            print(f"❌ Error: Target version {v_tag} does not exist.")
            sys.exit(1)
        print(f"🔄 Forcing execution of archived version: {v_tag}")

    # --- STRATEGY B: DETECT OR CREATE FROM ROOT ---
    else:
        if not os.path.exists(ENGINE_FILE) or not os.path.exists(DRIVE_FILE):
            print("❌ Error: engine.py or drive.py missing in root directory.")
            sys.exit(1)

        target_folder = detect_and_archive(DATA_DIR, ENGINE_FILE, DRIVE_FILE)

    # 2. EXECUTE THE WORKER
    script_path = os.path.join(target_folder, DRIVE_FILE)

    # Pass unknown_args (like -N, -s) to the worker
    cmd = [sys.executable, script_path] + unknown_args

    print(f"🚀 Launching: {script_path}")
    print("-" * 40)

    if sys.platform == 'win32':
        subprocess.call(cmd)
    else:
        os.execv(sys.executable, cmd)

if __name__ == "__main__":
    main()
