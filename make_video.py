import os
import subprocess
import shutil
import argparse
import sys
from pathlib import Path

# --- CONFIGURATION ---
BASE_DIR = "runs"
OUTPUT_FILENAME = "evolution_smooth.mp4"

# 1. How fast to step through your original images?
# Lower = Slower Video. (e.g., 5 means we spend 0.2s on each original image)
INPUT_READ_RATE = 2

# 2. How smooth should the final video be?
# 60 is standard for smooth fluid motion.
OUTPUT_FPS = 60

def find_target_dir(base_path, version=None, nodes=None, seed=None):
    """
    Locates the specific render directory based on provided args.
    If args are missing, defaults to the latest modified directory.
    """
    base = Path(base_path)
    if not base.exists():
        print(f"Error: Base directory '{base_path}' does not exist.")
        return None

    # --- MODE 1: EXPLICIT TARGETING ---
    if nodes is not None and seed is not None:
        # If version is not provided, try to find the latest version for these parameters
        if version is None:
            # Search all E* directories
            engine_dirs = sorted([p for p in base.iterdir() if p.is_dir() and p.name.startswith("E")])
            if not engine_dirs:
                print("Error: No engine versions found in base directory.")
                return None

            # Try to find the specific N/S combination in the latest engines first
            found_path = None
            for e_dir in reversed(engine_dirs):
                target = e_dir / f"N{nodes}" / f"S{seed}" / "renders"
                if target.exists():
                    found_path = target
                    break

            if not found_path:
                print(f"Error: Could not find renders for N={nodes}, S={seed} in any engine version.")
                return None
            return found_path

        else:
            # Fully explicit path
            target = base / f"E{version}" / f"N{nodes}" / f"S{seed}" / "renders"
            if not target.exists():
                print(f"Error: Target directory not found: {target}")
                return None
            return target

    # --- MODE 2: AUTO-DETECT LATEST ---
    print("No specific target provided (-N, -s). Searching for latest render...")
    render_dirs = [p for p in base.rglob("renders") if p.is_dir()]

    if not render_dirs:
        print("Error: No 'renders' directories found anywhere in runs/.")
        return None

    # Sort by modification time (newest first)
    return max(render_dirs, key=lambda p: p.stat().st_mtime)

def create_smooth_video(render_dir):
    if not shutil.which("ffmpeg"):
        print("Error: 'ffmpeg' not found. Please install ffmpeg.")
        return

    # Output goes in the parent directory (S{seed})
    output_path = render_dir.parent / OUTPUT_FILENAME

    # We pattern match .png files inside the render dir
    input_pattern = str(render_dir / "*.png")

    # Check if there are actually images there
    if not list(render_dir.glob("*.png")):
        print(f"Error: No .png files found in {render_dir}")
        return

    print(f"\nProcessing source: {render_dir}")
    print(f"Target Speed:      {INPUT_READ_RATE} original frames/sec")
    print(f"Target Smoothness: {OUTPUT_FPS} fps (interpolated)")

    # --- THE MAGIC FILTERS ---
    # 1. minterpolate: Creates intermediate frames.
    #    fps=60: Target framerate
    #    mi_mode=mci: Motion Compensated Interpolation (The "AI" morphing look)
    #    mc_mode=aobmc: Advanced Overlapped Block Motion Compensation (Better quality)
    #    me_mode=bidir: Bidirectional (looks ahead and behind to average)

    video_filter = (
        f"minterpolate=fps={OUTPUT_FPS}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2" # Ensure even dimensions for encoder
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(INPUT_READ_RATE), # Read input slowly
        "-pattern_type", "glob",
        "-i", input_pattern,
        "-vf", video_filter,                # Apply the interpolation
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",                       # High quality
        str(output_path)
    ]

    try:
        print(">> Rendering... (This will take longer due to interpolation)")
        # subprocess.run(cmd, check=True) # verbose
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

        print("\n" + "="*60)
        print(f"SUCCESS! Smooth video saved to:\n{output_path}")
        print("="*60)
    except subprocess.CalledProcessError:
        print(f"\nFFmpeg Error. Try running the command manually to debug:")
        print(" ".join(cmd))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create smooth interpolated videos from simulation renders.")

    parser.add_argument("-N", "--nodes", type=int, help="Target specific Node count (e.g., 100)")
    parser.add_argument("-s", "--seed", type=int, help="Target specific Seed (e.g., 1000)")
    parser.add_argument("-v", "--version", type=int, help="Target specific Engine Version (e.g., 1). Optional.")

    args = parser.parse_args()

    print("--- Relational Reality Smooth Visualizer ---")

    target_dir = find_target_dir(BASE_DIR, version=args.version, nodes=args.nodes, seed=args.seed)

    if target_dir:
        create_smooth_video(target_dir)
