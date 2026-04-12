import os
import sys
import json
import shutil
import random
import subprocess
import datetime
from pathlib import Path

# CONFIGURATION
BASE_NAME = "sample3" # Needs to match the video/files you want, e.g., "sample", "sample_p2"
VIDEO_FILE = f"{BASE_NAME}.mp4" 
VISUAL_CSV = f"output/{BASE_NAME}_output_cancel/{BASE_NAME}_corrected_surface_waveform.csv"
OUTPUT_DIR_NAME = f"{BASE_NAME}_output_cancel"
TARGET_ROI_JSON = f"rois/{BASE_NAME}_cancel_rois.json"

# 自动在 txt_home 目录中查找所有以 BASE_NAME 开头的 txt 文件
SEISMIC_FILES = [str(p).replace('\\', '/') for p in Path("txt_home").glob(f"{BASE_NAME}_*.txt")]

# Random Search Constraints
MAX_X = 1920 # Feel free to change based on actual video size
MAX_Y = 1080 
MIN_W = 20
MIN_H = 20
MAX_ITERS = 100
R_THRESHOLD = 0.5
ELITE_POOL_SIZE = 6
MIN_DIVERSITY = 0.08  # ROI normalized distance threshold for keeping distinct search states
STALL_RESTART_ITERS = 8

def read_rois(filepath):
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'r') as f:
        return json.load(f)

def write_rois(filepath, rois):
    with open(filepath, 'w') as f:
        json.dump(rois, f)

def perturb_rois(rois, variance=5):
    """
    Hill climbing perturbation. Shift coords slightly, ensuring validity.
    rois: [[x1, y1, x2, y2], [x1, y1, x2, y2]]
    """
    new_rois = []
    for roi in rois:
        # Add random value from normal distribution
        x1 = int(round(roi[0] + random.gauss(0, variance)))
        y1 = int(round(roi[1] + random.gauss(0, variance)))
        x2 = int(round(roi[2] + random.gauss(0, variance)))
        y2 = int(round(roi[3] + random.gauss(0, variance)))

        # Ensure order
        if x1 >= x2:
            x1, x2 = x2 - MIN_W, x1 + MIN_W
        if y1 >= y2:
            y1, y2 = y2 - MIN_H, y1 + MIN_H

        # Boundary checks
        x1 = max(0, min(x1, MAX_X - MIN_W))
        y1 = max(0, min(y1, MAX_Y - MIN_H))
        x2 = max(x1 + MIN_W, min(x2, MAX_X))
        y2 = max(y1 + MIN_H, min(y2, MAX_Y))

        new_rois.append([x1, y1, x2, y2])
    return new_rois

def roi_distance(rois_a, rois_b):
    """Measure normalized geometric distance between two ROI sets."""
    if not rois_a or not rois_b or len(rois_a) != len(rois_b):
        return 1.0

    total = 0.0
    for ra, rb in zip(rois_a, rois_b):
        ax1, ay1, ax2, ay2 = ra
        bx1, by1, bx2, by2 = rb
        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0
        aw = max(1, ax2 - ax1)
        ah = max(1, ay2 - ay1)
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)
        total += (
            abs(acx - bcx) / MAX_X
            + abs(acy - bcy) / MAX_Y
            + abs(aw - bw) / MAX_X
            + abs(ah - bh) / MAX_Y
        )

    return total / len(rois_a)

def update_elite_pool(elite_pool, rois, score):
    """Keep a small set of high-scoring, sufficiently different candidates."""
    candidate = {"rois": rois, "score": score}

    # If the candidate is too similar to an existing one, only keep the better score.
    for idx, item in enumerate(elite_pool):
        if roi_distance(item["rois"], rois) < MIN_DIVERSITY:
            if score > item["score"]:
                elite_pool[idx] = candidate
            break
    else:
        elite_pool.append(candidate)

    elite_pool.sort(key=lambda x: x["score"], reverse=True)
    del elite_pool[ELITE_POOL_SIZE:]
    return elite_pool

def is_diverse_from_pool(rois, elite_pool):
    if not elite_pool:
        return True
    return all(roi_distance(rois, item["rois"]) >= MIN_DIVERSITY for item in elite_pool)

def choose_parent_rois(elite_pool, base_rois, stagnation_iters=0):
    """Pick a parent ROI from the elite pool, or fall back to the base ROI for exploration."""
    if not elite_pool:
        return base_rois

    # When stuck, increase exploration by sampling from older / weaker elites or the base ROI.
    if stagnation_iters >= STALL_RESTART_ITERS:
        if random.random() < 0.5:
            return base_rois
        return random.choice(elite_pool)["rois"]

    # Normally bias toward better candidates, but still keep diversity.
    weights = [max(0.01, item["score"] - 0.1) for item in elite_pool]
    return random.choices(elite_pool, weights=weights, k=1)[0]["rois"]

def run_main_cancel():
    cmd = ["python", "main_cancel.py", VIDEO_FILE]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running main_cancel:")
        print(result.stderr)
        return False
    return True

def run_evaluation():
    best_loc_r_x = -1.0
    best_loc_r_y = -1.0
    best_txt = ""
    best_freq_err_x = 100.0  # Initialize to high value
    best_freq_err_y = 100.0  # Initialize to high value

    visual_csv_path = Path(f"output/{OUTPUT_DIR_NAME}/{BASE_NAME}_corrected_surface_waveform.csv")
    if not visual_csv_path.exists():
        print(f"File not found: {visual_csv_path}")
        return None, None, None, None, None

    for txt_path in SEISMIC_FILES:
        out_img_path = Path(f"output/{OUTPUT_DIR_NAME}/alignment_{Path(txt_path).stem}.png")
        
        cmd = [
            "python", "compare_acc_disp.py",
            "--seismic_txt", str(txt_path),
            "--visual_csv", str(visual_csv_path),
            "--out_img", str(out_img_path)
            # You can also pass visual maps here if needed
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        r_val_x = -1.0
        r_val_y = -1.0
        freq_err_x_val = 100.0
        freq_err_y_val = 100.0
        # Parse Pearson r and frequency errors from output
        for line in res.stdout.replace('\r', '\n').split('\n'):
            if "__RESULT_R_X__:" in line:
                idx = line.find("__RESULT_R_X__:")
                r_val_x = float(line[idx:].split(":")[1].strip())
            if "__RESULT_R_Y__:" in line:
                idx = line.find("__RESULT_R_Y__:")
                r_val_y = float(line[idx:].split(":")[1].strip())
            if "__RESULT_FREQ_ERR_X__:" in line:
                idx = line.find("__RESULT_FREQ_ERR_X__:")
                freq_err_x_val = float(line[idx:].split(":")[1].strip())
            if "__RESULT_FREQ_ERR_Y__:" in line:
                idx = line.find("__RESULT_FREQ_ERR_Y__:")
                freq_err_y_val = float(line[idx:].split(":")[1].strip())
        
        if r_val_x == -1.0:
            print(f"Error evaluating {txt_path}. Stderr: {res.stderr[:200]}")
            
        print(f"  -> Evaluated vs {Path(txt_path).name}: R_X = {r_val_x}, R_Y = {r_val_y}, Freq Err X = {freq_err_x_val}%, Y = {freq_err_y_val}%")
        if r_val_x > best_loc_r_x:
            best_loc_r_x = r_val_x
            best_loc_r_y = r_val_y
            best_freq_err_x = freq_err_x_val
            best_freq_err_y = freq_err_y_val
            best_txt = txt_path
            
    return best_loc_r_x, best_loc_r_y, best_txt, best_freq_err_x, best_freq_err_y

def save_best_results(best_r_x, best_r_y, best_txt, best_freq_err_x, best_freq_err_y, rois):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Folder name: timestamp + {rx} + {ry} + {freq_err_x}% + {freq_err_y}%
    folder_name = f"{timestamp}_{best_r_x:.4f}_{best_r_y:.4f}_{best_freq_err_x:.2f}%_{best_freq_err_y:.2f}%"
    best_dir = Path("best") / folder_name
    best_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save ROI
    write_rois(best_dir / f"{BASE_NAME}_cancel_rois.json", rois)
    
    # 2. Copy entire output folder
    src_out_dir = Path(f"output/{OUTPUT_DIR_NAME}")
    if src_out_dir.exists():
        shutil.copytree(src_out_dir, best_dir / OUTPUT_DIR_NAME, dirs_exist_ok=True)
        
    # 3. Summary
    summary = {
        "video": VIDEO_FILE,
        "best_r_x": best_r_x,
        "best_r_y": best_r_y,
        "best_freq_err_x": best_freq_err_x,
        "best_freq_err_y": best_freq_err_y,
        "matched_with": best_txt,
        "rois": rois,
        "iteration_time": timestamp
    }
    with open(best_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"✅ Success! R_X={best_r_x}, Freq Err X={best_freq_err_x}%, Y={best_freq_err_y}%. Results saved to: {best_dir}")

def main():
    print("🚀 Starting Automated ROI Search (Elite Pool + Diversity)...")
    base_rois = read_rois(TARGET_ROI_JSON)
    if not base_rois:
        print(f"Could not find valid base ROIs in {TARGET_ROI_JSON}.")
        print("Falling back to sample_cancel_rois.json if it exists.")
        base_rois = read_rois("rois/sample_cancel_rois.json")
        if not base_rois:
            # Fake defaults if neither exists
            base_rois = [[400, 400, 600, 600], [10, 10, 200, 200]]
            
    current_best_r = -1.0
    current_best_rois = base_rois
    elite_pool = []
    stagnation_iters = 0
    
    # Pre-test current
    print("\n--- Iteration 0 (Initial Base) ---")
    write_rois(TARGET_ROI_JSON, current_best_rois)
    if run_main_cancel():
        rx, ry, txt, freq_err_x, freq_err_y = run_evaluation()
        if rx is not None:
            print(f"Base ROI Max R_X: {rx}, R_Y: {ry}, Freq Err X: {freq_err_x}%, Y: {freq_err_y}%")
            current_best_r = rx
            elite_pool = update_elite_pool(elite_pool, current_best_rois, rx)
            if rx > R_THRESHOLD:
                # Found a good match, but keep searching for potentially better ROIs.
                save_best_results(rx, ry, txt, freq_err_x, freq_err_y, current_best_rois)
                print("Good result found immediately. Saving result and continuing search.")
    
    for i in range(1, MAX_ITERS + 1):
        print(f"\n--- Iteration {i} ---")
        
        # Choose a parent from the elite pool instead of always mutating the single best.
        parent_rois = choose_parent_rois(elite_pool, base_rois, stagnation_iters=stagnation_iters)

        # Increase variance when the search is stuck, so it can escape the local neighborhood.
        if stagnation_iters >= STALL_RESTART_ITERS:
            variance = 8 + min(6, stagnation_iters)
        else:
            variance = 3 + (i % 4)  # 3, 4, 5, 6

        test_rois = perturb_rois(parent_rois, variance=variance)
        
        print(f"Testing new ROI: {test_rois}")
        write_rois(TARGET_ROI_JSON, test_rois)
        
        if not run_main_cancel():
            continue
            
        rx, ry, txt, freq_err_x, freq_err_y = run_evaluation()
        if rx is not None:
            print(f"Iter {i} Max R_X: {rx} (vs {txt}), Freq Err X: {freq_err_x}%, Y: {freq_err_y}%")
            previous_best_r = current_best_r
            elite_pool = update_elite_pool(elite_pool, test_rois, rx)

            if rx > current_best_r:
                print(f"🔥 New Local Best R! {rx} > {current_best_r}")
                current_best_r = rx
                current_best_rois = test_rois
                stagnation_iters = 0
            else:
                stagnation_iters += 1
                
            if rx > R_THRESHOLD:
                # Only save genuinely new high-quality results; do not spam near-duplicates.
                if is_diverse_from_pool(test_rois, elite_pool) or rx > previous_best_r + 0.01:
                    save_best_results(rx, ry, txt, freq_err_x, freq_err_y, test_rois)
                    print("Target R threshold reached. Saving a diverse result and continuing search...")

            # Periodically inject a random restart to avoid all samples collapsing around one local mode.
            if stagnation_iters >= STALL_RESTART_ITERS:
                print("🔁 Stagnation detected. Next iterations will explore more broadly.")

if __name__ == "__main__":
    main()
