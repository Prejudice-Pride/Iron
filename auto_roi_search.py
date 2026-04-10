import os
import sys
import json
import shutil
import random
import subprocess
import datetime
from pathlib import Path

# CONFIGURATION
# 正在运行的视频文件名（不含扩展名）
# 脚本将自动寻找：
# 1. 视频文件: {BASE_NAME}.mp4
# 2. 真值文件: txt_home/{BASE_NAME}_*.txt
# 3. ROI 文件: rois/{BASE_NAME}_cancel_rois.json
BASE_NAME = "pample2"

VIDEO_FILE = f"{BASE_NAME}.mp4"
OUTPUT_DIR_NAME = f"{BASE_NAME}_output_cancel"
CSV_SUFFIX = "corrected_surface_waveform.csv"
TARGET_ROI_JSON = f"rois/{BASE_NAME}_cancel_rois.json"

def get_seismic_files(base_name):
    """自动搜索 txt_home 目录下所有匹配该视频的真值文件"""
    txt_home = Path("txt_home")
    if not txt_home.exists():
        return []
    # 匹配规则：{base_name}_*.txt (例如 sample_p2_1301.txt)
    return [str(f) for f in txt_home.glob(f"{base_name}_*.txt")]

SEISMIC_FILES = get_seismic_files(BASE_NAME)
if not SEISMIC_FILES:
    print(f"⚠️ 警告: 在 txt_home 中未找到 {BASE_NAME} 的真值文件！")

# Random Search Constraints
MAX_X = 1920 # Feel free to change based on actual video size
MAX_Y = 1080 
MIN_W = 20
MIN_H = 20
MAX_ITERS = 100
R_THRESHOLD = 0.5

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
    best_loc_r = -1.0
    best_txt = ""

    visual_csv_path = Path(f"output/{OUTPUT_DIR_NAME}/{BASE_NAME}_{CSV_SUFFIX}")
    if not visual_csv_path.exists():
        print(f"File not found: {visual_csv_path}")
        return None, None

    for txt_path in SEISMIC_FILES:
        out_img_path = Path(f"output/{OUTPUT_DIR_NAME}/alignment_{Path(txt_path).stem}.png")
        
        # compare_acc_disp.py 会自动从 CSV 文件名检测坐标映射关系
        cmd = [
            "python", "compare_acc_disp.py",
            "--seismic_txt", str(txt_path),
            "--visual_csv", str(visual_csv_path),
            "--out_img", str(out_img_path)
        ]
        
        res = subprocess.run(cmd, capture_output=True, text=True)
        r_val = -1.0
        # Parse Pearson r from output
        for line in res.stdout.split('\n'):
            if line.startswith("__RESULT_R_X__:"):
                r_val = float(line.split(":")[1].strip())
        
        print(f"  -> Evaluated vs {Path(txt_path).name}: R = {r_val}")
        if r_val > best_loc_r:
            best_loc_r = r_val
            best_txt = txt_path
            
    return best_loc_r, best_txt

def save_best_results(best_r, best_txt, rois):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    best_dir = Path("best") / timestamp
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
        "best_r_x": best_r,
        "matched_with": best_txt,
        "rois": rois,
        "iteration_time": timestamp
    }
    with open(best_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
        
    print(f"✅ Success! R={best_r} > {R_THRESHOLD}. Results saved to: {best_dir}")

def main():
    print("🚀 Starting Automated ROI Search (Hill Climbing)...")
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
    
    # Pre-test current
    print("\n--- Iteration 0 (Initial Base) ---")
    write_rois(TARGET_ROI_JSON, current_best_rois)
    if run_main_cancel():
        r, txt = run_evaluation()
        if r is not None:
            print(f"Base ROI Max R: {r}")
            current_best_r = r
            if r > R_THRESHOLD:
                save_best_results(r, txt, current_best_rois)
                print("Good result found immediately. Exiting.")
                return
    
    for i in range(1, MAX_ITERS + 1):
        print(f"\n--- Iteration {i} ---")
        
        # Perturb from current best (Hill Climbing) or slightly wider 
        # Increase variance if stuck
        variance = 3 + (i % 5) # Cycles between 3,4,5,6,7 pixel std dev
        test_rois = perturb_rois(current_best_rois, variance=variance)
        
        print(f"Testing new ROI: {test_rois}")
        write_rois(TARGET_ROI_JSON, test_rois)
        
        if not run_main_cancel():
            continue
            
        r, txt = run_evaluation()
        if r is not None:
            print(f"Iter {i} Max R: {r} (vs {txt})")
            if r > current_best_r:
                print(f"🔥 New Local Best R! {r} > {current_best_r}")
                current_best_r = r
                current_best_rois = test_rois
                
            if r > R_THRESHOLD:
                save_best_results(r, txt, test_rois)
                print("Target R threshold reached. Execution Finished.")
                break

if __name__ == "__main__":
    main()
