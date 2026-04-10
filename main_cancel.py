"""
Seismic Surface Tracking with Reference Cancellation
===================================================

This script extends the basic surface tracking by introducing a reference cancellation mechanism.
It uses two ROIs:
1. TARGET ROI: The area of interest (e.g., near surface).
2. REFERENCE ROI: A stable background area (e.g., distant building).

The motion of the reference points is assumed to be due to camera movement only.
This camera movement is subtracted from the target points to isolate vibration/motion.

Usage:
    python main_surface_cancel.py [video_file] [output_dir]
"""

import os
import sys
import argparse
import numpy as np
import json

# 解决 OpenMP 运行时冲突 (OMP: Error #15)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setup path for co-tracker
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path and os.path.exists(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from core.engine import TrackingEngine
from core.sampler import Sampler
from core.analyzer import Analyzer
from core.cancel_analyzer import CancelAnalyzer
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks
from utils.self_calibration import SelfCalibration
from utils.image_enhancement import enhance_frames

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"

# Configuration
MAX_POINTS_TARGET = 30
MAX_POINTS_REF = 30 
CHUNK_LEN = 150
OVERLAP = 10

def main():
    parser = argparse.ArgumentParser(description="Seismic Surface Tracker with Reference Cancellation")
    parser.add_argument(
        "video", nargs="?", default=None, help="Input video filename or path"
    )
    parser.add_argument(
        "output", nargs="?", default=OUTPUT_DIR, help="Output directory"
    )
    args = parser.parse_args()

    # 1. Resolve Input
    if args.video:
        inp = args.video
        if not os.path.exists(inp):
            candidate = os.path.join(INPUT_DIR, inp)
            if os.path.exists(candidate):
                inp = candidate
            else:
                print(f"Error: Video not found {inp}")
                sys.exit(1)
    else:
        inp = os.path.join(INPUT_DIR, DEFAULT_VIDEO)
        if not os.path.exists(inp):
            print(f"Error: Default video not found {inp}")
            sys.exit(1)

    base_name = os.path.splitext(os.path.basename(inp))[0]
    out_dir = os.path.join(args.output, f"{base_name}_output_cancel")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing: {inp}")
    print(f"Output to: {out_dir}")

    # 2. Logic Setup
    sampler = Sampler(roi_dir="rois")
    engine = TrackingEngine()
    analyzer = Analyzer(output_dir=out_dir)
    c_analyzer = CancelAnalyzer(output_dir=out_dir)
    print("Sampler, Engine, Analyzer, and CancelAnalyzer are loaded successfully!")

    # 3. Read Video
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")

    # 3.a Attempt to load self-calibration parameters and apply to frames
    # Try video-specific calibration first, then fallback to global
    calib_filename = f"{base_name}_distortion_params.json"
    calib_path = os.path.join("output", "calibration_debug", calib_filename)
    global_calib_path = os.path.join("output", "calibration_debug", "distortion_params.json")
    
    if not os.path.exists(calib_path) and os.path.exists(global_calib_path):
        print(f"Specific calibration {calib_path} not found. Using global {global_calib_path}")
        calib_path = global_calib_path

    if os.path.exists(calib_path):
        try:
            with open(calib_path, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            print(f"Found distortion params: {calib_path}. Applying correction...")
            print(f"Distortion Data: {json.dumps(calib, indent=4)}")
            calib_obj = SelfCalibration()
            calib_obj.k3 = float(calib.get('k3', 0.0))
            calib_obj.k5 = float(calib.get('k5', 0.0))
            cx = float(calib.get('cx', W/2.0))
            cy = float(calib.get('cy', H/2.0))
            calib_obj.center = (cx, cy)

            # Apply correction to each frame (may be slow for long videos)
            corrected_frames = []
            print('Applying distortion correction to all frames (this may take a while)...')
            for i in range(frames.shape[0]):
                corrected = calib_obj.apply_to_image(frames[i])
                corrected_frames.append(corrected)
            frames = np.stack(corrected_frames, axis=0)
            print('Distortion correction applied to frames.')
        except Exception as e:
            print(f"Warning: Failed to load/apply distortion params: {e}")
    else:
        print(f"No distortion params file found at {calib_path}. Skipping auto-correction.")

    # 3.b 图像增强滤波
    print("\n--- Applying Image Enhancement ---")
    # 这里默认启用 CLAHE 局部对比度增强，这有助于更稳定地检测弱纹理表面的角点
    frames = enhance_frames(frames, use_clahe=True, use_bilateral=False)

    # 4. ROI Selection (Target + Reference)
    # We use a specific suffix for cancellation ROIs to avoid conflict with standard surface ROIs
    ROI_SUFFIX = "_cancel_rois.json"
    
    # Try to load existing ROIs
    rois, roi_path = sampler.load_rois(base_name, roi_suffix=ROI_SUFFIX)
    
    if rois is None:
        print("\n--- ROI SELECTION REQUIRED ---")
        print(f"No existing cancellation ROIs found for {base_name}.")
        print("Please elect exactly 2 ROIs:")
        print("  1. TRAGET ROI (The object or surface to analyze).")
        print("  2. REFERENCE ROI (Stable background/building for cancellation).")
        
        # Check interactivity
        can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
        if not can_interactive:
            print("Error: Running in headless mode and no ROI file found.")
            sys.exit(1)
            
        rois = sampler.select_rois_interactive(frames[0], window_name="Select TARGET then REFERENCE")
        
        if not rois or len(rois) != 2:
            print(f"Error: Expected exactly 2 ROIs, got {len(rois) if rois else 0}.")
            print("Please run again and select exactly 2 boxes.")
            sys.exit(1)
            
        sampler.save_rois(roi_path, rois)
        print(f"Saved ROIs to {roi_path}")
    else:
        print(f"Loaded rois from {roi_path}")
        if len(rois) != 2:
            print(f"Warning: Loaded ROI file contains {len(rois)} ROIs. Expected 2. Using first two.")
            if len(rois) < 2:
                print("Error: Insufficient ROIs.")
                sys.exit(1)
            rois = rois[:2]

    # ---------------------------------------------------------
    # ENGINEERED CALIBRATION SETUP (Sample.mp4 Specific)
    # ---------------------------------------------------------
    if base_name == "sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        src_pts = [
                [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620], [610, 719]
            ]
            
            # Destination Physical Coordinates (mm)
        dst_pts = [
                [0, 0], [5, 0], [5, 60], [135, 60], [135, 150], [130, 150], [135, 210], [0,60]
            ]
        
        analyzer.setup_calibration(src_pts, dst_pts)
    elif base_name == "old_sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]
        
        # Destination Physical Coordinates (mm)
        dst_calib = [
            [0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]
        ]
        
        analyzer.setup_calibration(src_calib, dst_calib)
        
    elif base_name == "sample2":
        print("\n[INFO] Applying Engineered Calibration for sample2.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            [478, 482], [656, 478], [597, 436], [688, 393]
        ]
        
        # Destination Physical Coordinates (mm)
        dst_calib = [
            [0, 0], [220, 600], [110, 270], [220, 600]
        ]
        
        analyzer.setup_calibration(src_calib, dst_calib)
        
    elif base_name == "sample3":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[438, 751], [809, 665], [1036, 611], [1218, 568], [1313, 729], [876, 908]]
        dst_pts = [[0,0],[250,0],[500,0],[750,0],[375,100],[0,100]]
        
        analyzer.setup_calibration(src_pts, dst_pts)
        
    elif base_name == "sample4":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [
            [402, 430],
            [1045, 398],
            [1300, 594],
            [500, 623], 
        ]
        [[1121, 953], [1133, 951], [990, 811], [998, 809], [1765, 833], [1771, 830], [1568, 759], [1575, 757], [873, 686], [882, 685], [833, 641], [841, 641]]
        dst_pts = [
        [0, 0],
        [415, 0],
        [415, 200],
        [0, 200],]
        
        analyzer.setup_calibration(src_pts, dst_pts)
    elif base_name == "qample":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[308, 1002], [325, 1003], [729, 960], [742, 960], [567, 753], [579, 754], [829, 743], [838, 743], [655, 668], [665, 668], [864, 662], [876, 662]]
        dst_pts = [[0,0],[5,0],[65,15],[70,15],[0,90],[5,90],[65,105],[75,105],[0,150],[5,150],[65,165],[70,165]]

        analyzer.setup_calibration(src_pts, dst_pts)
        
    elif base_name == "qample2":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[1123, 953], [1133, 951], [988, 810], [998, 809], [1768, 833], [1773, 830], [1569, 759], [1574, 757], [876, 687], [880, 686], [833, 642], [840, 642], [1324, 667], [1327, 665], [1220, 627], [1224, 626]]
        dst_pts = [[0, 0], [15, 0], [0, 60], [15, 60], [865, 0], [880, 0], [865, 60], [880, 60], [0, 150], [15, 150], [0, 210], [15, 210], [865, 150], [880, 150], [865, 210], [880, 210]]

        analyzer.setup_calibration(src_pts, dst_pts)
        
    elif base_name == "pample2":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[320, 846], [364, 849], [693, 878], [740, 881], [315, 780], [333, 768], [582, 606], [594, 597]]
        dst_pts = [[40,0],[85,0],[355,0],[400,0],[0,60],[0,105],[0,585],[0,630]]

        analyzer.setup_calibration(src_pts, dst_pts)
    else:
        print("\n[INFO] No specific calibration found for this video. Using linear fallback (10mm/pixel).")

    # 5. Point Detection
    # Detect independently to know which is which
    print("Detecting points...")
    roi_target = rois[0]
    roi_ref = rois[1]
    
    pts_target = sampler.detect_roi_points_orb(frames[0], [roi_target], max_points=MAX_POINTS_TARGET)
    pts_ref = sampler.detect_roi_points_orb(frames[0], [roi_ref], max_points=MAX_POINTS_REF)
    
    print(f"Detected {len(pts_target)} Target points.")
    print(f"Detected {len(pts_ref)} Reference points.")
    
    if len(pts_target) == 0:
        print("Error: No target points found.")
        sys.exit(1)
    if len(pts_ref) == 0:
        print("Error: No reference points found. Cannot perform cancellation.")
        sys.exit(1)

    all_points = np.concatenate([pts_target, pts_ref], axis=0)
    num_target = len(pts_target)
    
    # 6. Tracking
    print("Tracking video...")
    tracks, visibilities = engine.track_video_chunked(frames, all_points, chunk_len=CHUNK_LEN, overlap=OVERLAP)
    print(f"Tracking complete. Shape: {tracks.shape}")
    
    # 7. Cancellation Logic (with Assessment & Smoothing)
    tracks_target = tracks[:num_target]
    tracks_ref = tracks[num_target:]
    
    # --- A. 原始运动解算 ---
    # 1. 提取参考框(Camera)全局位移
    ref_initial = np.nanmean(tracks_ref[:, :3, :], axis=1, keepdims=True) 
    ref_shifts = tracks_ref - ref_initial
    avg_camera_shift = np.nanmean(ref_shifts, axis=0) # (T, 2)
    avg_camera_shift -= np.nanmean(avg_camera_shift, axis=0) 
    
    # 2. 提取目标框(Target)全局位移 (用于一致性检验)
    target_initial = np.nanmean(tracks_target[:, :3, :], axis=1, keepdims=True)
    target_shifts = tracks_target - target_initial
    avg_target_shift = np.nanmean(target_shifts, axis=0)
    avg_target_shift -= np.nanmean(avg_target_shift, axis=0)

    # 修复空白帧导致的 NaNs
    for i in range(2):
        mask_c = np.isnan(avg_camera_shift[:, i])
        if mask_c.any():
            idx_c = np.where(~mask_c, np.arange(mask_c.shape[0]), 0)
            np.maximum.accumulate(idx_c, out=idx_c)
            avg_camera_shift[:, i] = avg_camera_shift[idx_c, i]
            
        mask_t = np.isnan(avg_target_shift[:, i])
        if mask_t.any():
            idx_t = np.where(~mask_t, np.arange(mask_t.shape[0]), 0)
            np.maximum.accumulate(idx_t, out=idx_t)
            avg_target_shift[:, i] = avg_target_shift[idx_t, i]

    # --- B. 相关性评估模块 (Axis-Independent Gating Logic) ---
    print("\n--- Evaluating Motion Correlation ---")
    def safe_corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6: return 0.0
        return np.corrcoef(a, b)[0, 1]

    corr_x = safe_corr(avg_target_shift[:, 0], avg_camera_shift[:, 0])
    corr_y = safe_corr(avg_target_shift[:, 1], avg_camera_shift[:, 1])
    print(f"X-Axis Motion Correlation (Target vs Ref): {corr_x:.4f}")
    print(f"Y-Axis Motion Correlation (Target vs Ref): {corr_y:.4f}")

    # 为 X 轴和 Y 轴分别计算独立的自适应抵消权重
    # 如果相关性 < 0.1，权重视为 0（即视为噪音，不进行抵消，保护原数据）
    # 如果相关性 > 0.5，权重为 100% 抵消
    def get_axis_weight(corr):
        if corr > 0.5:
            return 1.0
        elif corr > 0.1:
            return (corr - 0.1) / 0.4
        else:
            return 0.0

    weight_x = get_axis_weight(corr_x)
    weight_y = get_axis_weight(corr_y)
    
    print(f"[评估] X轴抵消权重: {weight_x:.2f} | Y轴抵消权重: {weight_y:.2f}")

    # 创建 [T, 2] 的动态掩码矩阵
    cancel_weight_matrix = np.array([weight_x, weight_y])

    # --- C. 亚像素平滑滤波 (Low-pass Filter) ---
    # 移除强制高斯平滑，让原汁原味的相机高频抖动能够去抵消目标中的同频抖动
    print("[滤噪] 移除对参考框位移的高斯平滑滤波处理，保留高频相机抖动以进行完全抵消。")
    avg_camera_shift_smooth = avg_camera_shift

    # --- D. 修正核心轨迹应用 (独立分通道抵消) ---
    # 这一步非常关键：分别乘以对应的轴向权重，对于负相关轴，偏移直接置0
    final_cancellation = avg_camera_shift_smooth * cancel_weight_matrix[None, :]
    
    # 覆盖原变量以便于后续输出文件能够正确记录“最终真正被减掉”的有效位移
    # 注意：此时 avg_camera_shift 是像素单位的相对位移量 [T, 2]
    avg_camera_shift = final_cancellation
    
    # 将计算好的干净趋势，从目标追踪轨迹中减除（此时完全在原始绝对像素坐标下）
    corrected_tracks = tracks_target - avg_camera_shift[None, :, :]

    # =========================================================
    # APPLY HOMOGRAPHY CONVERSION FIRST (Absolute Pixel -> Absolute cm)
    # =========================================================
    print("Applying Homography Transformation: Converting absolute trajectories from pixels to cm...")
    
    # 先保存一份像素单位的参考位移图 (用于 CancelAnalyzer 绘图，避免非线性缩放导致幅值异常)
    tracks_ref_px = tracks_ref.copy()
    tracks_target_raw_px = tracks_target.copy()
    
    # 执行物理转换
    tracks_target = analyzer.pixel_to_physical(tracks_target)
    tracks_ref = analyzer.pixel_to_physical(tracks_ref)
    corrected_tracks = analyzer.pixel_to_physical(corrected_tracks)

    # 计算物理空间下真正的参考位移轨迹 (用于保存 csv)
    # 我们用 tracks_target (物理) - corrected_tracks (物理) 得到真正被减掉的 cm 数
    avg_camera_shift_cm = np.nanmean(tracks_target - corrected_tracks, axis=0)

    print("Cancellation module completed (Pure movement, detrending removed).")

    # 8. Analyze & Visualize
    
    # Save RAW tracks (for debugging / comparison)
    print("Saving RAW results...")
    analyzer.save_tracks(f"{base_name}_raw", tracks_target, fps)

    # Save REFERENCE tracks (The cancellation/reference box)
    print("Saving REFERENCE results...")
    ref_track_path, _ = analyzer.save_tracks(f"{base_name}_reference", tracks_ref, fps)
    print(f"Saved reference tracks to {ref_track_path}")
    
    # Save CORRECTED tracks (Main Result)
    print("Saving CORRECTED results...")
    npz_path, csv_path = analyzer.save_tracks(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved corrected tracks to {npz_path}")

    # Generate Triple Comparison Plot
    # [关键修复]：要求绘图使用像素单位，这里传入像素轨迹
    print("Generating cancellation comparison chart (Triple Curves in PIXELS)...")
    # 我们需要构建像素单位的 corrected_tracks
    corrected_tracks_px = tracks_target_raw_px - avg_camera_shift[None, :, :]
    plot_path = c_analyzer.plot_comparison(base_name, tracks_target_raw_px, tracks_ref_px, corrected_tracks_px, fps)
    print(f"Saved triple cancellation analysis (pixels) to {plot_path}")

    # Visualize (We visualize the RAW tracking to show *what* was tracked, 
    # but we might want to indicate which were target vs ref)
    # We can visualize all tracks.
    # Visualization video generation is disabled per request.
    vis_path = os.path.join(out_dir, f"tracking_vis_{base_name}.mp4")
    from utils.visualizer import visualize_tracks
    # 修改：使用 visualize_tracks (无折线图版本) 并增加颜色渲染逻辑
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
    # Compute Waveform on RAW data
    raw_wave_path = analyzer.compute_and_save_waveform(f"{base_name}_raw", tracks_target, fps)
    print(f"Saved Raw Waveform to {raw_wave_path}")

    # Compute Waveform on CORRECTED data
    wave_path = analyzer.compute_and_save_waveform(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved Corrected Waveform to {wave_path}")

    # Compute Waveform on REFERENCE data (Camera/Reference Motion)
    ref_wave_path = analyzer.compute_and_save_waveform(f"{base_name}_reference", tracks_ref, fps)
    print(f"Saved Reference Waveform to {ref_wave_path}")
    
    # ---------------------------------------------------------
    # NEW: Advanced Analysis on Corrected Tracks
    # ---------------------------------------------------------
    print("\n--- Performing Advanced Analysis on Corrected Tracks ---")
    
    # 1. Plot Side-by-Side X/Y Waveforms for Corrected Tracks
    double_wave_path = analyzer.compute_and_save_double_waveforms(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved Double Waveform Plot (Corrected) to {double_wave_path}")

    # 2. Compute and save Spectrum (FFT) for Corrected Tracks
    spec_path = analyzer.compute_and_save_spectrum(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved Frequency Spectrum (Corrected) to {spec_path}")

    # 3. Calculate Multi-Dimensional Seismic Intensity for Corrected Tracks
    # Note: If base_name=='sample', it uses Homography, else linear fallback
    intensity_results = analyzer.calculate_intensity(corrected_tracks, fps)
    print(f"Intensity Analysis Complete for {base_name}.")

    # Optional: Save reference motion itself
    ref_path = os.path.join(out_dir, f"{base_name}_camera_motion.csv")
    np.savetxt(ref_path, avg_camera_shift_cm, delimiter=",", header="dx_cm,dy_cm", comments="")
    print(f"Saved estimated camera motion in cm to {ref_path}")

    print("\n[TIP] For advanced analysis (Intensity, Spectrum, etc.), run:")
    print(f"python analyze_results.py {npz_path}")

if __name__ == "__main__":
    main()