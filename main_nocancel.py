"""
Seismic Surface Tracking
========================

This script performs basic surface tracking without reference cancellation.
It uses two ROIs:
1. TARGET ROI: The area of interest (e.g., near surface).
2. REFERENCE ROI: Kept for compatibility, but not used for cancellation.

Usage:
    python main_surface.py [video_file] [output_dir]
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
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks
from utils.self_calibration import SelfCalibration
from utils.image_enhancement import enhance_frames

INPUT_DIR = "discussion\\videos"
DEFAULT_VIDEO = "cond1.mp4"
OUTPUT_DIR = "output"

# Configuration
MAX_POINTS_TARGET = 30
MAX_POINTS_REF = 30
CHUNK_LEN = 150
OVERLAP = 20


def main():
    parser = argparse.ArgumentParser(description="Seismic Surface Tracker")
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
    out_dir = os.path.join(args.output, f"{base_name}_output")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing: {inp}")
    print(f"Output to: {out_dir}")

    # 2. Logic Setup
    engine = TrackingEngine()
    analyzer = Analyzer(output_dir=out_dir)
    print("Engine, and Analyzer are loaded successfully!")

    # 3. Read Video
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")

    # 3.a Attempt to load self-calibration parameters and apply to frames
    calib_path = os.path.join("output", "calibration_debug", "distortion_params.json")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            print(f"Found distortion params: {calib_path}. Applying correction...")
            calib_obj = SelfCalibration()
            calib_obj.k3 = float(calib.get('k3', 0.0))
            calib_obj.k5 = float(calib.get('k5', 0.0))
            cx = float(calib.get('cx', W / 2.0))
            cy = float(calib.get('cy', H / 2.0))
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

    # 4. Corner Point Sampling
    print("Loading corner points...")
    corner_file = os.path.join("corner_points", f"{base_name}_corners.json")
    if os.path.exists(corner_file):
        with open(corner_file, 'r') as f:
            pts_target = np.array(json.load(f))
        pts_ref = np.array([])  # No reference points for nocancel
    else:
        print(f"Error: Corner points file not found: {corner_file}")
        sys.exit(1)

    print(f"Loaded {len(pts_target)} Target points.")
    print(f"Loaded {len(pts_ref)} Reference points.")

    if len(pts_target) == 0:
        print("Error: No target points found.")
        sys.exit(1)

    all_points = pts_target  # Only target points
    num_target = len(pts_target)
    # ENGINEERED CALIBRATION SETUP (Sample.mp4 Specific)
    # ---------------------------------------------------------
    if base_name == "sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]

        # Destination Physical Coordinates (cm)
        dst_calib = [
            [0, 0], [1.5, 0], [1.5, 6.0], [88.0, 6.0], [88.0, 15.0], [86.5, 15.0], [88.0, 24.0]
        ]

        analyzer.setup_calibration(src_calib, dst_calib)

    elif base_name == "old_sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]

        # Destination Physical Coordinates (cm)
        dst_calib = [
            [0, 0], [15, 0], [15, 60], [880, 6.0], [880, 150], [865, 150], [880, 240]
        ]

        analyzer.setup_calibration(src_calib, dst_calib)

    elif base_name == "sample2":
        print("\n[INFO] Applying Engineered Calibration for sample2.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            [478, 482], [656, 478], [597, 436], [688, 393]
        ]

        # Destination Physical Coordinates (cm)
        dst_calib = [
            [0, 0], [22.0, 60.0], [11.0, 27.0], [22.0, 60.0]
        ]

        analyzer.setup_calibration(src_calib, dst_calib)

    elif base_name == "sample3":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[435, 753], [1312, 728], [1515, 668], [1587, 818], [1641, 785], [1765, 733], [1806, 711]]

        dst_pts = [[0, 0], [52.5, 33.0], [87.5, 33.0], [45.0, 63.0], [60.0, 63.0], [80.0, 63.0], [95.0, 63.0]]

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
            [41.5, 0],
            [41.5, 20.0],
            [0, 20.0],
        ]

        analyzer.setup_calibration(src_pts, dst_pts)

    elif base_name == "qample2":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[1123, 953], [1133, 951], [988, 810], [998, 809], [1768, 833], [1773, 830], [1569, 759], [1574, 757], [876, 687], [880, 686], [833, 642], [840, 642], [1324, 667], [1327, 665], [1220, 627], [1224, 626]]
        dst_pts = [[0, 0], [1.5, 0], [0, 6.0], [1.5, 6.0], [86.5, 0], [88.0, 0], [86.5, 6.0], [88.0, 6.0], [0, 15.0], [1.5, 15.0], [0, 21.0], [1.5, 21.0], [86.5, 15.0], [88.0, 15.0], [86.5, 21.0], [88.0, 21.0]]

        analyzer.setup_calibration(src_pts, dst_pts)
    elif base_name == "cond" :
        print("\n[INFO] Applying Engineered Calibration for cond1.mp4...")
        src_pts = [[740, 674], [678, 1015], [1057, 1021], [1026, 678]]
        # dst_pts provided in cm
        dst_pts = [[0, 0], [18.0, 0], [18.0, 14.0], [0, 14.0]]
        
        analyzer.setup_calibration(src_pts, dst_pts)

    elif base_name == "cond1" :
        print("\n[INFO] Applying Engineered Calibration for cond1.mp4...")
        src_pts = [[740, 674], [716, 804], [808, 808], [1070, 702]]
        # dst_pts provided in cm
        dst_pts = [[0, 0], [8.0, 0], [8.0, 4.0], [2.0, 16.0]]
        
        analyzer.setup_calibration(src_pts, dst_pts)
        
    elif base_name == "cond2":
        print("\n[INFO] Applying Engineered Calibration for cond2.mp4...")
        src_pts = [[688, 669], [715, 806], [802, 596], [1071, 703]]
        # dst_pts provided in cm
        dst_pts = [[0, 0], [8.0, 0], [8.0, 4.0], [2.0, 16.0]]
        analyzer.setup_calibration(src_pts, dst_pts)
        
    elif base_name == "cond3":
        print("\n[INFO] Applying Engineered Calibration for cond3.mp4...")
        src_pts = [[740, 670], [692, 937], [1049, 944], [1026, 677]]
        # dst_pts provided in cm
        dst_pts = [[0, 0], [18.0, 0], [18.0, 14.0], [0, 14.0]]
        analyzer.setup_calibration(src_pts, dst_pts)
    
    elif base_name == "cond4":
        print("\n[INFO] Applying Engineered Calibration for cond4.mp4...")
        src_pts = [[688, 669], [644, 994], [1024, 995], [981, 671]]
        # dst_pts provided in cm
        dst_pts = [[0, 0], [18.0, 0], [18.0, 14.0], [0, 14.0]]
        analyzer.setup_calibration(src_pts, dst_pts)

    else:
        print("\n[INFO] No specific calibration found for this video. Using linear fallback (1.0 cm/pixel).")

    # 5. Corner Point Sampling
    print("Loading corner points...")
    corner_file = os.path.join("corner_points", f"{base_name}_corners.json")
    if os.path.exists(corner_file):
        with open(corner_file, 'r') as f:
            pts_target = np.array(json.load(f))
        pts_ref = np.array([])  # No reference points for nocancel
    else:
        print(f"Error: Corner points file not found: {corner_file}")
        sys.exit(1)

    print(f"Loaded {len(pts_target)} Target points.")
    print(f"Loaded {len(pts_ref)} Reference points.")

    if len(pts_target) == 0:
        print("Error: No target points found.")
        sys.exit(1)

    all_points = pts_target  # Only target points
    num_target = len(pts_target)

    # 6. Tracking
    print("Tracking video...")
    tracks, visibilities = engine.track_video_chunked(frames, all_points, chunk_len=CHUNK_LEN, overlap=OVERLAP)
    print(f"Tracking complete. Shape: {tracks.shape}")

    # 7. No cancellation logic
    # Only use target tracks as the final result
    tracks_target = tracks[:num_target]

    # 修复空白帧导致的 NaNs
    for i in range(2):
        mask_t = np.isnan(tracks_target[:, :, i])
        if mask_t.any():
            idx_t = np.where(~mask_t, np.arange(mask_t.shape[1]), 0)
            np.maximum.accumulate(idx_t, axis=1, out=idx_t)
            tracks_target[:, :, i] = np.take_along_axis(tracks_target[:, :, i], idx_t, axis=1)

    print("Converting absolute trajectories from pixels to cm...")

    # 先保存一份像素单位轨迹（用于可视化或调试）
    tracks_target_raw_px = tracks_target.copy()

    # 执行物理转换
    tracks_target = analyzer.pixel_to_physical(tracks_target)

    # 8. Analyze & Visualize

    # Save RAW tracks (for debugging / comparison)
    print("Saving RAW results...")
    analyzer.save_tracks(f"{base_name}_raw", tracks_target, fps)

    # Save TARGET tracks (Main Result)
    print("Saving TARGET results...")
    npz_path, csv_path = analyzer.save_tracks(f"{base_name}_tracked", tracks_target, fps)
    print(f"Saved tracked results to {npz_path}")

    # Visualize
    vis_path = os.path.join(out_dir, f"tracking_vis_{base_name}.mp4")
    from utils.visualizer import visualize_tracks_with_plot
    visualize_tracks_with_plot(frames, tracks, visibilities, vis_path, fps)
    print(f"Saved visualization to {vis_path}")

    # Compute Waveform on TARGET data
    wave_path = analyzer.compute_and_save_waveform(f"{base_name}_tracked", tracks_target, fps)
    print(f"Saved Waveform to {wave_path}")

    # ---------------------------------------------------------
    # Advanced Analysis on Target Tracks
    # ---------------------------------------------------------
    print("\n--- Performing Advanced Analysis on Tracked Results ---")

    # 1. Plot Side-by-Side X/Y Waveforms for Target Tracks
    double_wave_path = analyzer.compute_and_save_double_waveforms(f"{base_name}_tracked", tracks_target, fps)
    print(f"Saved Double Waveform Plot to {double_wave_path}")

    # 2. Compute and save Spectrum (FFT) for Target Tracks
    spec_path = analyzer.compute_and_save_spectrum(f"{base_name}_tracked", tracks_target, fps)
    print(f"Saved Frequency Spectrum to {spec_path}")

    # 3. Calculate Multi-Dimensional Seismic Intensity for Target Tracks
    # Note: If base_name=='sample', it uses Homography, else linear fallback
    intensity_results = analyzer.calculate_intensity(tracks_target, fps)
    print(f"Intensity Analysis Complete for {base_name}.")

    print("\n[TIP] For advanced analysis (Intensity, Spectrum, etc.), run:")
    print(f"python analyze_results.py {npz_path}")


if __name__ == "__main__":
    main()