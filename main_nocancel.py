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

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
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
    sampler = Sampler(roi_dir="rois")
    engine = TrackingEngine()
    analyzer = Analyzer(output_dir=out_dir)
    print("Sampler, Engine, and Analyzer are loaded successfully!")

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

    # 4. ROI Selection (Target + Reference)
    # 保留双 ROI 结构以兼容原有数据和流程，但不再执行取消逻辑
    ROI_SUFFIX = "_rois.json"

    # Try to load existing ROIs
    rois, roi_path = sampler.load_rois(base_name, roi_suffix=ROI_SUFFIX)

    if rois is None:
        print("\n--- ROI SELECTION REQUIRED ---")
        print(f"No existing ROIs found for {base_name}.")
        print("Please select exactly 2 ROIs:")
        print("  1. TARGET ROI (The object or surface to analyze).")
        print("  2. REFERENCE ROI (Stable background/building).")

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
        # Source Pixels (Extracted manually)
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]

        # Destination Physical Coordinates (mm)
        dst_calib = [
            [0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]
        ]

        analyzer.setup_calibration(src_calib, dst_calib)

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
        src_pts = [[435, 753], [1312, 728], [1515, 668], [1587, 818], [1641, 785], [1765, 733], [1806, 711]]

        dst_pts = [[0, 0], [525, 330], [875, 330], [450, 630], [600, 630], [800, 630], [950, 630]]

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
            [0, 200],
        ]

        analyzer.setup_calibration(src_pts, dst_pts)

    elif base_name == "qample2":
        print("\n[INFO] Applying Engineered Calibration for sample4.mp4...")
        # Source Pixels (Extracted manually)
        src_pts = [[1123, 953], [1133, 951], [988, 810], [998, 809], [1768, 833], [1773, 830], [1569, 759], [1574, 757], [876, 687], [880, 686], [833, 642], [840, 642], [1324, 667], [1327, 665], [1220, 627], [1224, 626]]
        dst_pts = [[0, 0], [15, 0], [0, 60], [15, 60], [865, 0], [880, 0], [865, 60], [880, 60], [0, 150], [15, 150], [0, 210], [15, 210], [865, 150], [880, 150], [865, 210], [880, 210]]

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
        print("Warning: No reference points found. Continuing without cancellation.")

    all_points = np.concatenate([pts_target, pts_ref], axis=0)
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
    visualize_tracks_with_plot(frames, tracks, visibilities, vis_path, fps, rois=rois)
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