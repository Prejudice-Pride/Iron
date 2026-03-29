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
CHUNK_LEN = 100
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
    calib_path = os.path.join("output", "calibration_debug", "distortion_params.json")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            print(f"Found distortion params: {calib_path}. Applying correction...")
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
        # Source Pixels (Extracted manually)
        src_pts = [
        [1002, 544],
        [1375, 620],
        [1155, 925],
        [608, 717],]
        
        dst_pts = [
        [0, 0],
        [880, 0],
        [880, 150],
        [0, 150],]
        
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
    
    # 7. Cancellation Logic
    tracks_target = tracks[:num_target]
    tracks_ref = tracks[num_target:]
    
    # Calculate Reference Motion (Camera Motion)
    # 改进：取前几帧的均值作为参考原点，避免第一帧单帧跳变影响全局
    ref_initial = np.nanmean(tracks_ref[:, :3, :], axis=1, keepdims=True) 
    ref_shifts = tracks_ref - ref_initial # (N_ref, T, 2)
    
    # Average over all reference points, ignoring NaNs
    avg_camera_shift = np.nanmean(ref_shifts, axis=0) # (T, 2)
    # 强制将均值漂移归零，确保抵消只处理动态抖动
    avg_camera_shift -= np.nanmean(avg_camera_shift, axis=0) 
    
    # Fill any NaNs in camera shift (if all ref points lost) with 0 or previous
    # Simple forward fill for now
    mask = np.isnan(avg_camera_shift)
    if mask.any():
        print("Warning: Some frames have no valid reference tracks. Interpolating.")
        # Very basic fill
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        avg_camera_shift = avg_camera_shift[idx]

    # Correct Target Tracks
    # Target_corrected(t) = Target(t) - Camera_Shift(t)
    corrected_tracks = tracks_target - avg_camera_shift[None, :, :]
    
    print("Cancellation applied.")

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
    print("Generating cancellation comparison chart (Triple Curves)...")
    plot_path = c_analyzer.plot_comparison(base_name, tracks_target, tracks_ref, corrected_tracks, fps)
    print(f"Saved triple cancellation analysis to {plot_path}")

    # Visualize (We visualize the RAW tracking to show *what* was tracked, 
    # but we might want to indicate which were target vs ref)
    # We can visualize all tracks.
    vis_path = os.path.join(out_dir, f"tracking_vis_{base_name}.mp4")
    # 更改为基础可视化，移除侧边栏实时波形图
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
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
    np.savetxt(ref_path, avg_camera_shift, delimiter=",", header="dx,dy", comments="")
    print(f"Saved estimated camera motion to {ref_path}")

    print("\n[TIP] For advanced analysis (Intensity, Spectrum, etc.), run:")
    print(f"python analyze_results.py {npz_path}")

if __name__ == "__main__":
    main()
