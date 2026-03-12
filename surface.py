"""
Seismic Surface Tracking (Standard)
===================================

This script provides standard surface tracking without the reference cancellation mechanism.
It focuses on a single TARGET ROI to monitor vibration or motion.

Usage:
    python main_surface_no_cancel.py [video_file] [output_dir]
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

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"

# Configuration
MAX_POINTS_TARGET = 30
CHUNK_LEN = 100
OVERLAP = 10

def main():
    parser = argparse.ArgumentParser(description="Seismic Surface Tracker (Standard)")
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
    out_dir = os.path.join(args.output, f"{base_name}_output_surface")
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
            cx = float(calib.get('cx', W/2.0))
            cy = float(calib.get('cy', H/2.0))
            calib_obj.center = (cx, cy)

            # Apply correction to each frame
            corrected_frames = []
            print('Applying distortion correction to all frames...')
            for i in range(frames.shape[0]):
                corrected = calib_obj.apply_to_image(frames[i])
                corrected_frames.append(corrected)
            frames = np.stack(corrected_frames, axis=0)
            print('Distortion correction applied to frames.')
        except Exception as e:
            print(f"Warning: Failed to load/apply distortion params: {e}")
    else:
        print(f"No distortion params file found at {calib_path}. Skipping auto-correction.")

    # 4. ROI Selection (Target only)
    ROI_SUFFIX = "_surface_rois.json"
    
    # Try to load existing ROIs
    rois, roi_path = sampler.load_rois(base_name, roi_suffix=ROI_SUFFIX)
    
    if rois is None:
        print("\n--- ROI SELECTION REQUIRED ---")
        print(f"No existing surface ROIs found for {base_name}.")
        print("Please select the TARGET ROI (The area of interest).")
        
        can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
        if not can_interactive:
            print("Error: Running in headless mode and no ROI file found.")
            sys.exit(1)
            
        rois = sampler.select_rois_interactive(frames[0], window_name="Select TARGET ROI")
        
        if not rois or len(rois) == 0:
            print("Error: No ROI selected.")
            sys.exit(1)
            
        sampler.save_rois(roi_path, rois)
        print(f"Saved ROIs to {roi_path}")
    else:
        print(f"Loaded rois from {roi_path}")

    # ---------------------------------------------------------
    # ENGINEERED CALIBRATION SETUP (Sample.mp4 Specific)
    # ---------------------------------------------------------
    if base_name == "sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]
        dst_calib = [
            [0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]
        ]
        analyzer.setup_calibration(src_calib, dst_calib)
    else:
        print("\n[INFO] No specific calibration found. Using linear fallback.")

    # 5. Point Detection
    print("Detecting points...")
    pts_target = sampler.detect_roi_points_orb(frames[0], [rois[0]], max_points=MAX_POINTS_TARGET)
    
    print(f"Detected {len(pts_target)} Target points.")
    
    if len(pts_target) == 0:
        print("Error: No target points found.")
        sys.exit(1)

    # 6. Tracking
    print("Tracking video...")
    tracks, visibilities = engine.track_video_chunked(frames, pts_target, chunk_len=CHUNK_LEN, overlap=OVERLAP)
    print(f"Tracking complete. Shape: {tracks.shape}")
    
    # 7. Analyze & Visualize
    print("Saving results...")
    npz_path, csv_path = analyzer.save_tracks(f"{base_name}_surface", tracks, fps)
    print(f"Saved tracks to {npz_path}")

    # Visualize
    vis_path = os.path.join(out_dir, f"tracking_vis_{base_name}.mp4")
    from utils.visualizer import visualize_tracks_with_plot
    visualize_tracks_with_plot(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
    # Compute Waveform
    wave_path = analyzer.compute_and_save_waveform(f"{base_name}_surface", tracks, fps)
    print(f"Saved Waveform to {wave_path}")

    # ---------------------------------------------------------
    # Advanced Analysis
    # ---------------------------------------------------------
    print("\n--- Performing Advanced Analysis ---")
    
    # 1. Plot Side-by-Side X/Y Waveforms
    double_wave_path = analyzer.compute_and_save_double_waveforms(f"{base_name}_surface", tracks, fps)
    print(f"Saved Double Waveform Plot to {double_wave_path}")

    # 2. Compute and save Spectrum (FFT)
    spec_path = analyzer.compute_and_save_spectrum(f"{base_name}_surface", tracks, fps)
    print(f"Saved Frequency Spectrum to {spec_path}")

    # 3. Calculate Multi-Dimensional Seismic Intensity
    intensity_results = analyzer.calculate_intensity(tracks, fps)
    print(f"Intensity Analysis Complete for {base_name}.")

    print("\n[TIP] For advanced analysis (Intensity, Spectrum, etc.), run:")
    print(f"python analyze_results.py {npz_path}")

if __name__ == "__main__":
    main()
