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

# Setup path for co-tracker
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path and os.path.exists(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from core.engine import TrackingEngine
from core.sampler import Sampler
from core.analyzer import Analyzer
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks

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
    print("Sampler, Engine, Analyzer is loaded successfully!")

    # 3. Read Video
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")

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
    tracks, visibilities = engine.track_video(frames, all_points)
    print(f"Tracking complete. Shape: {tracks.shape}")
    
    # 7. Cancellation Logic
    tracks_target = tracks[:num_target]
    tracks_ref = tracks[num_target:]
    
    # Calculate Reference Motion (Camera Motion)
    # Ref shift at time t = Ref(t) - Ref(0)
    # tracks_ref shape: (N_ref, T, 2)
    # We need robust mean of shifts
    
    ref_initial = tracks_ref[:, 0:1, :] # (N_ref, 1, 2)
    ref_shifts = tracks_ref - ref_initial # (N_ref, T, 2)
    
    # Average over all reference points, ignoring NaNs
    # Note: track_video output might contain NaNs if point is lost? 
    # CoTracker usually keeps predicting, but let's be safe.
    avg_camera_shift = np.nanmean(ref_shifts, axis=0) # (T, 2)
    
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
    
    # Save CORRECTED tracks (Main Result)
    print("Saving CORRECTED results...")
    npz_path, csv_path = analyzer.save_tracks(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved corrected tracks to {npz_path}")

    # Visualize (We visualize the RAW tracking to show *what* was tracked, 
    # but we might want to indicate which were target vs ref)
    # We can visualize all tracks.
    vis_path = os.path.join(out_dir, f"tracking_vis_{base_name}.mp4")
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
    # Compute Waveform on CORRECTED data
    wave_path = analyzer.compute_and_save_waveform(f"{base_name}_corrected", corrected_tracks, fps)
    print(f"Saved Corrected Waveform to {wave_path}")
    
    # Optional: Save reference motion itself
    ref_path = os.path.join(out_dir, f"{base_name}_camera_motion.csv")
    np.savetxt(ref_path, avg_camera_shift, delimiter=",", header="dx,dy", comments="")
    print(f"Saved estimated camera motion to {ref_path}")

if __name__ == "__main__":
    main()
