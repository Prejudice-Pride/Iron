"""
Seismic Analysis Tool
=====================

This script performs post-processing analysis on tracking results saved as .npz files.
It computes waveforms, frequency spectrums, and seismic intensity.

Usage:
    python analyze_results.py [track_file.npz] [output_dir] --fps [fps] --pixels_to_mm [value]
"""

import os
import sys
import argparse
import numpy as np
from core.analyzer import Analyzer

def main():
    parser = argparse.ArgumentParser(description="Analyze seismic tracking results")
    parser.add_argument("tracks", help="Path to the .npz track file")
    parser.add_argument("output", nargs="?", default=None, help="Output directory (defaults to track file dir)")
    parser.add_argument("--fps", type=float, default=None, help="Override FPS from file")
    parser.add_argument("--pixels_to_mm", type=float, default=10.0, help="Linear calibration factor (mm/pixel)")
    
    args = parser.parse_args()

    # 1. Load Tracks
    if not os.path.exists(args.tracks):
        print(f"Error: Track file not found: {args.tracks}")
        sys.exit(1)
        
    data = np.load(args.tracks)
    tracks = data['tracks']
    file_fps = data['fps']
    
    fps = args.fps if args.fps is not None else file_fps
    
    # 2. Setup Output
    track_dir = os.path.dirname(os.path.abspath(args.tracks))
    out_dir = args.output if args.output else track_dir
    os.makedirs(out_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(args.tracks))[0]
    # Remove common suffixes if present to get clean base name
    clean_name = base_name.replace("_surface_tracks", "").replace("_corrected", "").replace("_raw", "").replace("_reference", "")

    print(f"Analyzing: {base_name}")
    print(f"Output to: {out_dir}")
    print(f"FPS: {fps}")

    # 3. Initialize Analyzer
    analyzer = Analyzer(output_dir=out_dir)
    
    # Apply special calibration for sample video if recognized in name
    if "sample" in clean_name.lower():
        print("[INFO] Applying Engineered Calibration for sample data...")
        src_calib = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
        ]
        dst_calib = [
            [0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]
        ]
        analyzer.setup_calibration(src_calib, dst_calib)
    
    # 4. Perform Analysis
    print("Computing waveforms...")
    wave_path = analyzer.compute_and_save_waveform(base_name, tracks, fps)
    print(f" - CSV Waveform: {wave_path}")
    
    double_wave_path = analyzer.compute_and_save_double_waveforms(base_name, tracks, fps)
    print(f" - Plot Waveforms: {double_wave_path}")

    print("Computing frequency spectrum...")
    spec_path = analyzer.compute_and_save_spectrum(base_name, tracks, fps)
    print(f" - Spectrum: {spec_path}")

    print("Calculating seismic intensity...")
    intensity_results = analyzer.calculate_intensity(tracks, fps, pixels_to_mm=args.pixels_to_mm)
    
    print("\nAnalysis Complete.")

if __name__ == "__main__":
    main()
