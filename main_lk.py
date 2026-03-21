"""
Seismic LK Optical Flow Tracking 模块化架构
====================================

该脚本使用传统的 Lucas-Kanade (LK) 光流算法进行追踪。
它不需要深度学习模型权重，运行速度极快，适合背景简单、运动较小的场景。

使用方法：
-----------
python main_lk.py [video_file] [output_dir]
"""

import os
import sys
import argparse
import numpy as np

from core.lk_engine import LKTrackingEngine
from core.sampler import Sampler
from core.analyzer import Analyzer
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"

# Configuration
MAX_POINTS = 10  # 最佳 10 个点，可根据需要适当调整

def main():
    parser = argparse.ArgumentParser(description="Seismic LK Optical Flow Tracker")
    parser.add_argument("video", nargs="?", default=None, help="Input video filename or path")
    parser.add_argument("output", nargs="?", default=OUTPUT_DIR, help="Output directory")
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
    out_dir = os.path.join(args.output, f"{base_name}_output_lk")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"LK Processing: {inp}")
    print(f"Output to: {out_dir}")

    # 2. Logic Setup
    sampler = Sampler(roi_dir="rois")
    analyzer = Analyzer(output_dir=out_dir)
    
    # 3. Read Video
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")
    
    # 4. Point Detection
    points, points_path = sampler.load_points(base_name)
    if points is not None:
        print(f"Loaded {len(points)} points from {points_path}")
        rois = None  # No ROIs needed for point sampling
    else:
        print("No points file found. Falling back to ROI-based detection...")
        # ROI & Point Detection
        rois, roi_path = sampler.load_rois(base_name)
        if rois is None:
            print("No ROI file found. Switching to interactive selection...")
            rois = sampler.select_rois_interactive(frames[0])
            if rois and len(rois) > 0:
                sampler.save_rois(roi_path, rois)
            else:
                print("No ROIs selected. Using full image.")
                rois = None
        
        points = sampler.detect_roi_points(frames[0], rois, max_points=MAX_POINTS)
    if points.shape[0] == 0:
        print("No points detected. Exiting.")
        sys.exit(1)
    print(f"Detected {points.shape[0]} points for LK tracking.")
    
    # 5. Initialize LK Engine & Track
    engine = LKTrackingEngine()
    tracks = engine.track_video(frames, points)
    print(f"LK tracking complete. Shape: {tracks.shape}")
    
    # 6. Analyze & Visualize
    npz_path, csv_path = analyzer.save_tracks(f"{base_name}_lk", tracks, fps)
    print(f"Saved LK tracks to {npz_path}")
    
    vis_path = os.path.join(out_dir, f"tracked_lk_{base_name}.mp4")
    visibilities = ~np.isnan(tracks[:, :, 0])
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
    wave_path = analyzer.compute_and_save_waveform(f"{base_name}_lk", tracks, fps)
    print(f"Saved LK Average Waveform to {wave_path}")

if __name__ == "__main__":
    main()
