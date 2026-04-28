"""
AllTracker Seismic Surface Tracking
====================================

该脚本使用 AllTracker 引擎替代 CoTracker 进行高分辨率、全像素覆盖的震动追踪。

模块功能简介：
---------------
1. core.all_engine (AllTracker 引擎):
   - 基于 AllTracker 实现的高精度追踪。
   - 通过全图流场采样实现对指定角点的稀疏追踪。

2. core.sampler (采样管理):
   - 同 main_surface.py，负责 ROI 选择与角点检测。

3. core.analyzer (数据分析):
   - 提取并保存追踪轨迹及震动波形数据。

使用方法：
-----------
python main_alltracker.py [video_file] [output_dir] --tiny --image_size 1024
"""

import os
import sys
import argparse
import numpy as np

# Setup path for AllTracker
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "alltracker"))
if REPO_DIR not in sys.path and os.path.exists(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from core.all_engine import AllTrackingEngine
from core.sampler import Sampler
from core.analyzer import Analyzer
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"

# Configuration
MAX_POINTS = 30


def main():
    parser = argparse.ArgumentParser(description="AllTracker Seismic Surface Tracker")
    parser.add_argument(
        "video", nargs="?", default=None, help="Input video filename or path"
    )
    parser.add_argument(
        "output", nargs="?", default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--tiny", action="store_true", help="Use the tiny version of AllTracker"
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="Max dimension for inference"
    )
    parser.add_argument(
        "--iters", type=int, default=4, help="Inference iterations"
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
    # 使用 alltracker 前缀区分输出目录
    out_dir = os.path.join(args.output, f"{base_name}_output_alltracker")
    os.makedirs(out_dir, exist_ok=True)

    print(f"Processing: {inp}")
    print(f"Output to: {out_dir}")

    # 2. Logic Setup
    sampler = Sampler(roi_dir="rois")
    # 初始化 AllTracker 引擎
    engine = AllTrackingEngine(tiny=args.tiny)
    analyzer = Analyzer(output_dir=out_dir)
    print("Sampler, AllTrackerEngine, Analyzer loaded successfully!")

    # 3. Read Video
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")

    # 4. ROI & Grid Generation (角点坐标作为采样点)
    rois = sampler.get_or_select_rois(base_name, frames[0])
    if rois is None:
        print("ROI is unavailable.")
        sys.exit(1)

    # ---------------------------------------------------------
    # ENGINEERED CALIBRATION SETUP (Sample.mp4 Specific)
    # ---------------------------------------------------------
    if base_name == "sample":
        print("\n[INFO] Applying Engineered Calibration for sample.mp4...")
        # Source Pixels (Extracted manually)
        src_calib = [
            # Group 1: Left Lane Line 1 (ll1) - Bottom Left
            [297, 860], [309, 863], [609, 718], [620, 721],
            # Group 2: Right Lane Line 2 (rl2) - Top Right
            [1309, 701], [1319, 702], [1367, 620], [1376, 621]
        ]
        
        # Destination Physical Coordinates (mm)
        # Origin (0,0) is bottom-left corner of ll1
        # X axis: Along road (longitudinal)
        # Y axis: Across road (lateral, towards right lane)
        # Lane Width: 3500mm, Line Gap: 4000mm, Line Len: 2000mm, DoubleYellow: 450mm
        # Y_offset to rl2 = LeftLane1 + DoubleYellow + RightLane1 = 3500 + 450 + 3500 = 7450mm
        # X_offset to rl2 (2nd line) = Line1(2000) + Gap(4000) = 6000mm
        
        dst_calib = [
            # Group 1: ll1 (Reference)
            [0.0, 0.0], [0, 150], [2000, 0], [2000, 150],
            # Group 2: rl2 (Offset)
            [6000, 7450], [6000, 7600], [8000, 7450], [8000, 7600]
        ]
        
        analyzer.setup_calibration(src_calib, dst_calib)
    else:
        print("\n[INFO] No specific calibration found for this video. Using linear fallback (10mm/pixel).")

    # 仍然使用角点检测，这些坐标将作为 AllTracker Dense Map 的采样索引
    points = sampler.detect_roi_points_orb(frames[0], rois, max_points=MAX_POINTS)
    if points.shape[0] == 0:
        print("No points detected within ROI. Exiting.")
        sys.exit(1)
    print(f"Detected {points.shape[0]} corner points in ROI for sampling.")

    # 5. Initialize Engine & Track (全图推理后采样)
    tracks, visibilities = engine.track_video(
        frames, 
        points, 
        inference_iters=args.iters, 
        image_size=args.image_size
    )
    print(f"Tracking complete. Shape: {tracks.shape}")

    # 6. Analyze & Visualize
    npz_path, csv_path = analyzer.save_tracks(base_name, tracks, fps)
    print(f"Saved tracks to {npz_path}")

    vis_path = os.path.join(out_dir, f"tracked_alltracker_{base_name}.mp4")
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")

    # 1. Save Waveform (Now includes both X and Y for all points in CSV)
    wave_path = analyzer.compute_and_save_waveform(base_name, tracks, fps)
    print(f"Saved Average Waveform to {wave_path}")
    
    # 2. Plot Side-by-Side X/Y Waveforms
    double_wave_path = analyzer.compute_and_save_double_waveforms(base_name, tracks, fps)
    print(f"Saved Double Waveform Plot to {double_wave_path}")

    # 3. Compute and save spectrum (Now includes FFT for both X and Y)
    spec_path = analyzer.compute_and_save_spectrum(base_name, tracks, fps)
    print(f"Saved Frequency Spectrum to {spec_path}")

    # 4. Calculate Seismic Intensity (Intensity Calculation at the end)
    # Note: pixels_to_mm depends on your scene, set a default or calibrated value here
    # If calibration was setup, pixels_to_mm argument is ignored in favor of Homography
    intensity_results = analyzer.calculate_intensity(tracks, fps, pixels_to_mm=10.0) 


if __name__ == "__main__":
    main()
