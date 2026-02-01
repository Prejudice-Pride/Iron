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

    wave_path = analyzer.compute_and_save_waveform(base_name, tracks, fps)
    print(f"Saved Average Waveform to {wave_path}")


if __name__ == "__main__":
    main()
