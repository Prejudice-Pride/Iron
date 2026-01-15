"""
Seismic Online Surface Tracking 模块化架构
====================================

该脚本演示了基于 CoTracker Online 模型的实时/伪在线追踪流程。
它使用 core.online_engine 模块进行帧序列的流式处理。

使用方法：
-----------
python main_online.py [video_file] [output_dir]
"""

import os
import sys
import argparse

# Setup path for co-tracker
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path and os.path.exists(REPO_DIR):
    sys.path.insert(0, REPO_DIR)

from core.online_engine import OnlineTrackingEngine
from core.sampler import Sampler
from core.analyzer import Analyzer
from utils.video_io import read_frames
from utils.visualizer import visualize_tracks

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"
# 注意：在线模式建议使用 scaled_online.pth 权重
CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "scaled_online.pth")

# Configuration
MAX_POINTS = 500

def main():
    parser = argparse.ArgumentParser(description="Seismic Online Surface Tracker")
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
    out_dir = os.path.join(args.output, f"{base_name}_output_online")
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Online Processing: {inp}")
    print(f"Output to: {out_dir}")

    # 2. Logic Setup
    sampler = Sampler(roi_dir="rois")
    analyzer = Analyzer(output_dir=out_dir)
    
    # 3. Read Video (模拟在线，实际还是先载入全量或流式读取)
    print("Reading frames...")
    frames, fps = read_frames(inp)
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")
    
    # 4. ROI & Point Detection
    rois, roi_path = sampler.load_rois(base_name)
    if rois is None:
        # 如果是真正的在线，这里可能需要预设 ROI 或第一帧交互
        print("No ROI file found. Switching to interactive selection (First frame)...")
        rois = sampler.select_rois_interactive(frames[0])
        if rois and len(rois) > 0:
            sampler.save_rois(roi_path, rois)
        else:
            print("No ROIs selected. Using full image.")
    
    points = sampler.detect_roi_points(frames[0], rois, max_points=MAX_POINTS)
    if points.shape[0] == 0:
        print("No points detected. Exiting.")
        sys.exit(1)
    print(f"Detected {points.shape[0]} points to track online.")
    
    # 5. Initialize ONLINE Engine & Track
    try:
        engine = OnlineTrackingEngine(checkpoint_path=CKPT_PATH)
    except FileNotFoundError as e:
        print(f"Checkpoint Error: {e}")
        print(f"Please ensure {CKPT_PATH} exists.")
        sys.exit(1)

    # 模拟在线流式追踪
    tracks = engine.track_video_online(frames, points)
    print(f"Online tracking complete. Shape: {tracks.shape}")
    
    # 6. Analyze & Visualize
    npz_path, csv_path = analyzer.save_tracks(base_name + "_online", tracks, fps)
    print(f"Saved online tracks to {npz_path}")
    
    vis_path = os.path.join(out_dir, f"tracked_online_{base_name}.mp4")
    visualize_tracks(frames, tracks, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")
    
    wave_path = analyzer.compute_and_save_waveform(base_name + "_online", tracks, fps)
    print(f"Saved Online Average Waveform to {wave_path}")

if __name__ == "__main__":
    main()
