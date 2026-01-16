"""
Seismic Surface Tracking 模块化架构
====================================

该脚本将视频震动追踪处理流程拆分为多个功能模块，以便于在不同追踪算法与分析策略间进行切换。

模块功能简介：
---------------
1. core.engine (追踪引擎):
   - 负责核心运动追踪逻辑的生命周期管理。
   - 处理视频流与追踪点之间的推理交互，确保在时间序列上的运动连续性。

2. core.sampler (采样管理):
   - 负责确定和生成视频中需要追踪的目标坐标。
   - 管理感知区域 (ROI) 的定义，并根据图像特征或预设规则生成初始追踪点位。

3. core.analyzer (数据分析):
   - 负责对生成的原始轨迹数据进行物理意义上的后处理与统计计算。
   - 提取目标物体的震动特征，并将计算结果持久化为结构化数据文件。

4. utils.video_io (视频输入输出):
   - 封装底层多媒体库，提供统一的视频帧读取接口。

5. utils.visualizer (结果可视化):
   - 负责将追踪数据与原始视频进行融合渲染。
   - 生成直观的可视化报告，展示追踪点的运动路径及其对应的采样区域。

使用方法：
-----------
python main_surface.py [video_file] [output_dir]
"""

import os
import sys
import argparse

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
MAX_POINTS = 30
CHUNK_LEN = 100
OVERLAP = 10


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
    out_dir = os.path.join(args.output, f"{base_name}_output_surface")
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

    # 4. ROI & Grid Generation
    rois = sampler.get_or_select_rois(base_name, frames[0])
    if rois is None:
        print("ROI is unavailable.")
        sys.exit(1)

    points = sampler.detect_roi_points(frames[0], rois, max_points=MAX_POINTS)
    if points.shape[0] == 0:
        print("No points detected within ROI. Exiting.")
        sys.exit(1)
    print(f"Detected {points.shape[0]} corner points in ROI.")

    # 5. Initialize Engine & Track
    tracks,visibilities = engine.track_video_chunked(
        frames, points, chunk_len=CHUNK_LEN, overlap=OVERLAP
    )
    print(f"Tracking complete. Shape: {tracks.shape}")

    # 6. Analyze & Visualize
    npz_path, csv_path = analyzer.save_tracks(base_name, tracks, fps)
    print(f"Saved tracks to {npz_path}")

    vis_path = os.path.join(out_dir, f"tracked_surface_{base_name}.mp4")
    visualize_tracks(frames, tracks, visibilities,vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")

    wave_path = analyzer.compute_and_save_waveform(base_name, tracks, fps)
    print(f"Saved Average Waveform to {wave_path}")


if __name__ == "__main__":
    main()
