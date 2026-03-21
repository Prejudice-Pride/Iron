"""
Seismic Points Tracking 模块化架构
====================================

该脚本模仿 main_surface.py 的风格与代码结构，用于追踪由外部 JSON 文件指定的特定坐标点。
它跳过了传统的 ROI 选择和自动特征点检测，直接使用用户定义的初始点位。

模块功能简介：
---------------
1. core.engine (追踪引擎):
   - 负责核心运动追踪逻辑。使用 co-tracker 模型进行时间序列上的运动连续性追踪。

2. core.analyzer (数据分析):
   - 负责对生成的轨迹数据进行后处理。提取震动特征，并计算物理位移、波形和频谱。

3. corner_points (点位输入):
   - 从指定的 JSON 文件（如 corner_points\\sample_corners.json）加载初始像素坐标。

使用方法：
-----------
python main_points.py [video_file] [output_dir] --points corner_points\\sample_corners.json
"""

import os
import sys
import argparse
import json
import numpy as np

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

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample(protected).mp4"
OUTPUT_DIR = "output"
DEFAULT_POINTS = os.path.join("corner_points", "sample_corners.json")

# Configuration
CHUNK_LEN = 100
OVERLAP = 10


def main():
    parser = argparse.ArgumentParser(description="Seismic Points Tracker")
    parser.add_argument(
        "video", nargs="?", default=None, help="Input video filename or path"
    )
    parser.add_argument(
        "output", nargs="?", default=OUTPUT_DIR, help="Output directory"
    )
    parser.add_argument(
        "--points", default=DEFAULT_POINTS, help="Path to JSON file containing points"
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
    out_dir = os.path.join(args.output, f"{base_name}_output_points")
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

    # 4. Point Loading (Specified points instead of ROI detection)
    if not os.path.exists(args.points):
        print(f"Error: Points JSON file not found at {args.points}")
        sys.exit(1)
    
    try:
        with open(args.points, 'r') as f:
            point_list = json.load(f)
        points = np.array(point_list, dtype=np.float32)
        print(f"Loaded {points.shape[0]} points from {args.points}.")
    except Exception as e:
        print(f"Error loading points: {e}")
        sys.exit(1)

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
    else:
        print("\n[INFO] No specific calibration found for this video. Using linear fallback (10mm/pixel).")

    # 5. Initialize Engine & Track
    tracks, visibilities = engine.track_video_chunked(frames, points, chunk_len=CHUNK_LEN, overlap=OVERLAP)
    print(f"Tracking complete. Shape: {tracks.shape}")

    # 6. Analyze & Visualize
    npz_path, csv_path = analyzer.save_tracks(base_name, tracks, fps)
    print(f"Saved tracks to {npz_path}")

    # Try to load existing ROIs just for visualization context
    rois, _ = sampler.load_rois(base_name)
    vis_path = os.path.join(out_dir, f"tracked_points_{base_name}.mp4")
    visualize_tracks(frames, tracks, visibilities, vis_path, fps, rois=rois)
    print(f"Saved visualization to {vis_path}")

    # Additional Analysis (Waveforms & Spectrum)
    wave_path = analyzer.compute_and_save_waveform(base_name, tracks, fps)
    print(f"Saved Average Waveform to {wave_path}")

    double_wave_path = analyzer.compute_and_save_double_waveforms(base_name, tracks, fps)
    print(f"Saved Double Waveform Plot to {double_wave_path}")

    spec_path = analyzer.compute_and_save_spectrum(base_name, tracks, fps)
    print(f"Saved Frequency Spectrum to {spec_path}")

    analyzer.calculate_intensity(tracks, fps, pixels_to_mm=10.0) 


if __name__ == "__main__":
    main()
