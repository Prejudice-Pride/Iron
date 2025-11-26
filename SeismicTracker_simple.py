"""
使用 cotracker3_offline 对视频角点进行追踪并绘制轨迹（完全重写版）

修复说明：
1. 使用 CoTrackerPredictor 官方封装，自动处理 resize 和坐标变换。
2. 正确加载 checkpoint 权重（解决角点悬浮/随机乱飞的问题）。
3. 移除手动 resize 和分块逻辑，直接输入全量视频。
"""

import os
import sys
import numpy as np
import cv2
import torch
import imageio.v3 as iio

# 添加 co-tracker 路径到 sys.path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cotracker.predictor import CoTrackerPredictor

# 配置
CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "scaled_offline.pth")


# -------------------------
#  读取视频帧
# -------------------------
def read_frames(path):
    # 读取所有帧，返回 (T, H, W, 3) RGB
    frames = iio.imread(path, plugin="FFMPEG")
    # imageio 读取的是 RGB，OpenCV 处理通常用 BGR，但 CoTracker 需要 RGB
    # 这里我们保持 RGB 用于模型输入，可视化时转 BGR
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.release()
    return frames, fps


# -------------------------
#  角点检测（第一帧）
# -------------------------
def detect_corners_firstframe(first_frame_rgb, max_corners=200):
    gray = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2GRAY)
    p = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners,
                                qualityLevel=0.01, minDistance=7, blockSize=7)
    if p is None:
        return np.empty((0, 2), dtype=np.float32)
    return p.reshape(-1, 2).astype(np.float32)


# -------------------------
#  绘制最终轨迹
# -------------------------
def visualize_tracks(frames_rgb, tracks, out_path, fps):
    """
    frames_rgb: (T, H, W, 3)
    tracks: (N, T, 2)
    """
    N, T, _ = tracks.shape
    rng = np.random.RandomState(123)
    colors = [tuple(rng.randint(30, 255, size=3).tolist()) for _ in range(N)]

    vis = []
    for t in range(T):
        # 转为 BGR 用于 OpenCV 绘图
        img = cv2.cvtColor(frames_rgb[t], cv2.COLOR_RGB2BGR)
        
        for i in range(N):
            pts = tracks[i]
            # 绘制轨迹线
            valid = ~np.isnan(pts[:t+1, 0])
            if valid.sum() >= 2:
                vps = pts[:t+1][valid].astype(np.int32)
                for k in range(len(vps) - 1):
                    cv2.line(img, tuple(vps[k]), tuple(vps[k + 1]), colors[i], 2)
            
            # 绘制当前点
            if not np.isnan(pts[t, 0]):
                x, y = int(pts[t, 0]), int(pts[t, 1])
                cv2.circle(img, (x, y), 4, colors[i], -1)

        vis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # 转回 RGB 保存

    iio.imwrite(out_path, np.array(vis), plugin="FFMPEG", fps=fps)
    print(f"追踪结果已保存至: {out_path}")


# -------------------------
#  主程序
# -------------------------
def main():
    INPUT_DIR = "input_videos"
    DEFAULT_VIDEO = "rolling.mp4"
    OUTPUT_DIR = "output_videos"
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    inp = os.path.join(INPUT_DIR, DEFAULT_VIDEO)
    out = os.path.join(OUTPUT_DIR, "tracked_fixed_cotracker3.mp4")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载模型 (使用 Predictor 自动处理加载和 resize)
    if not os.path.exists(CKPT_PATH):
        print(f"Error: Checkpoint not found at {CKPT_PATH}")
        return

    print(f"Loading model from {CKPT_PATH}...")
    model = CoTrackerPredictor(checkpoint=CKPT_PATH).to(device)

    # 2. 读取视频
    print(f"Reading video: {inp}")
    frames, fps = read_frames(inp)
    T, H, W, _ = frames.shape
    print(f"Video shape: {frames.shape}, FPS: {fps}")

    # 3. 检测角点
    pts = detect_corners_firstframe(frames[0])
    if len(pts) == 0:
        print("No corners detected.")
        return
    
    # 仅使用前 20 个点演示，避免太乱
    pts = pts[:20]
    print(f"Tracking {len(pts)} points.")

    # 4. 构造 Queries
    # CoTrackerPredictor 接受 queries 格式: (B, N, 3) -> (t, x, y)
    # 坐标为原始分辨率下的像素坐标
    queries = np.zeros((1, len(pts), 3), dtype=np.float32)
    queries[0, :, 0] = 0        # 从第 0 帧开始
    queries[0, :, 1:] = pts     # (x, y)
    queries = torch.from_numpy(queries).to(device)

    # 5. 准备视频输入
    # CoTrackerPredictor 接受 (B, T, 3, H, W) float [0, 255]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(device)

    # 6. 执行追踪
    print("Running tracking...")
    # CoTrackerPredictor 会自动 resize video 和 queries 到模型所需分辨率，
    # 并在输出时将 tracks 还原回原始分辨率。
    pred_tracks, pred_visibility = model(video, queries=queries)
    
    # pred_tracks: (B, T, N, 2)
    tracks = pred_tracks[0].cpu().numpy() # (T, N, 2)
    
    # 转换 shape 为 (N, T, 2) 以便可视化
    tracks = tracks.transpose(1, 0, 2)
    print(f"Output tracks shape: {tracks.shape}")

    # 7. 可视化
    visualize_tracks(frames, tracks, out, fps)


if __name__ == "__main__":
    main()
