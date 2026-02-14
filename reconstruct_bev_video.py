"""
Birds-Eye View Track Visualizer
====================================

该脚本将原始追踪数据（包含在 .npz 文件中）映射到鸟瞰图（BEV）上，并生成动态的追踪视频。
它利用 calibration.py 中定义的同态矩阵 (Homography) 以及缩放矩阵，将原始像素坐标转换到 BEV 的画布坐标上。

输入要求：
-----------
1. 鸟瞰图背景：output\calibration_debug\sample_birds_eye_view.jpg
2. 追踪数据：output\sample_output_surface\sample_surface_tracks.npz

使用方法：
-----------
python reconstruct_bev_video.py
"""

import os
import cv2
import numpy as np
import imageio

# 路径配置
BEV_IMAGE_PATH = r"output\calibration_debug\sample_birds_eye_view.jpg"
TRACKS_NPZ_PATH = r"output\sample_output_points\sample_surface_tracks.npz"
OUTPUT_VIDEO_PATH = r"output\calibration_debug\sample_bev_reconstruction.mp4"

# 标定参数 (必须与 calibration.py 保持一致以保证坐标对齐)
SRC_PTS = [
    [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]
]
DST_PTS = [
    [0, 0], [150, 0], [150, 600], [8800, 600], [8800, 1500], [8650, 1500], [8800, 2400]
]
TARGET_W, TARGET_H = 800, 800


def get_bev_transform():
    """
    计算从原始像素坐标到 BEV 画布像素坐标的复合变换矩阵 M。
    M = S @ H
    """
    src_np = np.array(SRC_PTS, dtype=np.float32)
    dst_np = np.array(DST_PTS, dtype=np.float32)

    # 1. 计算原始图像到物理空间的 Homography (H)
    H, _ = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
    
    # 2. 计算物理空间到画布的缩放矩阵 (S)
    min_x, min_y = np.min(dst_np, axis=0)
    max_x, max_y = np.max(dst_np, axis=0)
    
    scale_x = TARGET_W / (max_x - min_x + 1000)
    scale_y = TARGET_H / (max_y - min_y + 1000)
    
    S = np.array([
        [scale_x, 0, -min_x * scale_x + 50],
        [0, scale_y, -min_y * scale_y + 50],
        [0, 0, 1]
    ])
    
    return S @ H


def main():
    # 1. 加载数据
    if not os.path.exists(BEV_IMAGE_PATH):
        print(f"Error: BEV image not found at {BEV_IMAGE_PATH}")
        return
    if not os.path.exists(TRACKS_NPZ_PATH):
        print(f"Error: Tracks data not found at {TRACKS_NPZ_PATH}")
        return

    bev_bg = cv2.imread(BEV_IMAGE_PATH)
    data = np.load(TRACKS_NPZ_PATH)
    tracks = data['tracks']  # (N, T, 2)
    fps = data.get('fps', 30.0)
    
    N, T, _ = tracks.shape
    print(f"Loaded tracks: {N} points, {T} frames, FPS: {fps}")

    # 2. 准备变换矩阵
    M = get_bev_transform()

    # 3. 变换所有轨迹坐标
    # tracks 为像素坐标 (x, y)，形式为 (N, T, 2)
    tracks_flat = tracks.reshape(-1, 1, 2).astype(np.float32)
    # 映射到 BEV 画布坐标
    tracks_bev_flat = cv2.perspectiveTransform(tracks_flat, M)
    tracks_bev = tracks_bev_flat.reshape(N, T, 2)

    # 4. 视频渲染
    print(f"Rendering BEV tracking video to: {OUTPUT_VIDEO_PATH}")
    
    # 随机颜色生成
    rng = np.random.RandomState(42)
    colors = [tuple(int(c) for c in (rng.randint(50, 255), rng.randint(50, 255), rng.randint(50, 255))) for _ in range(N)]

    writer = imageio.get_writer(OUTPUT_VIDEO_PATH, fps=float(fps), macro_block_size=None)

    for t in range(T):
        # 复制背景图
        frame_img = bev_bg.copy()
        
        for i in range(N):
            # 获取当前帧的点位
            curr_pt = tracks_bev[i, t]
            if np.isnan(curr_pt[0]) or np.isnan(curr_pt[1]):
                continue
            
            px, py = int(curr_pt[0]), int(curr_pt[1])
            color = colors[i]

            # 绘制轨迹点
            cv2.circle(frame_img, (px, py), 4, color, -1)
            
            # 绘制尾迹 (最近30帧)
            trail_start = max(0, t - 30)
            for tt in range(trail_start, t):
                p1 = tracks_bev[i, tt]
                p2 = tracks_bev[i, tt+1]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    cv2.line(frame_img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, 2)

        # 转换为 RGB 格式写入视频文件
        writer.append_data(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
        
        if t % 50 == 0:
            print(f"  Processed frame {t}/{T}")

    writer.close()
    print("Reconstruction complete!")


if __name__ == "__main__":
    main()
