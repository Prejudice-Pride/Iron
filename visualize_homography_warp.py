import cv2
import numpy as np
import os
import argparse

def load_calibration_by_video_name(video_path):
    """根据视频文件名自动加载对应的标定数据"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    calibrations = {
        "qample2": {
            "src_pts": [[1123, 953], [1133, 951], [988, 810], [998, 809], 
                        [1768, 833], [1773, 830], [1569, 759], [1574, 757], 
                        [876, 687], [880, 686], [833, 642], [840, 642], 
                        [1324, 667], [1327, 665], [1220, 627], [1224, 626]],
            "dst_pts_mm": [[0,0],[15,0],[0,60],[15,60],
                          [865,0],[880,0],[865,60],[880,60],
                          [0,150],[15,150],[0,210],[15,210],
                          [865,150],[880,150],[865,210],[880,210]]
        },
        "sample": {
            "src_pts": [[297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]],
            "dst_pts_mm": [[0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]]
        },
        "sample2": {
            "src_pts": [[478, 482], [656, 478], [597, 436], [688, 393]],
            "dst_pts_mm": [[0, 0], [220, 600], [110, 270], [220, 600]]
        },
        "sample4": {
            "src_pts": [[402, 430], [1045, 398], [1300, 594], [500, 623]],
            "dst_pts_mm": [[0, 0], [415, 0], [415, 200], [0, 200]]
        }
    }
    
    if base_name in calibrations:
        print(f"[✓] 自动识别到 {base_name}，加载对应的标定数据")
        cal = calibrations[base_name]
        return np.array(cal["src_pts"], dtype=np.float32), np.array(cal["dst_pts_mm"], dtype=np.float32)
    else:
        print(f"[⚠] 视频 {base_name} 未找到标定数据，使用 qample2 的默认值")
        cal = calibrations["qample2"]
        return np.array(cal["src_pts"], dtype=np.float32), np.array(cal["dst_pts_mm"], dtype=np.float32)

def plot_grid(img, px_per_cm, max_x_cm, max_y_cm):
    """在变换后的图像上绘制物理网格和坐标刻度 (单位: cm)"""
    h, w = img.shape[:2]
    # 绘制背景网格
    for x_cm in range(0, max_x_cm, 10): # 每 10cm 一条竖线
        x_px = int(x_cm * px_per_cm)
        cv2.line(img, (x_px, 0), (x_px, h), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{x_cm}cm", (x_px + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    for y_cm in range(0, max_y_cm, 10): # 每 10cm 一条横线
        y_px = int(y_cm * px_per_cm)
        cv2.line(img, (0, y_px), (w, y_px), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, f"{y_cm}cm", (5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # 绘制主坐标轴 X / Y
    cv2.line(img, (0, 0), (w, 0), (0, 0, 255), 3, cv2.LINE_AA) # X 轴 (红色)
    cv2.line(img, (0, 0), (0, h), (0, 255, 0), 3, cv2.LINE_AA) # Y 轴 (绿色)
    
    return img

def infer_track_units(tracks):
    """根据轨迹数值范围粗略判断轨迹单位是否已经是物理坐标(cm)。"""
    finite = tracks[np.isfinite(tracks)]
    if finite.size == 0:
        return "unknown"
    max_abs = float(np.nanmax(np.abs(finite)))
    # 像素坐标通常在几百到上千；cm 坐标通常远小于这个量级
    return "physical_cm" if max_abs < 200.0 else "pixel"

def main():
    parser = argparse.ArgumentParser(description="Warp Video and Tracks Map to Physical Space")
    parser.add_argument("--video", nargs="?", default="input_videos\sample.mp4", help="Input video")
    parser.add_argument("--tracks", nargs="?", default="output\sample_output_cancel\sample_raw_surface_tracks.npz", help="Track npz file")
    parser.add_argument("--output", nargs="?", default="output/birdseye_warp.avi", help="Output video path")
    args = parser.parse_args()

    # 1. 根据视频名自动加载标定数据
    src_pts, dst_pts_mm = load_calibration_by_video_name(args.video)

    dst_pts_cm = dst_pts_mm / 10.0

    # 2. 设定可视化输出的分辨率尺
    # 物理宽度最大约 90cm, 高度最大约 30cm
    # 我们设 1cm = 15 pixels，这样输出视频的宽度约 1350，高度约 450，适合观看
    PX_PER_CM = 15.0
    MAX_X_CM = 100
    MAX_Y_CM = 40
    out_w, out_h = int(MAX_X_CM * PX_PER_CM), int(MAX_Y_CM * PX_PER_CM)

    # 可视化的目标拉平像素坐标
    vis_dst_pts = dst_pts_cm * PX_PER_CM

    # 计算 像素到可视化图像像素 的单应性矩阵
    H_vis, _ = cv2.findHomography(src_pts, vis_dst_pts, cv2.RANSAC, 5.0)

    # 计算 像素到真实物理cm 的单应性矩阵 (用于点坐标换算验证)
    H_phys, _ = cv2.findHomography(src_pts, dst_pts_cm, cv2.RANSAC, 5.0)

    # 3. 加载轨迹数据
    if not os.path.exists(args.tracks):
        print(f"[Error] Tracks file not found: {args.tracks}")
        print("请确保先运行 main_cancel.py 输出 raw tracks。")
        return
        
    data = np.load(args.tracks)
    raw_tracks = data['tracks'] # (N, T, 2)
    # 修复：从 np.load 中加载的可能是 scalar array，直接强转为 float
    fps = float(data['fps']) 
    N_pts, T, _ = raw_tracks.shape
    track_units = infer_track_units(raw_tracks)
    print(f"[✓] 轨迹单位识别为: {track_units}")

    # 4. 加载原视频
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"[Error] Video not found: {args.video}")
        return

    # 创建输出目录
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    
    # 尝试使用多种编码器，确保兼容性
    if args.output.lower().endswith('.avi'):
        fourcc_options = [cv2.VideoWriter_fourcc(*'MJPG'),
                          cv2.VideoWriter_fourcc(*'XVID'),
                          cv2.VideoWriter_fourcc(*'mp4v')]
    else:
        fourcc_options = [cv2.VideoWriter_fourcc(*'mp4v'), 
                          cv2.VideoWriter_fourcc(*'H264'),
                          cv2.VideoWriter_fourcc(*'MJPG')]
    
    out = None
    for fourcc in fourcc_options:
        try:
            out = cv2.VideoWriter(args.output, fourcc, fps, (out_w, out_h))
            if out.isOpened():
                print(f"[✓] 视频编码器初始化成功")
                break
        except:
            continue
    
    if out is None or not out.isOpened():
        print(f"[Error] 无法初始化视频写入器")
        return
    
    print(f"Generating physical bird's-eye view video... Expected frames: {T}")
    
    # 颜色表，区分不同的点
    colors = np.random.randint(0, 255, (N_pts, 3), dtype=int)
    
    first_frame_done = False
    
    t_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or t_idx >= T:
            break
            
        # a. 图像拉平变换 (Warp to Physical Birds-eye View)
        warped_frame = cv2.warpPerspective(frame, H_vis, (out_w, out_h))
        
        # b. 绘制基础物理网格
        warped_frame = plot_grid(warped_frame, PX_PER_CM, MAX_X_CM, MAX_Y_CM)
        
        # c. 将对应的原始像素点，映射并绘制到当前画面中
        points_drawn = 0
        for i in range(N_pts):
            pt_px = raw_tracks[i, t_idx]
            if np.isnan(pt_px[0]) or np.isnan(pt_px[1]):
                continue

            if track_units == "physical_cm":
                # 轨迹已经是物理坐标(cm)，直接缩放到可视化画布
                mapped_phys = pt_px.astype(np.float32)
                x_vis = int(round(mapped_phys[0] * PX_PER_CM))
                y_vis = int(round(mapped_phys[1] * PX_PER_CM))
            else:
                # 原始像素轨迹：先走单应性，再得到物理坐标
                vec = np.array([pt_px[0], pt_px[1], 1.0])
                mapped_vis = H_vis.dot(vec)
                mapped_vis = mapped_vis[:2] / mapped_vis[2]
                x_vis, y_vis = int(mapped_vis[0]), int(mapped_vis[1])
                mapped_phys = H_phys.dot(vec)
                mapped_phys = mapped_phys[:2] / mapped_phys[2]
            
            # 边界检查，确保点在画布内
            if 0 <= x_vis < out_w and 0 <= y_vis < out_h:
                color = tuple(int(c) for c in colors[i])
                cv2.circle(warped_frame, (x_vis, y_vis), 4, color, -1)
                points_drawn += 1
            
            # 第一帧诊断信息
            if t_idx == 0 and not first_frame_done:
                status = "✓" if (0 <= x_vis < out_w and 0 <= y_vis < out_h) else "✗ 超出范围"
                print(f"  点 {i}: 输入 ({pt_px[0]:.1f}, {pt_px[1]:.1f}) -> 画布 ({x_vis}, {y_vis}) -> 物理 ({mapped_phys[0]:.2f}cm, {mapped_phys[1]:.2f}cm) [{status}]")
        
        if t_idx == 0:
            print(f"第一帧绘制的点数: {points_drawn}/{N_pts}")
            first_frame_done = True

        out.write(warped_frame)
        t_idx += 1
        print(f"\rProcessed frame {t_idx}/{T}", end="")

    cap.release()
    out.release()
    print(f"\nDone! Bird's-eye physical video saved to {args.output}")

if __name__ == "__main__":
    main()
