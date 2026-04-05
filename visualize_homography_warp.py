import cv2
import numpy as np
import os
import argparse
import json

def load_calibration_params(video_path):
    """加载视频对应的畸变参数"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 常用路径
    param_path = f"output/calibration_debug/{base_name}_distortion_params.json"
    
    if os.path.exists(param_path):
        with open(param_path, 'r') as f:
            params = json.load(f)
        print(f"[✓] 加载畸变参数: {param_path}")
        return params
    else:
        print(f"[⚠] 未找到畸变参数文件: {param_path}，将跳过畸变矫正")
        return None

def undistort_points(pts, params, w, h):
    """手动应用径向和切向畸变矫正 (针对归一化坐标或像素坐标)"""
    if params is None:
        return pts
    
    # 这里的参数通常是基于图像中心的
    cx, cy = params.get('cx', w/2), params.get('cy', h/2)
    k3 = params.get('k3', 0)
    k5 = params.get('k5', 0)
    p1 = params.get('p1', 0)
    p2 = params.get('p2', 0)

    pts_undistorted = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        dx = x - cx
        dy = y - cy
        r2 = dx*dx + dy*dy
        
        # 径向畸变
        radial = 1 + k3 * r2 + k5 * r2 * r2
        
        # 切向畸变
        tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
        ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy
        
        pts_undistorted[i, 0] = cx + dx * radial + tx
        pts_undistorted[i, 1] = cy + dy * radial + ty
        
    return pts_undistorted

def undistort_image(img, params):
    """应用畸变矫正到图像"""
    if params is None:
        return img
    
    h, w = img.shape[:2]
    cx, cy = params.get('cx', w/2), params.get('cy', h/2)
    k3 = params.get('k3', 0)
    k5 = params.get('k5', 0)
    p1 = params.get('p1', 0)
    p2 = params.get('p2', 0)
    
    # 构建相机矩阵和畸变系数 (OpenCV 格式)
    # 注意：这里的 k3, k5, p1, p2 是自定义模型参数，具体映射取决于训练时的模型
    # 如果是 standard OpenCV 畸变 [k1, k2, p1, p2, k3]
    # 根据 pample2 示例，它只有几个参数，我们模拟其变换
    
    # 手动变换通常比 map/remap 慢，但在 Python 中我们可以使用 mapx/mapy
    # 或者如果参数较少，直接用 undistort
    dist_coeffs = np.array([0, 0, p1, p2, k3, 0, 0, k5]) # 占位尝试，具体看解析
    # 如果参数定义不同，建议用手动映射
    
    # 这里我们采用与 points 相同的逻辑进行快速映射（如果性能允许）
    # 或者为了简单起见，重新实现一个简单的 remap
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    x_range = np.arange(w)
    y_range = np.arange(h)
    xv, yv = np.meshgrid(x_range, y_range)
    
    dx = xv - cx
    dy = yv - cy
    r2 = dx*dx + dy*dy
    radial = 1 + k3 * r2 + k5 * r2 * r2
    tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
    ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy
    
    map_x = (cx + dx * radial + tx).astype(np.float32)
    map_y = (cy + dy * radial + ty).astype(np.float32)
    
    undistorted_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)
    return undistorted_img

def load_calibration_by_video_name(video_path):
    """根据视频文件名自动加载对应的标定数据"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    calibrations = {
        "pample2":{
            "src_pts" : [[321, 847], [365, 850], [693, 878], [742, 883], [312, 780], [333, 767], [583, 605], [592, 597]],
            "dst_pts_cm" : [[40,0],[85,0],[355,0],[400,0],[0,60],[0,105],[0,585],[0,630]]
        },
        "qample2": {
            "src_pts": [[1123, 953], [1133, 951], [988, 810], [998, 809], 
                        [1768, 833], [1773, 830], [1569, 759], [1574, 757], 
                        [876, 687], [880, 686], [833, 642], [840, 642], 
                        [1324, 667], [1327, 665], [1220, 627], [1224, 626]],
            "dst_pts_cm": [[0,0],[15,0],[0,60],[15,60],
                          [865,0],[880,0],[865,60],[880,60],
                          [0,150],[15,150],[0,210],[15,210],
                          [865,150],[880,150],[865,210],[880,210]]
        },
        "qample":{
            "src_pts": [[308, 1002], [325, 1003], [729, 960], [742, 960], [567, 753], [579, 754], [829, 743], [838, 743], [655, 668], [665, 668], [864, 662], [876, 662]],
            "dst_pts_cm": [[0,0],[15,0],[75,15],[90,15],[0,90],[15,90],[75,105],[90,105],[0,150],[15,150],[75,165],[90,165]]   
        },
        "sample": {
            "src_pts": [[297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620]],
            "dst_pts_cm": [[0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240]]
        },
        "sample2": {
            "src_pts": [[478, 482], [656, 478], [597, 436], [688, 393]],
            "dst_pts_cm": [[0, 0], [220, 600], [110, 270], [220, 600]]
        },
        "sample4": {
            "src_pts": [[402, 430], [1045, 398], [1300, 594], [500, 623]],
            "dst_pts_cm": [[0, 0], [415, 0], [415, 200], [0, 200]]
        }
    }
    
    if base_name in calibrations:
        print(f"[✓] 自动识别到 {base_name}，加载对应的标定数据")
        cal = calibrations[base_name]
        return np.array(cal["src_pts"], dtype=np.float32), np.array(cal["dst_pts_cm"], dtype=np.float32)
    else:
        print(f"[⚠] 视频 {base_name} 未找到标定数据，使用 qample2 的默认值")
        cal = calibrations["qample2"]
        return np.array(cal["src_pts"], dtype=np.float32), np.array(cal["dst_pts_cm"], dtype=np.float32)

def plot_grid(img, px_per_cm, max_x_cm, max_y_cm, offset_x=0.0, offset_y=20.0):
    """在变换后的图像上绘制物理网格和坐标刻度 (单位: cm, 支持偏移)"""
    h, w = img.shape[:2]
    
    # 绘制背景网格 (灰色)
    for x_cm in range(0, max_x_cm, 10): # 每 10cm 一条竖线
        x_px = int(x_cm * px_per_cm)
        real_x = x_cm - offset_x
        cv2.line(img, (x_px, 0), (x_px, h), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(real_x)}", (x_px + 2, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    for y_cm in range(0, max_y_cm, 10): # 每 10cm 一条横线
        y_px = int(y_cm * px_per_cm)
        real_y = y_cm - offset_y
        cv2.line(img, (0, y_px), (w, y_px), (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(img, f"{int(real_y)}", (5, y_px - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    # 绘制物理 (0,0) 主坐标轴
    origin_x_px = int(offset_x * px_per_cm)
    origin_y_px = int(offset_y * px_per_cm)
    
    if 0 <= origin_x_px < w:
        cv2.line(img, (origin_x_px, 0), (origin_x_px, h), (0, 255, 0), 2, cv2.LINE_AA) # Y 轴 (绿色)
    if 0 <= origin_y_px < h:
        cv2.line(img, (0, origin_y_px), (w, origin_y_px), (0, 0, 255), 2, cv2.LINE_AA) # X 轴 (红色)
    
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
    parser.add_argument("--video", nargs="?", default="input_videos\pample2.mp4", help="Input video")
    parser.add_argument("--tracks", nargs="?", default="output\pample2_output_cancel\pample2_raw_surface_tracks.npz", help="Track npz file")
    parser.add_argument("--output", nargs="?", default="output/birdseye_warp.avi", help="Output video path")
    args = parser.parse_args()

    # 1. 根据视频名自动加载标定数据
    src_pts, dst_pts_cm = load_calibration_by_video_name(args.video)
    
    # 1.1 加载畸变模型参数
    distort_params = load_calibration_params(args.video)

    # 2. 设定可视化输出的分辨率尺
    # 物理跨度: X 约 100cm, Y 约 40cm. 
    # 我们增加偏移以容纳负数坐标点
    PX_PER_CM = 15.0
    MAX_X_CM = 120 
    MAX_Y_CM = 80 
    
    # 【偏移设置】物理坐标 (0,0) 在画布上的偏移 cm
    OFFSET_X_CM = 10.0 # 让物理 0 映射到画布 10cm 处
    OFFSET_Y_CM = 30.0 # 让物理 0 映射到画布 30cm 处 (解决 y=-2.9 问题)

    out_w, out_h = int(MAX_X_CM * PX_PER_CM), int(MAX_Y_CM * PX_PER_CM)

    # 可视化的目标拉平像素坐标 (加上物理偏移)
    vis_dst_pts = (dst_pts_cm + np.array([OFFSET_X_CM, OFFSET_Y_CM])) * PX_PER_CM

    # 计算 像素到可视化图像像素 的单应性矩阵 (必须基于矫正后的坐标)
    # 因此我们需要将 src_pts 进行畸变变换 (矫正后的点是输入到 homography 的点)
    
    # 矫正后的 src_pts_distorted (从原始像素空间映射到矫正后的像素空间，单应性基于矫正后点)
    # 或者直接在 H 计算前将 src_pts “反向” 处理
    # 如果我们的 homography 是对矫正后图像做的。
    
    # 这里我们认为 src_pts 是基于原图的点，我们手动把它们转换到矫正后的位置
    # (如果 undistort_image 操作把原图拉伸或弯曲，src_pts 也必须随之移动)
    
    # 为了简化，我们假设 H 映射的是 (原图采样/渲染) -> (平面)，即渲染时直接在每帧做 undistort
    # 为了准确，先处理 src_pts
    
    # 获取视频原始宽高
    v_cap = cv2.VideoCapture(args.video)
    vw = int(v_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh = int(v_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_cap.release()

    src_pts_corr = undistort_points(src_pts, distort_params, vw, vh)

    # 使用矫正后的点计算 H
    H_vis, _ = cv2.findHomography(src_pts_corr, vis_dst_pts, cv2.RANSAC, 5.0)

    # 计算 像素到真实物理cm 的单应性矩阵 (用于点坐标换算验证)
    H_phys, _ = cv2.findHomography(src_pts_corr, dst_pts_cm, cv2.RANSAC, 5.0)

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
            
        # a1. 图像畸变矫正 (Apply Distortion Model)
        if distort_params is not None:
            frame = undistort_image(frame, distort_params)
        
        # a2. 图像拉平变换 (Warp to Physical Birds-eye View)
        warped_frame = cv2.warpPerspective(frame, H_vis, (out_w, out_h))
        
        # b. 绘制基础物理网格
        warped_frame = plot_grid(warped_frame, PX_PER_CM, MAX_X_CM, MAX_Y_CM, OFFSET_X_CM, OFFSET_Y_CM)
        
        # c. 将对应的原始像素点，映射并绘制到当前画面中
        points_drawn = 0
        for i in range(N_pts):
            pt_px = raw_tracks[i, t_idx]
            if np.isnan(pt_px[0]) or np.isnan(pt_px[1]):
                continue

            if track_units == "physical_cm":
                # 轨迹已经是物理坐标(cm)，手动应用偏移后再缩放到可视化画布
                x_vis = int(round((pt_px[0] + OFFSET_X_CM) * PX_PER_CM))
                y_vis = int(round((pt_px[1] + OFFSET_Y_CM) * PX_PER_CM))
                mapped_phys = pt_px.astype(np.float32)
            else:
                # 原始像素轨迹：先走畸变处理，再走单应性 (H_vis 已经包含物理偏移)
                pt_px_arr = np.array([pt_px], dtype=np.float32)
                pt_px_corr = undistort_points(pt_px_arr, distort_params, frame.shape[1], frame.shape[0])[0]

                vec = np.array([pt_px_corr[0], pt_px_corr[1], 1.0])
                mapped_vis = H_vis.dot(vec)
                mapped_vis = mapped_vis[:2] / mapped_vis[2]
                x_vis, y_vis = int(mapped_vis[0]), int(mapped_vis[1])
                # 计算真实物理坐标 (不带画布偏移)
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
