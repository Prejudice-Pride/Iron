import os
import sys
import cv2
import numpy as np
import argparse
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_distortion_params(video_path):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    candidates = [
        f"output/calibration_debug/{base_name}_distortion_params.json",
        "output/calibration_debug/distortion_params.json"
    ]
    for p in candidates:
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    return None


def undistort_points_manual(pts, params, w, h):
    if params is None:
        return np.array(pts, dtype=np.float32)

    cx, cy = params.get('cx', w / 2.0), params.get('cy', h / 2.0)
    k3, k5 = params.get('k3', 0.0), params.get('k5', 0.0)
    p1, p2 = params.get('p1', 0.0), params.get('p2', 0.0)

    pts_np = np.array(pts, dtype=np.float32).reshape(-1, 2)
    corrected = pts_np.copy()
    for i, (x, y) in enumerate(pts_np):
        dx, dy = x - cx, y - cy
        r2 = dx * dx + dy * dy
        radial = 1 + k3 * r2 + k5 * r2 * r2
        tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
        ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy
        corrected[i, 0] = cx + dx * radial + tx
        corrected[i, 1] = cy + dy * radial + ty
    return corrected


def undistort_image_manual(img, params):
    if params is None:
        return img

    h, w = img.shape[:2]
    cx, cy = params.get('cx', w / 2.0), params.get('cy', h / 2.0)
    k3, k5 = params.get('k3', 0.0), params.get('k5', 0.0)
    p1, p2 = params.get('p1', 0.0), params.get('p2', 0.0)

    xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dx, dy = xv - cx, yv - cy
    r2 = dx * dx + dy * dy
    radial = 1 + k3 * r2 + k5 * r2 * r2
    tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
    ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy

    map_x = (cx + dx * radial + tx).astype(np.float32)
    map_y = (cy + dy * radial + ty).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

from core.analyzer import Analyzer
from utils.video_io import read_frames

def main():
    parser = argparse.ArgumentParser(description="Homography Calibration Debugger")
    parser.add_argument("video", nargs="?", default="input_videos/pample2.mp4", help="Video path")
    args = parser.parse_args()

    # 1. 加载第一帧
    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, fps = read_frames(args.video)
    img = frames[0].copy()
    base_name = os.path.splitext(os.path.basename(args.video))[0]

    distortion_params = load_distortion_params(args.video)
    if distortion_params is not None:
        print(f"[✓] Loaded distortion params for {base_name}. Applying image correction before calibration.")
        img = undistort_image_manual(img, distortion_params)
    else:
        print(f"[!] No distortion params found for {base_name}. Proceeding without image correction.")
    
    # -------------------------------------------------------------
    # [关键配置] 请在这里输入你实际测量的像素坐标和物理坐标 (cm)
    # -------------------------------------------------------------
    
    # sample.mp4
    src_pts = [
            [297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620], [610, 719]
        ]
        
        # Destination Physical Coordinates (cm)
    dst_pts = [
            [0, 0], [5, 0], [5, 60], [135, 60], [135, 150], [130, 150], [135, 210], [0,60]
        ]
    
    # qample.mp4
    src_pts = [[308, 1002], [325, 1003], [729, 960], [742, 960], [567, 753], [579, 754], [829, 743], [838, 743], [655, 668], [665, 668], [864, 662], [876, 662]]
    dst_pts = [[0,0],[5,0],[65,15],[70,15],[0,90],[5,90],[65,105],[75,105],[0,150],[5,150],[65,165],[70,165]]
    
    # qample2.mp4
    src_pts = [[1120, 950], [1132, 948], [989, 808], [1000, 808], [1561, 756], [1568, 755], [874, 687], [880, 687], [835, 642], [840, 642],[1322, 666], [1327, 665], [1216, 628], [1222, 626]]
    dst_pts = [[0, 0], [5, 0], [0, 60], [5, 60], [190, 60], [195, 60], [0, 150], [5, 150], [0, 210], [5, 210],[190, 150],[195, 150],[190, 210],[195, 210]]

    # # pample2.mp4
    src_pts = [[320, 846], [364, 849], [693, 878], [740, 881], [315, 780], [333, 768], [582, 606], [594, 597]]
    dst_pts = [[60,0],[90,0],[330,0],[360,0],[0,70],[0,100],[0,580],[0,610]]
    
    # # sample3,mp4
    # src_pts = [[438, 751], [809, 665], [1036, 611], [1218, 568], [1313, 729], [876, 908]]
    # dst_pts = [[0,0],[250,0],[500,0],[750,0],[375,100],[0,100]]
    
    print(f"\n[Step 1] Initializing Calibration for: {base_name}")
    analyzer = Analyzer(output_dir="output/calibration_debug")
    
    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)

    if distortion_params is not None:
        src_np = undistort_points_manual(src_np, distortion_params, img.shape[1], img.shape[0])

    # 2. 计算 Homography
    # 用 RANSAC 自动剔除错误点对，阈值设为 5 像素
    H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
    
    if H is None:
        print("Error: Could not calculate Homography matrix. Points might be collinear!")
        return

    inliers = np.sum(mask)
    print(f"[Results] Status: {'Success' if inliers >= 4 else 'FAILED (Insufficient Inliers)'}")
    print(f"[Results] Inliers: {inliers} / {len(src_pts)}")
    
    # 3. 数学评估：重投影误差 (Reprojection Error)
    src_reshaped = src_np.reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(src_reshaped, H).reshape(-1, 2)
    errors = np.sqrt(np.sum((projected - dst_np)**2, axis=1))
    inlier_errors = errors[mask.ravel() == 1]
    mean_err = np.mean(inlier_errors) if len(inlier_errors) > 0 else 0
    
    print(f"[Metrics] Mean Inlier Reprojection Error: {mean_err:.4f} mm")

    # 4. 视觉评估：鸟瞰图变换 (Birds-Eye View)
    # 验证逻辑：如果矩阵正确，原本透视倾斜的面在投影后应当符合物理几何形状
    print("\n[Step 2] Generating Bird's Eye View for verification...")
    min_x, min_y = np.min(dst_np, axis=0)
    max_x, max_y = np.max(dst_np, axis=0)
    
    # 构建可视化画布 (800x800) 的缩放矩阵
    target_w, target_h = 1980, 1080
    scale_x = target_w / (max_x - min_x + 1000)
    scale_y = target_h / (max_y - min_y + 1000)
    S = np.array([
        [scale_x, 0, -min_x * scale_x + 50],
        [0, scale_y, -min_y * scale_y + 50],
        [0, 0, 1]
    ])
    
    # 复合变换矩阵：像素(Image) -> 物理(Physical) -> 画布(Canvas)
    M = S @ H
    warped = cv2.warpPerspective(img, M, (target_w, target_h))
    
    # -------------------------------------------------------------
    # [新增] 在鸟瞰图中绘制坐标系和标定点
    # -------------------------------------------------------------
    # 将物理坐标 dst_pts 映射到画布坐标
    dst_np_reshaped = dst_np.reshape(-1, 1, 2)
    canvas_pts = cv2.perspectiveTransform(dst_np_reshaped, S).reshape(-1, 2)

    # 定义颜色
    axis_color = (255, 255, 255) # 白色坐标轴
    point_color = (0, 255, 255)  # 青色标定点
    
    # 1. 绘制物理坐标轴 (X轴和Y轴)
    # 起点是 (0,0) 的物理坐标在画布上的位置
    origin_phys = np.array([[[0, 0]]], dtype=np.float32)
    origin_canvas = cv2.perspectiveTransform(origin_phys, S).reshape(2).astype(int)
    
    # 绘制 X 轴 (横向)
    cv2.arrowedLine(warped, tuple(origin_canvas), (origin_canvas[0] + 100, origin_canvas[1]), axis_color, 2, tipLength=0.3)
    cv2.putText(warped, "X (cm)", (origin_canvas[0] + 110, origin_canvas[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1)
    
    # 绘制 Y 轴 (纵向)
    cv2.arrowedLine(warped, tuple(origin_canvas), (origin_canvas[0], origin_canvas[1] + 100), axis_color, 2, tipLength=0.3)
    cv2.putText(warped, "Y (cm)", (origin_canvas[0] - 20, origin_canvas[1] + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1)

    # 2. 绘制每一个标定点及其物理坐标标签
    for i, (c_pt, p_pt) in enumerate(zip(canvas_pts, dst_pts)):
        cx, cy = int(c_pt[0]), int(c_pt[1])
        # 画个小十字准星
        cv2.drawMarker(warped, (cx, cy), point_color, cv2.MARKER_CROSS, 15, 2)
        # 写上物理坐标 (x, y)
        label = f"#{i}:({p_pt[0]},{p_pt[1]})"
        cv2.putText(warped, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, point_color, 1)

    # 5. 保存调试图像
    debug_img_path = os.path.join(analyzer.output_dir, f"{base_name}_debug_points.jpg")
    bev_path = os.path.join(analyzer.output_dir, f"{base_name}_birds_eye_view.jpg")
    
    # 在原图上画点：绿色为算法采纳的内点，红色为被剔除的离群点
    for i, (p_src, m) in enumerate(zip(src_pts, mask)):
        color = (0, 255, 0) if m else (0, 0, 255) 
        cv2.circle(img, (int(p_src[0]), int(p_src[1])), 8, color, -1)
        cv2.putText(img, str(i), (int(p_src[0])+10, int(p_src[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imwrite(debug_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(bev_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    
    print("-" * 50)
    print(f"Debug image: {debug_img_path} (绿点: OK, 红点: 误差过大)")
    print(f"Bird's Eye View: {bev_path} (检查在该图中物体是否不再倾斜且比例正常)")
    print("-" * 50)

if __name__ == "__main__":
    main()
