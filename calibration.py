import os
import sys
import cv2
import numpy as np
import argparse

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.analyzer import Analyzer
from utils.video_io import read_frames

def main():
    parser = argparse.ArgumentParser(description="Homography Calibration Debugger")
    parser.add_argument("video", nargs="?", default="../input_videos/qample2.mp4", help="Video path")
    args = parser.parse_args()

    # 1. 加载第一帧
    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, fps = read_frames(args.video)
    img = frames[0].copy()
    base_name = os.path.splitext(os.path.basename(args.video))[0]
    
    # -------------------------------------------------------------
    # [关键配置] 请在这里输入你实际测量的像素坐标和物理坐标 (mm)
    # -------------------------------------------------------------
    # src_pts: 视频画面中的像素 [x, y]
    # [[297, 860], [308, 864], [620, 720], [1157, 921], [1319, 703], [1308, 702], [1377, 620], [1433, 542], [888, 596]]
    
    # 2026/3/21 新增:
    # src_pts = [
    # [1002, 544],
    # [1375, 620],
    # [1155, 925],
    # [608, 717],]
    
    # src_pts = [
    # [1002, 544],
    # [1375, 620],
    # [1155, 925],
    # [608, 717],]
    
    # dst_pts: 对应的物理空间坐标 [x_mm, y_mm]
    # [[0, 0], [15, 0], [15, 60], [880, 60], [880, 150], [865, 150], [880, 240], [0, 1500], [8800, 3300]]
    
    # 2026/3/21 新增：
    # dst_pts = [
    # [0, 0],
    # [880, 0],
    # [0, 150],
    # [880, 150],]
    
    # dst_pts = [
    # [0, 0],
    # [880, 0],
    # [880, 150],
    # [0, 150],]
    
    
    src_pts = [[1123, 953], [1133, 951], [988, 810], [998, 809], [1768, 833], [1773, 830], [1569, 759], [1574, 757], [876, 687], [880, 686], [833, 642], [840, 642], [1324, 667], [1327, 665], [1220, 627], [1224, 626]]
    dst_pts = [[0,0],[15,0],[0,60],[15,60],[865,0],[880,0],[865,60],[880,60],[0,150],[15,150],[0,210],[15,210],[865,150],[880,150],[865,210],[880,210]]

    # src_pts = [
    #     [835, 644],
    #     [1227, 628],
    #     [1794, 837],
    #     [1123, 955],
    # ]
    # dst_pts=[[0,0],[880,0],[0,210],[880,210]]
    print(f"\n[Step 1] Initializing Calibration for: {base_name}")
    analyzer = Analyzer(output_dir="output/calibration_debug")
    
    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)

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
    cv2.putText(warped, "X (mm)", (origin_canvas[0] + 110, origin_canvas[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1)
    
    # 绘制 Y 轴 (纵向)
    cv2.arrowedLine(warped, tuple(origin_canvas), (origin_canvas[0], origin_canvas[1] + 100), axis_color, 2, tipLength=0.3)
    cv2.putText(warped, "Y (mm)", (origin_canvas[0] - 20, origin_canvas[1] + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, axis_color, 1)

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
