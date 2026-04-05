import os
import sys
import cv2
import numpy as np
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_io import read_frames

# ---------------------------------------------------------
# 畸变矫正相关函数
# ---------------------------------------------------------
def load_distortion_params(video_path):
    """加载视频对应的畸变参数"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    # 尝试在不同级目录寻找
    possible_paths = [
        f"output/calibration_debug/{base_name}_distortion_params.json",
        f"../output/calibration_debug/{base_name}_distortion_params.json"
    ]
    for p in possible_paths:
        if os.path.exists(p):
            with open(p, 'r') as f:
                return json.load(f)
    return None

def undistort_points_manual(pts, params, w, h):
    """手动将像素坐标点转换到矫正后的位置"""
    if params is None:
        return np.array(pts, dtype=np.float32).reshape(-1, 2)
    
    cx, cy = params.get('cx', w/2), params.get('cy', h/2)
    k3, k5 = params.get('k3', 0), params.get('k5', 0)
    p1, p2 = params.get('p1', 0), params.get('p2', 0)

    pts_np = np.array(pts, dtype=np.float32).reshape(-1, 2)
    pts_corr = pts_np.copy()
    
    for i in range(len(pts_np)):
        x, y = pts_np[i]
        dx, dy = x - cx, y - cy
        r2 = dx*dx + dy*dy
        radial = 1 + k3 * r2 + k5 * r2 * r2
        tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
        ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy
        pts_corr[i, 0] = cx + dx * radial + tx
        pts_corr[i, 1] = cy + dy * radial + ty
    return pts_corr

def undistort_image_manual(img, params):
    """应用畸变矫正模型到图像"""
    if params is None:
        return img
    h, w = img.shape[:2]
    cx, cy = params.get('cx', w/2), params.get('cy', h/2)
    k3, k5 = params.get('k3', 0), params.get('k5', 0)
    p1, p2 = params.get('p1', 0), params.get('p2', 0)
    
    xv, yv = np.meshgrid(np.arange(w), np.arange(h))
    dx, dy = xv - cx, yv - cy
    r2 = dx*dx + dy*dy
    radial = 1 + k3 * r2 + k5 * r2 * r2
    tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
    ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy
    
    map_x = (cx + dx * radial + tx).astype(np.float32)
    map_y = (cy + dy * radial + ty).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

# ---------------------------------------------------------
# 全局状态变量
# ---------------------------------------------------------
points = []          # 原图上的 4 个多边形顶点
dragging_idx = -1    # 当前正在拖动的点的索引
original_img = None  # 原始视频帧 (带畸变)
img_undistorted = None # 预处理后的矫正图像 (用于 BEV 实时刷新)
distort_params = None # 畸变参数
output_size = (800, 800) # 鸟瞰图输出尺寸

def draw_overlay():
    """在原图上绘制控制点和连线"""
    img_display = original_img.copy()
    if len(points) == 4:
        # 画多边形连线
        cv2.polylines(img_display, [np.array(points, np.int32)], True, (0, 255, 0), 2)
    
    # 画控制点
    for i, pt in enumerate(points):
        color = (0, 0, 255) if i == dragging_idx else (0, 255, 255)
        cv2.circle(img_display, pt, 8, color, -1)
        cv2.putText(img_display, str(i), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
    return img_display

def update_birds_eye_view():
    """实时计算并显示鸟瞰图 (使用预矫正图像以获得极速体验)"""
    if len(points) != 4:
        return
        
    # 获取图像尺寸
    h, w = original_img.shape[:2]
    
    # 将用户拖动的点 (src_pts) 映射到矫正后的坐标空间
    src_pts = np.array(points, dtype=np.float32)
    src_pts_corr = undistort_points_manual(src_pts, distort_params, w, h)
    
    # 鸟瞰图的目标 4 个矩形点 (固定大小)
    target_w, target_h = output_size
    dst_pts = np.array([
        [0, 0],
        [target_w, 0],
        [target_w, target_h],
        [0, target_h]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵并对预矫正后的图像进行变换
    H = cv2.getPerspectiveTransform(src_pts_corr, dst_pts)
    warped = cv2.warpPerspective(img_undistorted, H, output_size)
    
    cv2.imshow("Birds Eye View (Result)", warped)

def mouse_callback(event, x, y, flags, param):
    global dragging_idx
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # 寻找是否点中了某个控制点 (误差范围内)
        for i, pt in enumerate(points):
            if np.linalg.norm(np.array(pt) - np.array([x, y])) < 15:
                dragging_idx = i
                break
                
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx != -1:
            points[dragging_idx] = (x, y)
            cv2.imshow("Original View (Drag corners)", draw_overlay())
            update_birds_eye_view()
            
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1

def main():
    global original_img, img_undistorted, points, distort_params

    parser = argparse.ArgumentParser(description="Interactive Homography Calibration")
    parser.add_argument("video", nargs="?", default="../input_videos/pample2.mp4", help="Video path")
    args = parser.parse_args()

    # 1. 加载第一帧
    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, _ = read_frames(args.video)
    if len(frames) == 0:
        print("Error: Could not read video frames.")
        return
        
    # OpenCV 是 BGR 格式
    original_img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    h, w = original_img.shape[:2]

    # 加载畸变模型参数
    distort_params = load_distortion_params(args.video)
    
    # 【性能优化】在程序启动时预先完成整个画面的畸变矫正
    # 这样在鼠标拖动时，只需要做轻量级的 warpPerspective，不再重复解析 map/remap
    if distort_params:
        print(f"[✓] 检测到畸变参数，正在预先处理矫正画面以保证实时流畅性...")
        img_undistorted = undistort_image_manual(original_img, distort_params)
        print(f"[✓] 图像预矫正完成，已进入实时校准模式")
    else:
        print(f"[!] 未检测到畸变 JSON，将按照普通平面直接拉伸")
        img_undistorted = original_img.copy()
    
    # 初始化 4 个控制点（在画面中间画个大矩形）
    margin = 200
    points = [
        (margin, margin),
        (w - margin, margin),
        (w - margin, h - margin),
        (margin, h - margin)
    ]

    # 2. 设置界面
    cv2.namedWindow("Original View (Drag corners)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original View (Drag corners)", mouse_callback)
    
    # 初始化显示
    cv2.imshow("Original View (Drag corners)", draw_overlay())
    update_birds_eye_view()
    
    print("\n[交互式校准指南 - 流畅优化版]")
    print("1. 现在所有的重度图像计算已预处理完成，拖动手感应该非常丝滑。")
    print("2. 在 'Original View' 窗口中拖动点，'Birds Eye View' 会即时反馈结果。")
    print("3. 按 'S' 键保存并退出。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('s') or key == ord('S'):
            print("\n[保存成功] 请将以下点位复制到您的主程序中：")
            print("src_pts = [")
            for pt in points:
                print(f"    [{pt[0]}, {pt[1]}],")
            print("]")
            
            # --- 保存转换前后的图片 ---
            out_dir = os.path.join("output", "calibration_debug")
            if not os.path.exists(out_dir):
                 out_dir = os.path.join("..", "output", "calibration_debug")
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.video))[0]
            
            # 1. 保存带有标注的原图
            orig_save_path = os.path.join(out_dir, f"{base_name}_gui_original_points.jpg")
            cv2.imwrite(orig_save_path, draw_overlay())
            
            # 2. 使用之前预处理好的 img_undistorted 直接保存 BEV 图
            tw, th = output_size
            src_np = np.array(points, dtype=np.float32)
            src_corr = undistort_points_manual(src_np, distort_params, w, h)
            dst_np = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_corr, dst_np)
            warped = cv2.warpPerspective(img_undistorted, H, output_size)
            
            bev_save_path = os.path.join(out_dir, f"{base_name}_gui_birds_eye_view.jpg")
            cv2.imwrite(bev_save_path, warped)
            
            print(f"\n[图片已保存]")
            print(f"1. 带有标定的参数原图: {orig_save_path}")
            print(f"2. 最终生成的鸟瞰图（经畸变矫正）: {bev_save_path}")
            
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
