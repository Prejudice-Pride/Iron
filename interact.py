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
    """从本地读取畸变参数 JSON"""
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    param_path = f"output/calibration_debug/{base_name}_distortion_params.json"
    if os.path.exists(param_path):
        with open(param_path, 'r') as f:
            return json.load(f)
    return None

def undistort_points_manual(pts, params, w, h):
    """手动将像素坐标点转换到矫正后的位置"""
    if params is None:
        return np.array(pts, dtype=np.float32)
    
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
# 初始像素坐标
src_pts = [[306, 854], [352, 856], [693, 880], [739, 884], [299, 785], [319, 772], [580, 604], [592, 596]]
# 对应的物理空间坐标 (mm)

dst_pts = [[40,0],[85,0],[355,0],[400,0],[0,60],[0,105],[0,585],[0,630]]

dragging_idx = -1    # 当前拖动的点
original_img = None  # 原始画面 (带畸变)
img_undistorted = None # 预矫正后的画面 (用于 BEV 实时刷新)
distort_params = None # 畸变参数

# 鸟瞰图画布参数
canvas_w, canvas_h = 0, 0
S_matrix = None      # 物理到画布的缩放平移矩阵

def init_canvas_params():
    """计算真实等比例的物理缩放矩阵"""
    global canvas_w, canvas_h, S_matrix
    
    dst_np = np.array(dst_pts, dtype=np.float32)
    min_x, min_y = np.min(dst_np, axis=0)
    max_x, max_y = np.max(dst_np, axis=0)
    
    # 物理世界的真实宽高 (mm)
    phys_w = max_x - min_x
    phys_h = max_y - min_y
    
    # 我们不仅想要看到标定点范围，四周再多留出 1000mm 的余量
    margin_mm = 0 
    padded_w = phys_w + 2 * margin_mm
    padded_h = phys_h + 2 * margin_mm
    
    # 假设用户的屏幕最大显示尺寸是 1600x900
    # 为了保持 X 和 Y 是 1:1 的真实物理比例，这里只用一个全局 scale
    scale = min(1600.0 / padded_w, 900.0 / padded_h)
    
    canvas_w = int(padded_w * scale)
    canvas_h = int(padded_h * scale)
    
    # 构建仿射矩阵 S: 先把最小坐标平移到 margin 处，然后再统等比例缩放
    S_matrix = np.array([
        [scale, 0, -(min_x - margin_mm) * scale],
        [0, scale, -(min_y - margin_mm) * scale],
        [0, 0, 1]
    ])

def draw_overlay():
    img_display = original_img.copy()
    
    # 画控制点
    for i, pt in enumerate(src_pts):
        pt_int = (int(pt[0]), int(pt[1]))
        color = (0, 0, 255) if i == dragging_idx else (0, 255, 255)
        cv2.circle(img_display, pt_int, 6, color, -1)
        cv2.putText(img_display, str(i), (pt_int[0]+8, pt_int[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    return img_display

def update_birds_eye_view():
    if len(src_pts) < 4:
        return
        
    src_np = np.array(src_pts, dtype=np.float32)
    dst_np = np.array(dst_pts, dtype=np.float32)
    
    # 获取当前显示画面的宽高
    h, w = original_img.shape[:2]
    
    # 核心优化：在计算 H 之前，先将用户标记的像素点 src_pts 转换到“矫正后”的空间
    src_pts_corr = undistort_points_manual(src_np, distort_params, w, h)
    
    # 重新计算单应性矩阵 (矫正后的像素坐标 -> 物理坐标)
    H, mask = cv2.findHomography(src_pts_corr, dst_np, cv2.RANSAC, 5.0)
    if H is None:
        return
        
    # 计算复合矩阵 M: 矫正图像素 -> 物理空间 -> 画布空间
    M = S_matrix @ H
    
    # 生成鸟瞰图 (直接使用预先矫正好的图像 img_undistorted)
    warped = cv2.warpPerspective(img_undistorted, M, (canvas_w, canvas_h))
    
    # 在鸟瞰图上绘制物理坐标系基准点
    dst_np_reshaped = dst_np.reshape(-1, 1, 2)
    canvas_pts = cv2.perspectiveTransform(dst_np_reshaped, S_matrix).reshape(-1, 2)
    
    for i, (c_pt, p_pt) in enumerate(zip(canvas_pts, dst_pts)):
        cx, cy = int(c_pt[0]), int(c_pt[1])
        cv2.drawMarker(warped, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 15, 2)
        label = f"#{i}:({p_pt[0]},{p_pt[1]})"
        cv2.putText(warped, label, (cx + 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow("True Proportion Bird's Eye View", warped)

def mouse_callback(event, x, y, flags, param):
    global dragging_idx
    
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(src_pts):
            if np.linalg.norm(np.array(pt) - np.array([x, y])) < 15:
                dragging_idx = i
                break
                
    elif event == cv2.EVENT_MOUSEMOVE:
        if dragging_idx != -1:
            src_pts[dragging_idx] = [x, y]
            cv2.imshow("Original View (Drag points)", draw_overlay())
            update_birds_eye_view()
            
    elif event == cv2.EVENT_LBUTTONUP:
        dragging_idx = -1

def main():
    global original_img, img_undistorted, distort_params

    parser = argparse.ArgumentParser(description="True Proportion Homography Calibration")
    parser.add_argument("video", nargs="?", default="input_videos/pample2.mp4", help="Video path")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, _ = read_frames(args.video)
    if len(frames) == 0:
        print("Error: Could not read video frames.")
        return
        
    original_img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    
    # 1. 加载畸变模型
    distort_params = load_distortion_params(args.video)
    
    # 【性能优化】预先矫正画面
    if distort_params:
        print(f"[✓] 检测到畸变参数，正在预计算矫正画面...")
        img_undistorted = undistort_image_manual(original_img, distort_params)
    else:
        print(f"[!] 未检测到畸变参数 JSON，按普通拉平处理")
        img_undistorted = original_img.copy()

    # 初始化真实比例的画布参数
    init_canvas_params()

    cv2.namedWindow("Original View (Drag points)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("True Proportion Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original View (Drag points)", mouse_callback)
    
    cv2.imshow("Original View (Drag points)", draw_overlay())
    update_birds_eye_view()
    
    print("\n[真实等比例校准工具 - 流畅性能版]")
    print("- 左图显示原始画面（拖拽控制点）。")
    print("- 右图（鸟瞰图）使用预先矫正的图像进行秒级刷新。")
    print("- 调整满意后，按 'S' 键保存。")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('s') or key == ord('S'):
            print("\n================ 完美点位保存 ================")
            print("src_pts = [")
            for i, pt in enumerate(src_pts):
                print(f"    [{pt[0]}, {pt[1]}], # Point {i}")
            print("]")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
