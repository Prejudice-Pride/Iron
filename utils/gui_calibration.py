import os
import sys
import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_io import read_frames
from utils.self_calibration import SelfCalibration
import json

# ---------------------------------------------------------
# 全局状态变量
# ---------------------------------------------------------
points = []          # 原图上的 4 个多边形顶点
dragging_idx = -1    # 当前正在拖动的点的索引
original_img = None  # 原始视频帧
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
        cv2.putText(img_display, str(i), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1)
        
    return img_display

def update_birds_eye_view():
    """实时计算并显示鸟瞰图"""
    if len(points) != 4:
        return
        
    # 原图的 4 个控制点
    src_pts = np.array(points, dtype=np.float32)
    
    # 鸟瞰图的目标 4 个矩形点 (固定大小)
    w, h = output_size
    dst_pts = np.array([
        [0, 0],
        [w, 0],
        [w, h],
        [0, h]
    ], dtype=np.float32)
    
    # 计算透视变换矩阵并进行变换
    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(original_img, H, output_size)
    
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
    global original_img, points

    parser = argparse.ArgumentParser(description="Interactive Homography Calibration")
    parser.add_argument("video", nargs="?", default="../input_videos/sample.mp4", help="Video path")
    args = parser.parse_args()

    # 1. 加载第一帧
    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, _ = read_frames(args.video)
    if len(frames) == 0:
        print("Error: Could not read video frames.")
        return
        
    # OpenCV 是 BGR 格式，如果是 RGB 需要转换一下
    original_img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    h, w = original_img.shape[:2]
    
    # --- 新增：仿照 cancel 逻辑，加载畸变校准参数并应用 ---
    calib_path = os.path.join("output", "calibration_debug", "distortion_params.json")
    if os.path.exists(calib_path):
        try:
            with open(calib_path, 'r', encoding='utf-8') as f:
                calib = json.load(f)
            print(f"Found distortion params: {calib_path}. Applying correction to first frame...")
            calib_obj = SelfCalibration()
            calib_obj.k3 = float(calib.get('k3', 0.0))
            calib_obj.k5 = float(calib.get('k5', 0.0))
            cx = float(calib.get('cx', w/2.0))
            cy = float(calib.get('cy', h/2.0))
            calib_obj.center = (cx, cy)

            # 对第一帧应用畸变矫正
            original_img = calib_obj.apply_to_image(original_img)
            print('Distortion correction applied to the first frame for calibration.')
        except Exception as e:
            print(f"Warning: Failed to load/apply distortion params: {e}")
    else:
        print(f"No distortion params file found at {calib_path}. Using raw frame.")

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
    
    print("\n[交互式校准指南]")
    print("1. 在 'Original View' 窗口中，用鼠标拖动 4 个红色/黄色控制点")
    print("2. 观察 'Birds Eye View' 窗口，直到物理平面的纹理/线条变得横平竖直、不再有透视畸变")
    print("3. 按 'S' 键保存当前的控制点坐标")
    print("4. 按 'ESC' 键退出")

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27: # ESC
            break
        elif key == ord('s') or key == ord('S'):
            print("\n[保存成功] 请将以下点位复制您的主程序中：")
            print("src_pts = [")
            for pt in points:
                print(f"    [{pt[0]}, {pt[1]}],")
            print("]")
            
            # --- 新增：保存转换前后的图片 ---
            out_dir = os.path.join("output", "calibration_debug")
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(args.video))[0]
            
            # 1. 保存带有标注的原图
            orig_save_path = os.path.join(out_dir, f"{base_name}_gui_original_points.jpg")
            cv2.imwrite(orig_save_path, draw_overlay())
            
            # 2. 重新计算并保存鸟瞰图
            w, h = output_size
            src_pts_np = np.array(points, dtype=np.float32)
            dst_pts_np = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
            H = cv2.getPerspectiveTransform(src_pts_np, dst_pts_np)
            warped = cv2.warpPerspective(original_img, H, output_size)
            
            bev_save_path = os.path.join(out_dir, f"{base_name}_gui_birds_eye_view.jpg")
            cv2.imwrite(bev_save_path, warped)
            
            print(f"\n[图片已保存]")
            print(f"1. 带有标定的原图: {orig_save_path}")
            print(f"2. 展平后的鸟瞰图: {bev_save_path}")
            
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
