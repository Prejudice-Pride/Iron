import os
import sys
import cv2
import numpy as np
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.video_io import read_frames

# ---------------------------------------------------------
# 全局状态变量
# ---------------------------------------------------------
# 初始像素坐标
src_pts = [
    [1002, 544],
    [1375, 620],
    [1155, 925],
    [608, 717],]
# 对应的物理空间坐标 (mm)
dst_pts = [
    [0, 0],
    [880, 0],
    [880, 150],
    [0, 150],]

dragging_idx = -1    # 当前拖动的点
original_img = None  # 原始画面

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
    
    # 重新计算单应性矩阵 (像素坐标 -> 物理坐标)
    H, mask = cv2.findHomography(src_np, dst_np, cv2.RANSAC, 5.0)
    if H is None:
        return
        
    # 计算复合矩阵 M: 原图像素 -> 物理空间 -> 画布空间
    M = S_matrix @ H
    
    # 生成鸟瞰图
    warped = cv2.warpPerspective(original_img, M, (canvas_w, canvas_h))
    
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
    global original_img

    parser = argparse.ArgumentParser(description="True Proportion Homography Calibration")
    parser.add_argument("video", nargs="?", default="../input_videos/sample.mp4", help="Video path")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found at {args.video}")
        return

    frames, _ = read_frames(args.video)
    if len(frames) == 0:
        print("Error: Could not read video frames.")
        return
        
    original_img = cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR)
    
    # 初始化真实比例的画布参数
    init_canvas_params()

    cv2.namedWindow("Original View (Drag points)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("True Proportion Bird's Eye View", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Original View (Drag points)", mouse_callback)
    
    cv2.imshow("Original View (Drag points)", draw_overlay())
    update_birds_eye_view()
    
    print("\n[真实等比例校准工具]")
    print("- 原图有 7 个控制点，你可以直接拖拽任何一点修正其实际位置。")
    print("- 鸟瞰图保持了 1:1 的物理比例。如果在鸟瞰图里某些直线还是弯的，或比例不对，试着微调最边缘离谱的那个点。")
    print("- 调整满意后，按 'S' 键保存最新的点位到终端并退出。")

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
