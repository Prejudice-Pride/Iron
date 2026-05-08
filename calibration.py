import os
import sys
import cv2
import numpy as np
import argparse
import json
from functools import lru_cache

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

    # 缓存 remap 表，当图像尺寸和畸变参数相同时复用
    # key 中对浮点数做小数截断以避免过度细粒度
    key = (w, h, round(cx, 3), round(cy, 3), float(k3), float(k5), float(p1), float(p2))
    if not hasattr(undistort_image_manual, '_remap_cache'):
        undistort_image_manual._remap_cache = {}

    cache = undistort_image_manual._remap_cache
    if key in cache:
        map_x, map_y = cache[key]
    else:
        xv, yv = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        dx, dy = xv - cx, yv - cy
        r2 = dx * dx + dy * dy
        radial = 1 + k3 * r2 + k5 * r2 * r2
        tx = 2 * p1 * dx * dy + p2 * (r2 + 2 * dx * dx)
        ty = p1 * (r2 + 2 * dy * dy) + 2 * p2 * dx * dy

        map_x = (cx + dx * radial + tx).astype(np.float32)
        map_y = (cy + dy * radial + ty).astype(np.float32)
        # 限制缓存大小，防止长期运行占用过多内存
        if len(cache) > 8:
            cache.clear()
        cache[key] = (map_x, map_y)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

from core.analyzer import Analyzer
from utils.video_io import read_frames


def load_first_frame(media_path):
    """优先按图片读取；若不是图片，则按视频读取第一帧。"""
    img = cv2.imread(media_path)
    if img is not None:
        # 统一为 RGB，和 read_frames 的返回格式保持一致
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), "image"

    frames, _ = read_frames(media_path)
    if frames is None or len(frames) == 0:
        raise ValueError(f"Could not read media: {media_path}")
    return frames[0].copy(), "video"

def main():
    parser = argparse.ArgumentParser(description="Homography Calibration Debugger")
    parser.add_argument("video", nargs="?", default="input_videos\pample2.mp4", help="Image or video path")
    args = parser.parse_args()

    # 1. 加载第一帧 / 单张图片
    if not os.path.exists(args.video):
        print(f"Error: Media not found at {args.video}")
        return

    img, media_type = load_first_frame(args.video)
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
    src_pts = [[281, 867], [295, 870], [619, 720], [1156, 924], [1321, 704], [1310, 703], [1381, 621], [1437, 542]]
    dst_pts = [[0, 0], [5, 0], [5, 60], [135, 60], [135, 150], [130, 150], [135, 210], [0,60]]
    
    # # sample.mp4(改)
    # src_pts = [[295, 870], [619, 720], [1156, 924], [1321, 704]]
    # dst_pts = [[5, 0], [5, 60], [135, 60], [135, 150]]

    # -------------------------------------------------------------
    # [新增] 需要通过 H 映射到现实坐标系的像素点列表 (x, y)
    # 说明：这里填写图像像素坐标，程序会输出对应的物理坐标(cm)
    # -------------------------------------------------------------
    # query_pts = [
    #     [1687, 419]
    # ]
    
    # # qample.mp4
    # src_pts = [[308, 1002], [325, 1003], [729, 960], [742, 960], [567, 753], [579, 754], [829, 743], [838, 743], [655, 668], [665, 668], [864, 662], [876, 662]]
    # dst_pts = [[0,0],[5,0],[65,15],[70,15],[0,90],[5,90],[65,105],[75,105],[0,150],[5,150],[65,165],[70,165]]
    
    # # qample2.mp4
    src_pts = [[1120, 950], [1132, 948], [989, 808], [1000, 808], [1561, 756], [1568, 755], [874, 687], [880, 687], [835, 642], [840, 642],[1322, 666], [1327, 665], [1216, 628], [1222, 626]]
    dst_pts = [[0, 0], [5, 0], [0, 60], [5, 60], [190, 60], [195, 60], [0, 150], [5, 150], [0, 210], [5, 210],[190, 150],[195, 150],[190, 210],[195, 210]]

    # # pample2.mp4
    src_pts = [[320, 846], [364, 849], [693, 878], [740, 881], [315, 780], [333, 768], [582, 606], [594, 597]]
    dst_pts = [[60,0],[90,0],[330,0],[360,0],[0,70],[0,100],[0,580],[0,610]]
    
    # # sample3.mp4
    # src_pts = [[438, 751], [809, 665], [1036, 611], [1218, 568], [1313, 729], [876, 908]]
    # dst_pts = [[0,0],[250,0],[500,0],[750,0],[375,100],[0,100]]
    
    # sample2.mp4
    # src_pts = [[478, 482], [655, 479], [684, 394], [519, 393]]
    # dst_pts = [[0, 0], [250, 0], [0, 600], [250, 600]]
    
    # cond.mp4
    # src_pts = [[740, 674], [678, 1015], [1057, 1021], [1026, 678]]
    # dst_pts = [[0, 0], [18, 0], [18, 14], [0, 14]]

    # # cond1.mp4
    # src_pts = [[740, 674], [716, 804], [808, 808], [1070, 702]]
    # dst_pts = [[0, 0], [8, 0], [8, 4], [2, 16]]
    
    # # cond2.mp4
    # src_pts = [[688, 669], [715, 806], [802, 596], [1071, 703]]
    # dst_pts = [[0, 0], [8, 0], [8, 4], [2, 16]]

    # # cond3.mp4
    # src_pts = [[740, 670], [692, 937], [1049, 944], [1026, 677]]
    # dst_pts = [[0, 0], [18, 0], [18, 14], [0, 14]]
    
    # # cond3.mp4
    # src_pts = [[688, 669], [644, 994], [1024, 995], [981, 669]] 
    # dst_pts = [[0, 0], [18, 0], [18, 14], [0, 14]]    
    
    # # cond4.mp4
    # src_pts = [[688, 669], [644, 994], [1024, 995], [981, 669]] 
    # dst_pts = [[0, 0], [18, 0], [18, 14], [0, 14]]
    
    print(f"\n[Step 1] Initializing Calibration for: {base_name} ({media_type})")
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
    # [新增] 将 query_pts 从像素坐标映射到物理坐标，并显示在两张图上
    # -------------------------------------------------------------
    # 查询点可选：支持用户注释掉 `query_pts` 或将其设为空列表
    query_pts_loc = globals().get('query_pts', None)
    if not query_pts_loc:
        query_np = None
        query_phys = None
        query_canvas = None
        print("[Info] No query_pts provided or empty; skipping query mapping.")
    else:
        query_np = np.array(query_pts_loc, dtype=np.float32).reshape(-1, 2)
        # 使用 H 得到物理坐标，使用复合矩阵 M (= S @ H) 直接得到画布坐标，减少一次透视变换
        query_phys = cv2.perspectiveTransform(query_np.reshape(-1, 1, 2), H).reshape(-1, 2)
        query_canvas = cv2.perspectiveTransform(query_np.reshape(-1, 1, 2), M).reshape(-1, 2)

        print("[Query Points] Pixel -> Physical (cm):")
        for i, (p_pix, p_phy) in enumerate(zip(query_np, query_phys)):
            print(
                f"  #{i}: ({p_pix[0]:.2f}, {p_pix[1]:.2f}) -> "
                f"({p_phy[0]:.2f}, {p_phy[1]:.2f})"
            )
    
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

    # 3. 绘制 query 点在鸟瞰图上的位置（带物理坐标）
    query_color = (255, 0, 255)  # 紫色
    if query_np is not None:
        for i, (c_pt, p_phy) in enumerate(zip(query_canvas, query_phys)):
            cx, cy = int(c_pt[0]), int(c_pt[1])
            cv2.circle(warped, (cx, cy), 7, query_color, -1)
            label = f"Q{i}:({p_phy[0]:.1f},{p_phy[1]:.1f})cm"
            cv2.putText(warped, label, (cx + 10, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, query_color, 1)

    # 5. 保存调试图像
    debug_img_path = os.path.join(analyzer.output_dir, f"{base_name}_debug_points.jpg")
    bev_path = os.path.join(analyzer.output_dir, f"{base_name}_birds_eye_view.jpg")
    
    # 在原图上画点：绿色为算法采纳的内点，红色为被剔除的离群点
    for i, (p_src, m) in enumerate(zip(src_pts, mask)):
        color = (0, 255, 0) if m else (0, 0, 255) 
        cv2.circle(img, (int(p_src[0]), int(p_src[1])), 8, color, -1)
        cv2.putText(img, str(i), (int(p_src[0])+10, int(p_src[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 在原图上画 query 点（并标注其物理坐标）
    # 当前流程中 img 已经是矫正后的图像，query_pts 也应为矫正后图像坐标
    if query_np is not None:
        query_draw_pts = query_np
        for i, (p_draw, p_phy) in enumerate(zip(query_draw_pts, query_phys)):
            x, y = int(p_draw[0]), int(p_draw[1])
            cv2.circle(img, (x, y), 7, query_color, -1)
            label = f"Q{i}:({p_phy[0]:.1f},{p_phy[1]:.1f})cm"
            cv2.putText(img, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, query_color, 2)

    cv2.imwrite(debug_img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(bev_path, cv2.cvtColor(warped, cv2.COLOR_RGB2BGR))
    
    print("-" * 50)
    print(f"Debug image: {debug_img_path} (绿点: OK, 红点: 误差过大)")
    print(f"Bird's Eye View: {bev_path} (检查在该图中物体是否不再倾斜且比例正常)")
    print("-" * 50)

if __name__ == "__main__":
    main()
