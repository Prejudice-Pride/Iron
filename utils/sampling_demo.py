import cv2
import numpy as np
import os
import sys
import json

# Ensure parent directory is in path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.sampler import Sampler

def generate_grid_points(rois, spacing=30):
    points = []
    for r in rois:
        x1, y1, x2, y2 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
        for y in range(y1 + spacing//2, y2, spacing):
            for x in range(x1 + spacing//2, x2, spacing):
                points.append([x, y])
    return np.array(points, dtype=np.float32) if len(points) > 0 else np.empty((0, 2), dtype=np.float32)

def draw_features(img, rois, points, color=(0, 255, 0)):
    out = img.copy()
    # 先画 ROI 框
    for r in rois:
        cv2.rectangle(out, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (255, 150, 0), 2)
    # 画点
    for pt in points:
        x, y = int(pt[0]), int(pt[1])
        cv2.drawMarker(out, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=10, thickness=2)
        cv2.circle(out, (x, y), 3, color, -1)
    return out

def add_label(img, text):
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(out, (10, 10), (20 + w, 30 + h), (0, 0, 0), -1)
    cv2.putText(out, text, (15, 25 + h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return out

def main():
    video_path = "input_videos/sample.mp4"
    if len(sys.argv) > 1:
        video_path = sys.argv[1]

    if not os.path.exists(video_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        video_path = os.path.join(project_root, video_path)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Error: Could not read video frame.")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rois_path = os.path.join(project_root, "rois", f"{base_name}_cancel_rois.json")
    if not os.path.exists(rois_path):
        rois_path = os.path.join(project_root, "rois", f"{base_name}_rois.json")
    
    rois = []
    if os.path.exists(rois_path):
        with open(rois_path, 'r') as f:
            rois = json.load(f)
            rois = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
    else:
        h, w = frame.shape[:2]
        rois = [[0, 0, w, h]]

    # 1. 原始图像 (明确要求不画 ROI 框)
    img_org = frame.copy()
    img_org_labeled = add_label(img_org, "(a) Original Image")

    # 2. 算法自动采样 (带 ROI)
    sampler = Sampler(roi_dir=os.path.join(project_root, "rois"))
    pts_auto = sampler.detect_roi_points_orb(frame, rois, max_points=100)
    img_auto = draw_features(frame, rois, pts_auto, color=(0, 0, 255))
    img_auto_labeled = add_label(img_auto, "(b) Algo Auto Sampling")

    # 3. 均匀网格采样 (带 ROI)
    pts_grid = generate_grid_points(rois, spacing=40)
    img_grid = draw_features(frame, rois, pts_grid, color=(0, 255, 255))
    img_grid_labeled = add_label(img_grid, "(c) Uniform Grid Sampling")

    # 4. 人工指定点采样 (明确要求不画 ROI 框，传入空列表 [])
    corners_path = os.path.join(project_root, "corner_points", f"{base_name}_corners.json")
    pts_manual = []
    if os.path.exists(corners_path):
        with open(corners_path, 'r') as f:
            data = json.load(f)
            pts_manual = np.array(data, dtype=np.float32)
    else:
        pts_manual = np.empty((0, 2), dtype=np.float32)
        
    img_manual = draw_features(frame, [], pts_manual, color=(255, 0, 255))
    img_manual_labeled = add_label(img_manual, "(d) Manual Spec Sampling")

    # 保存计算
    out_dir = os.path.join(project_root, "output", "sampling_debug")
    os.makedirs(out_dir, exist_ok=True)
    
    cv2.imwrite(os.path.join(out_dir, "0_original_roi.jpg"), img_org_labeled)
    cv2.imwrite(os.path.join(out_dir, "1_auto_sampling.jpg"), img_auto_labeled)
    cv2.imwrite(os.path.join(out_dir, "2_grid_sampling.jpg"), img_grid_labeled)
    cv2.imwrite(os.path.join(out_dir, "3_manual_sampling.jpg"), img_manual_labeled)

    top_row = np.hstack((img_org_labeled, img_auto_labeled))
    bottom_row = np.hstack((img_grid_labeled, img_manual_labeled))
    combined = np.vstack((top_row, bottom_row))

    combined_path = os.path.join(out_dir, "sampling_methods_combined.jpg")
    cv2.imwrite(combined_path, combined)

    print(f"Sampling images successfully generated and saved to {out_dir}")
    print(f"Combined image: {combined_path}")

if __name__ == "__main__":
    main()