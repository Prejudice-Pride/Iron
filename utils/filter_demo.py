import os
import sys
import cv2
import numpy as np

# 解析 utils 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.video_io import read_frames
from utils.self_calibration import SelfCalibration
import json

def generate_enhancement_comparison(video_path, output_dir="output/enhancement_debug"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"正在读取视频的第一帧: {video_path}")
    frames, _ = read_frames(video_path)
    if frames is None or len(frames) == 0:
        print("错误：无法读取视频。")
        return
        
    frame = frames[0]
    
    # 步骤 1：去畸变 (如果存在标定文件)
    calib_path = os.path.join("output", "calibration_debug", "distortion_params.json")
    if os.path.exists(calib_path):
        with open(calib_path, 'r', encoding='utf-8') as f:
            calib = json.load(f)
        calib_obj = SelfCalibration()
        calib_obj.k3 = float(calib.get('k3', 0.0))
        calib_obj.k5 = float(calib.get('k5', 0.0))
        calib_obj.p1 = float(calib.get('p1', 0.0))
        calib_obj.p2 = float(calib.get('p2', 0.0))
        
        H, W = frame.shape[:2]
        calib_obj.center = (float(calib.get('cx', W/2.0)), float(calib.get('cy', H/2.0)))
        
        corrected_frame = calib_obj.apply_to_image(frame)
        print("已应用畸变矫正。")
    else:
        corrected_frame = frame.copy()
        print("未找到畸变参数，使用原图。")

    # 提取 ROI 区域进行特写 (可以根据你的视频进行调整，默认取画面中间偏下的一部分)
    H, W = corrected_frame.shape[:2]
    # 取中心区域一个 400x400 的 crop 作为特写
    crop_size = 400
    start_y = int(H * 0.4)
    start_x = int(W * 0.4)
    
    def crop_and_convert_to_gray(img):
        cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # 1. 畸变校正后的原始状态 (灰度)
    step1_raw_gray = crop_and_convert_to_gray(corrected_frame)
    cv2.imwrite(os.path.join(output_dir, "step1_after_distortion_crop.jpg"), step1_raw_gray)
    
    # 2. 应用双边滤波
    print("正在应用双边滤波 (Bilateral Filter)...")
    step2_bilateral = cv2.bilateralFilter(step1_raw_gray, d=5, sigmaColor=50, sigmaSpace=50)
    cv2.imwrite(os.path.join(output_dir, "step2_after_bilateral_crop.jpg"), step2_bilateral)
    
    # 3. 应用局部对比度增强 (CLAHE) - 基于双边滤波后的结果
    print("正在应用 CLAHE 分块对比度增强...")
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    step3_clahe = clahe.apply(step2_bilateral)
    cv2.imwrite(os.path.join(output_dir, "step3_after_clahe_crop.jpg"), step3_clahe)
    
    # 4. 生成横向拼接图用于论文展示
    combined = np.hstack([step1_raw_gray, step2_bilateral, step3_clahe])
    
    # 为了防止因为裁剪区域太小看不清楚，可以整体放大 1.5 倍
    combined_resized = cv2.resize(combined, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    
    # 也可以加上文字标签 (英文更适合学术，或自行修改为中文)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    color = (255, 255, 255) # 白色文字
    thickness = 2
    
    # 计算每个区块放大后的宽度
    w = int(crop_size * 1.5)
    cv2.putText(combined_resized, "Raw (Grayscale)", (10, 40), font, font_scale, color, thickness)
    cv2.putText(combined_resized, "Bilateral Filter", (w + 10, 40), font, font_scale, color, thickness)
    cv2.putText(combined_resized, "+ CLAHE", (2*w + 10, 40), font, font_scale, color, thickness)

    final_path = os.path.join(output_dir, "enhancement_pipeline_comparison.jpg")
    cv2.imwrite(final_path, combined_resized)
    print(f"\n图像预处理效果比对图已生成: {final_path}")
    
    cv2.namedWindow("Image Quality Enhancement Pipeline", cv2.WINDOW_NORMAL)
    cv2.imshow("Image Quality Enhancement Pipeline", combined_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = "input_videos/sample.mp4"
    if len(sys.argv) > 1:
        video = sys.argv[1]
    
    # 兼容运行路径
    if not os.path.exists(video):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(script_dir)
        fallback = os.path.join(root, video)
        if os.path.exists(fallback):
            video = fallback
            
    generate_enhancement_comparison(video)