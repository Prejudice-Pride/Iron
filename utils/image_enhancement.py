import cv2
import numpy as np

def enhance_frame_for_tracking(frame, use_clahe=True, use_bilateral=False):
    """
    对单帧图像进行增强以优化角点追踪。
    Args:
        frame: 单帧图像 (H, W, 3) 默认 BGR 或 RGB
        use_clahe: bool, 是否应用局部对比度增强自适应直方图均衡化 (CLAHE)
        use_bilateral: bool, 是否应用双边滤波 (保留边缘的降噪)
    Returns:
        增强后的帧图像 (H, W, 3)
    """
    # 转换为灰度图以进行光照和对比度处理
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. 降噪 (可选) - 双边滤波
    if use_bilateral:
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        
    # 2. 局部对比度增强 (强烈推荐) - CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
    # 将处理后的灰度图转换回3通道 (许多深度学习追踪器如Co-Tracker需要3通道输入)
    enhanced_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    
    return enhanced_rgb

def enhance_frames(frames, use_clahe=True, use_bilateral=False):
    """
    对一个视频帧数组进行批量增强。
    Args:
        frames: np.ndarray 形状为 (T, H, W, 3)
        use_clahe: bool, 默认 True
        use_bilateral: bool, 默认 False 
    Returns:
        被增强的视频帧列表 np.ndarray 形状为 (T, H, W, 3)
    """
    if not use_clahe and not use_bilateral:
        return frames
        
    print(f"Applying image enhancements (CLAHE={use_clahe}, Bilateral={use_bilateral})...")
    
    # Create CLAHE object once if needed
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if use_clahe else None
    
    enhanced_frames = np.empty_like(frames)
    for i in range(frames.shape[0]):
        # Inline processing to avoid function call overhead
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        if use_bilateral:
            gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        if use_clahe:
            gray = clahe.apply(gray)
        enhanced_frames[i] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        
    print("Image enhancement complete.")
    return enhanced_frames
