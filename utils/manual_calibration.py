import cv2
import numpy as np
import os
from utils.self_calibration import SelfCalibration

def manual_tune(video_path):
    """
    交互式滑块调参工具
    使用方法：
    1. 拖动 K3 和 K5 滑块观察图像变化
    2. 红色垂直参考线可以帮助你判断直线是否校正成功
    3. 调整满意后，按 'ESC' 键退出并打印最终参数
    """
    if not os.path.exists(video_path):
        print(f"错误: 找不到视频文件 {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误: 无法读取视频帧")
        return
    
    # 创建两个独立的窗口：一个用于调节控制，一个用于显示视频结果
    cv2.namedWindow('Control Panel', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Control Panel', 400, 300)
    
    cv2.namedWindow('Corrected Result', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Corrected Result', 1280, 720)
    
    # 在 'Control Panel' 窗口创建滑动条
    cv2.createTrackbar('K3 (offset 1000, scale 1e-9)', 'Control Panel', 1000, 2000, lambda x: None)
    cv2.createTrackbar('K5 (offset 1000, scale 1e-15)', 'Control Panel', 1000, 2000, lambda x: None)
    cv2.createTrackbar('CenterX', 'Control Panel', int(frame.shape[1]/2), frame.shape[1], lambda x: None)
    cv2.createTrackbar('CenterY', 'Control Panel', int(frame.shape[0]/2), frame.shape[0], lambda x: None)

    calibrator = SelfCalibration()

    print("\n[交互模式已启动]")
    print("- 请在 'Control Panel' 窗口调节滑块")
    print("- 在 'Corrected Result' 窗口观察实时效果")
    print("- 点击任意窗口并按 'ESC' 键退出并保存参数")

    while True:
        # 1. 从 'Control Panel' 窗口获取滑块值
        k3_raw = cv2.getTrackbarPos('K3 (offset 1000, scale 1e-9)', 'Control Panel')
        k5_raw = cv2.getTrackbarPos('K5 (offset 1000, scale 1e-15)', 'Control Panel')
        cx = cv2.getTrackbarPos('CenterX', 'Control Panel')
        cy = cv2.getTrackbarPos('CenterY', 'Control Panel')
        
        # 2. 转换参数
        k3_val = (k3_raw - 1000) * 1e-9
        k5_val = (k5_raw - 1000) * 1e-15
        
        calibrator.k3 = k3_val
        calibrator.k5 = k5_val
        calibrator.center = (float(cx), float(cy))
        
        # 3. 执行校正
        corrected = calibrator.apply_to_image(frame)
        
        # 4. 绘制参考网格线
        h, w = corrected.shape[:2]
        # 绘制垂直参考线
        for x in range(0, w, 100):
            cv2.line(corrected, (x, 0), (x, h), (0, 0, 255), 1)
        # 绘制水平参考线
        for y in range(0, h, 100):
            cv2.line(corrected, (0, y), (w, y), (0, 0, 255), 1)
            
        # 叠加参数文字
        info = f"K3: {k3_val:.2e}, K5: {k5_val:.2e}, Center: ({cx}, {cy})"
        cv2.putText(corrected, info, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Corrected Result', corrected)
        
        key = cv2.waitKey(30) & 0xFF
        if key == 27: # ESC 键
            print("-" * 30)
            print(f"最终手动标定参数:")
            print(f"K3: {k3_val:.12e}")
            print(f"K5: {k5_val:.15e}")
            print(f"Center: ({cx}, {cy})")
            print("-" * 30)
            
            # 询问是否保存
            calibrator.save_results("output/calibration_debug")
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    default_video = "input_videos/sample.mp4"
    manual_tune(default_video)
