import cv2
import numpy as np
import os
import pandas as pd

def generate_sine_video(output_video_path, output_csv_path, width=640, height=480, fps=30, duration=10):
    """
    生成一个包含多个做正弦运动点的视频，并保存真实轨迹(Ground Truth)。
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    total_frames = int(fps * duration)
    
    # 定义几个点的运动参数
    # (base_x, base_y, amplitude, frequency_hz, phase_rad)
    points_config = [
        {"label": "Static", "x": 100, "y": 240, "amp": 0, "freq": 0, "phase": 0, "color": (0, 0, 255)}, # Red
        {"label": "Slow",   "x": 200, "y": 240, "amp": 30, "freq": 0.5, "phase": 0, "color": (0, 255, 0)}, # Green
        {"label": "Medium", "x": 300, "y": 240, "amp": 60, "freq": 1.0, "phase": 0, "color": (255, 0, 0)}, # Blue
        {"label": "Fast",   "x": 400, "y": 240, "amp": 50, "freq": 2.0, "phase": np.pi/2, "color": (0, 255, 255)}, # Yellow
        {"label": "Chirp",  "x": 500, "y": 240, "amp": 40, "freq": 0.2, "phase": 0, "color": (255, 0, 255)}, # Magenta (Variable freq handled in loop)
    ]

    gt_data = []

    print(f"Generating video: {output_video_path}")
    print(f"Total frames: {total_frames}")

    for i in range(total_frames):
        # 创建白色背景
        frame = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        t = i / fps
        
        frame_data = {"frame": i, "time": t}

        for p_idx, p in enumerate(points_config):
            base_x = p["x"]
            base_y = p["y"]
            amp = p["amp"]
            
            # 计算位移
            if p["label"] == "Chirp":
                # 频率随时间增加: f(t) = f0 + k*t
                # phase = integral(2*pi*f(t) dt) = 2*pi*(f0*t + 0.5*k*t^2)
                f0 = 0.5
                k = 0.2
                phase = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
                dy = amp * np.sin(phase)
            else:
                freq = p["freq"]
                phase_offset = p["phase"]
                dy = amp * np.sin(2 * np.pi * freq * t + phase_offset)
            
            curr_x = int(base_x)
            curr_y = int(base_y + dy)
            
            # 绘制点
            cv2.circle(frame, (curr_x, curr_y), 5, p["color"], -1)
            
            # 记录数据
            frame_data[f"pt{p_idx}_label"] = p["label"]
            frame_data[f"pt{p_idx}_x"] = curr_x
            frame_data[f"pt{p_idx}_y"] = curr_y
            frame_data[f"pt{p_idx}_dy_gt"] = dy # 相对位移 Ground Truth

        # 添加帧号和时间戳
        cv2.putText(frame, f"Frame: {i} Time: {t:.2f}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        out.write(frame)
        gt_data.append(frame_data)

    out.release()
    
    # 保存 Ground Truth
    df = pd.DataFrame(gt_data)
    df.to_csv(output_csv_path, index=False)
    print(f"Video saved to {output_video_path}")
    print(f"Ground truth saved to {output_csv_path}")

if __name__ == "__main__":
    output_dir = "input_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    video_path = os.path.join(output_dir, "sine_wave.mp4")
    csv_path = os.path.join(output_dir, "sine_wave_gt.csv")
    
    generate_sine_video(video_path, csv_path)
