import pandas as pd
import numpy as np
import argparse
from scipy.signal import butter, filtfilt, savgol_filter

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    effective_cutoff = min(cutoff, nyq * 0.9)
    if effective_cutoff <= 0:
        return data
    normal_cutoff = effective_cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    mask = ~np.isnan(data)
    if np.sum(mask) > 15:
        filtered = np.copy(data)
        filtered[mask] = filtfilt(b, a, data[mask])
        return filtered
    return data

def map_to_gbt17742(pga):
    """依据中国地震烈度表 (GB/T 17742-2020) 进行简约映射"""
    if pga < 0.8: return 1
    if pga < 2.5: return 2
    if pga < 8:   return 3
    if pga < 22:  return 4
    if pga < 44:  return 5
    if pga < 89:  return 6
    if pga < 177: return 7
    if pga < 353: return 8
    if pga < 707: return 9
    if pga < 1414: return 10
    return 12

def process_waveform(csv_path):
    df = pd.read_csv(csv_path)
    
    # 提取位移数据与时间戳
    time = df['time'].values
    dt_arr = np.diff(time)
    dt = np.median(dt_arr)
    fps = 1.0 / dt
    
    # 获取已经消除过相机抖动倾向的物理位移(假设该CSV已是物理维度)
    disp_cm = np.column_stack([df['mean_x'].values, df['mean_y'].values])
    T = len(disp_cm)
    
    # --- 1. Signal Smoothing ---
    cutoff = 1.5
    disp_cm_smooth = np.zeros_like(disp_cm)
    disp_cm_smooth[:, 0] = lowpass_filter(disp_cm[:, 0], cutoff, fs=fps)
    disp_cm_smooth[:, 1] = lowpass_filter(disp_cm[:, 1], cutoff, fs=fps)
    
    # --- 2. Kinematics ---
    window_length = min(15, T // 2 * 2 + 1)
    if window_length < 5:
        window_length = 5
        
    vel = np.zeros_like(disp_cm_smooth)
    acc = np.zeros_like(disp_cm_smooth)
    
    for ax in range(2):
        vel[:, ax] = savgol_filter(disp_cm_smooth[:, ax], window_length=window_length, polyorder=3, deriv=1, delta=dt)
        acc[:, ax] = savgol_filter(disp_cm_smooth[:, ax], window_length=window_length, polyorder=3, deriv=2, delta=dt)

    # 再次低通滤除求导撕裂噪声
    acc[:, 0] = lowpass_filter(acc[:, 0], cutoff, fs=fps)
    acc[:, 1] = lowpass_filter(acc[:, 1], cutoff, fs=fps)

    # --- 3. Peak Feature Extraction ---
    # 消除边缘滤波振铃
    margin = min(10, T // 10)
    if T > 2 * margin:
        acc_eval = acc[margin:-margin]
        vel_eval = vel[margin:-margin]
        T_eval = T - 2 * margin
    else:
        acc_eval = acc
        vel_eval = vel
        T_eval = T

    acc_mags_all = np.sqrt(np.sum(acc_eval**2, axis=1))
    vel_mags_all = np.sqrt(np.sum(vel_eval**2, axis=1))

    window_size = int(2.0 * fps)
    if window_size > T_eval: window_size = T_eval

    max_pga = 0.0
    max_pgv = 0.0
    for i in range(T_eval - window_size + 1):
        max_pga = max(max_pga, np.max(acc_mags_all[i : i + window_size]))
        max_pgv = max(max_pgv, np.max(vel_mags_all[i : i + window_size]))

    p95_pga = np.percentile(acc_mags_all, 95)
    effective_pga = 0.75 * max_pga

    # 映射烈度
    intensity_m1 = map_to_gbt17742(max_pga)
    intensity_m2 = map_to_gbt17742(p95_pga)
    intensity_m3 = map_to_gbt17742(effective_pga)

    # --- 4. 生成报告 ---
    print("======================================================================")
    print("地震烈度多维度分析报告 (Multi-Dimensional Seismic Analysis)")
    print("======================================================================")
    print("校准模式: 线性缩放 (1.0 cm/pix) 或 透视变换 (Homography 持平)")
    print(f"分析窗口: 2.0s ({window_size} 帧)")
    print("-" * 70)
    print(f"[方法 1] 传统峰值法 (PGA - Absolute Peak):")
    print(f"   最大水平加速度: {max_pga:.2f} cm/s^2 ({max_pga:.2f} Gal)")
    print(f"   最大水平速度:   {max_pgv:.2f} cm/s")
    print(f"   >>> 评估烈度: {intensity_m1} 度")
    print("-" * 40)
    print(f"[方法 2] 统计百分位法 (P95 Robust Peak):")
    print(f"   P95 加速度:     {p95_pga:.2f} cm/s^2 ({p95_pga:.2f} Gal)")
    print(f"   >>> 评估烈度: {intensity_m2} 度 (更具抗噪性)")
    print("-" * 40)
    print(f"[方法 3] 有效设计加速度 (Effective Acceleration):")
    print(f"   有效加速度:     {effective_pga:.2f} cm/s^2 ({effective_pga:.2f} Gal)")
    print(f"   >>> 评估烈度: {intensity_m3} 度 (结构设计实际感受)")
    print("======================================================================")

if __name__ == '__main__':
    DEFAULT_CSV_PATH = r"output\sample3_output_cancel\sample3_corrected_surface_waveform.csv"
    parser = argparse.ArgumentParser(description="从 waveform.csv 中计算地震烈度指标")
    parser.add_argument('csv_path', type=str, nargs='?', default=DEFAULT_CSV_PATH, help='输入的 tracking waveform csv 文件路径 (可选)')
    args = parser.parse_args()
    process_waveform(args.csv_path)