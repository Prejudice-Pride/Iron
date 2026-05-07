import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import os

def map_to_gbt17742(pga, pgv):
    # GB/T 17742-2020 map in Gal (cm/s^2)
    if pga < 0.8: return 1
    if pga < 2.5: return 2
    if pga < 8:   return 3
    if pga < 22:  return 4
    if pga < 44:  return 5
    if pga < 89:  return 6   # VI degree
    if pga < 177: return 7   # VII degree
    if pga < 353: return 8   # VIII degree
    if pga < 707: return 9
    if pga < 1414: return 10
    return 12

def lowpass_filter(data, cutoff, fs=30.0, order=4):
    nyq = 0.5 * fs
    effective_cutoff = min(cutoff, nyq * 0.9)
    if effective_cutoff <= 0: return data
    normal_cutoff = effective_cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    mask = ~np.isnan(data)
    if np.sum(mask) > 15:
        filtered = np.copy(data)
        filtered[mask] = filtfilt(b, a, data[mask])
        return filtered
    return data

def calculate_intensity_standalone(csv_path):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # 提取所需参数
    times = df['time'].values
    dt = times[1] - times[0] if len(times) > 1 else (1.0 / 60.0)
    fps = 1.0 / dt
    T = len(times)
    
    # 输入轨迹通常为相对位移 cm (直接读出)
    disp_cm = np.column_stack((df['mean_x'].values, df['mean_y'].values))
    
    # --- 安全处理 NaN (插值) --- 
    for i in range(2):
        mask = np.isnan(disp_cm[:, i])
        if mask.any() and not mask.all():
            disp_cm[mask, i] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), disp_cm[~mask, i])
        elif mask.all():
            disp_cm[:, i] = 0.0

    # --- 1. 原代码算法 (使用 diff + vstack 填零) ---
    cutoff = 10.0
    disp_cm_smooth = np.zeros_like(disp_cm)
    disp_cm_smooth[:, 0] = lowpass_filter(disp_cm[:, 0], cutoff, fs=fps)
    disp_cm_smooth[:, 1] = lowpass_filter(disp_cm[:, 1], cutoff, fs=fps)

    vel = np.diff(disp_cm_smooth, axis=0) / dt
    vel[:, 0] = lowpass_filter(vel[:, 0], cutoff, fs=fps)
    vel[:, 1] = lowpass_filter(vel[:, 1], cutoff, fs=fps)
    
    acc = np.diff(vel, axis=0) / dt
    acc[:, 0] = lowpass_filter(acc[:, 0], cutoff, fs=fps)
    acc[:, 1] = lowpass_filter(acc[:, 1], cutoff, fs=fps)
    
    # 原先粗暴的两尾部填零对齐 (会有跳动)
    vel_old = np.vstack([vel, [0, 0]])
    acc_old = np.vstack([acc, [0, 0], [0, 0]])

    acc_mags_all_old = np.sqrt(np.sum(acc_old**2, axis=1)) # cm/s^2 (Gal)
    vel_mags_all_old = np.sqrt(np.sum(vel_old**2, axis=1)) # cm/s

    # --- 2. 修复算法 (使用 np.gradient 居中差分) ---
    vel_new = np.gradient(disp_cm_smooth, dt, axis=0)
    vel_new[:, 0] = lowpass_filter(vel_new[:, 0], cutoff, fs=fps)
    vel_new[:, 1] = lowpass_filter(vel_new[:, 1], cutoff, fs=fps)

    acc_new = np.gradient(vel_new, dt, axis=0)
    acc_new[:, 0] = lowpass_filter(acc_new[:, 0], cutoff, fs=fps)
    acc_new[:, 1] = lowpass_filter(acc_new[:, 1], cutoff, fs=fps)

    acc_mags_all_new = np.sqrt(np.sum(acc_new**2, axis=1)) # cm/s^2 (Gal)
    vel_mags_all_new = np.sqrt(np.sum(vel_new**2, axis=1)) # cm/s

    def compute_metrics(acc_mags, vel_mags):
        window_size = int(2.0 * fps)
        if window_size > len(acc_mags): window_size = len(acc_mags)
        
        max_pga = 0.0
        max_pgv = 0.0
        for i in range(len(acc_mags) - window_size + 1):
            max_pga = max(max_pga, np.max(acc_mags[i : i + window_size]))
            max_pgv = max(max_pgv, np.max(vel_mags[i : i + window_size]))
            
        p95_pga = np.percentile(acc_mags, 95)
        effective_pga = 0.75 * max_pga 
        
        return {
            'max_pga': max_pga, 'max_pgv': max_pgv,
            'p95': p95_pga, 'eff': effective_pga,
            'int1': map_to_gbt17742(max_pga, max_pgv),
            'int2': map_to_gbt17742(p95_pga, 0)
        }

    old_m = compute_metrics(acc_mags_all_old, vel_mags_all_old)
    new_m = compute_metrics(acc_mags_all_new, vel_mags_all_new)
    
    # 理论计算
    # GT配置：频率 f = 1Hz, 单向峰值幅值 A = 2cm
    # 位移方程 x(t) = A * sin(2 * pi * f * t)
    # 理论最大加速度 (PGA) = A * (2 * pi * f)^2
    A = 2.0
    f = 1.0
    w = 2 * np.pi * f
    theoretical_pga = A * (w**2)  # = 2 * 39.478...

    print("\n--- 深入数据本体分析 (Deep Data Analysis) ---")
    
    y_raw = df['mean_y'].values
    y_disp = y_raw - np.nanmean(y_raw)  # Center it roughly
    max_disp_y = np.nanmax(y_disp)
    min_disp_y = np.nanmin(y_disp)
    
    x_raw = df['mean_x'].values
    x_disp = x_raw - np.nanmean(x_raw)
    max_disp_x = np.nanmax(x_disp)
    min_disp_x = np.nanmin(x_disp)
    
    print(f"1. 位移分析:")
    print(f"   CSV中 Y 轴位移幅度: {min_disp_y:.3f} 到 {max_disp_y:.3f} cm, 单强: {(max_disp_y - min_disp_y)/2:.3f} cm")
    print(f"   CSV中 X 轴位移幅度: {min_disp_x:.3f} 到 {max_disp_x:.3f} cm, 单强: {(max_disp_x - min_disp_x)/2:.3f} cm")
    print(f"   采样率 (FPS): {fps:.2f} Hz, 采样间隔 (dt): {dt:.4f} s")
    
    # 速度分析
    vel_raw = np.gradient(disp_cm[:, 1], dt)
    print(f"\n2. 速度分析 (X与Y轴):")
    print(f"   滤波后最大速度 X: {np.nanmax(np.abs(vel_new[:, 0])):.3f} cm/s")
    print(f"   滤波后最大速度 Y: {np.nanmax(np.abs(vel_new[:, 1])):.3f} cm/s")
    
    # 加速度分析
    acc_raw = np.gradient(vel_raw, dt)
    print(f"\n3. 加速度分析 (X与Y轴):")
    print(f"   单次滤波后最大加速度 X: {np.nanmax(np.abs(acc_new[:, 0])):.3f} cm/s^2")
    print(f"   单次滤波后最大加速度 Y: {np.nanmax(np.abs(acc_new[:, 1])):.3f} cm/s^2")
    
    print("\n" + "="*70)
    print("调试脚本 - 烈度算法对比 (原版差分 vs. Gradient差分)")
    print("="*70)
    print(f"理论 Ground Truth 设计指标估算:")
    print(f"   输入: 频率 {f} Hz, 幅值 {A} cm")
    print(f"   理论最大加速度 a(t): {theoretical_pga:.2f} cm/s^2 (Gal)")
    print(f"   理论对应的烈度等级: {map_to_gbt17742(theoretical_pga, 0)} 度")
    print("-" * 70)
    
    print("[1] 原生计算结果 (基于 np.diff 并粗暴填零):")
    print(f"   最大水平加速度 (PGA): {old_m['max_pga']:.2f} Gal -> 烈度 {old_m['int1']} 度")
    print(f"   P95 加速度:           {old_m['p95']:.2f} Gal -> 烈度 {old_m['int2']} 度")
    print("-" * 70)

    print("[2] 优化微分算子后 (基于 np.gradient):")
    print(f"   最大水平加速度 (PGA): {new_m['max_pga']:.2f} Gal -> 烈度 {new_m['int1']} 度")
    print(f"   P95 加速度:           {new_m['p95']:.2f} Gal -> 烈度 {new_m['int2']} 度")
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "output/cond1_output/cond1_tracked_surface_waveform.csv"
    calculate_intensity_standalone(path)
