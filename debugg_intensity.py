import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, detrend
import os

def map_to_gbt17742(pga, pgv):
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

def process_and_calculate(target_csv, ref_csv):
    print(f"Loading Target: {target_csv}")
    df_target = pd.read_csv(target_csv)
    
    times = df_target['time'].values
    dt = times[1] - times[0] if len(times) > 1 else (1.0 / 60.0)
    fps = 1.0 / dt
    
    target_x = df_target['mean_x'].values.copy()
    target_y = df_target['mean_y'].values.copy()
    
    for arr in [target_x, target_y]:
        mask = np.isnan(arr)
        if mask.any() and not mask.all():
            arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        elif mask.all():
            arr[:] = 0.0

    print(f"Loading Reference: {ref_csv}")
    df_ref = pd.read_csv(ref_csv)
    ref_x = df_ref['mean_x'].values.copy()
    ref_y = df_ref['mean_y'].values.copy()
    for arr in [ref_x, ref_y]:
        mask = np.isnan(arr)
        if mask.any() and not mask.all():
            arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
        elif mask.all():
            arr[:] = 0.0
            
    # 镜头平滑 (4.0Hz)
    ref_smooth_x = lowpass_filter(ref_x, cutoff=4.0, fs=fps)
    ref_smooth_y = lowpass_filter(ref_y, cutoff=4.0, fs=fps)
    
    # 强制 1.0 抵消
    target_x = target_x - ref_smooth_x
    target_y = target_y - ref_smooth_y
    print("=> Applied reference cancellation (4.0Hz smoothed).")

    # 去趋势
    target_x = detrend(target_x, type='linear')
    target_y = detrend(target_y, type='linear')
    
    disp_cm = np.column_stack((target_x, target_y))
    
    # --- 新版平滑与求导 (cutoff=3.0) ---
    cutoff = 3.0
    disp_cm_smooth = np.zeros_like(disp_cm)
    disp_cm_smooth[:, 0] = lowpass_filter(disp_cm[:, 0], cutoff, fs=fps)
    disp_cm_smooth[:, 1] = lowpass_filter(disp_cm[:, 1], cutoff, fs=fps)

    vel_new = np.gradient(disp_cm_smooth, dt, axis=0)
    vel_new[:, 0] = lowpass_filter(vel_new[:, 0], cutoff, fs=fps)
    vel_new[:, 1] = lowpass_filter(vel_new[:, 1], cutoff, fs=fps)

    acc_new = np.gradient(vel_new, dt, axis=0)
    acc_new[:, 0] = lowpass_filter(acc_new[:, 0], cutoff, fs=fps)
    acc_new[:, 1] = lowpass_filter(acc_new[:, 1], cutoff, fs=fps)

    acc_mags = np.sqrt(np.sum(acc_new**2, axis=1)) # cm/s^2 (Gal)
    vel_mags = np.sqrt(np.sum(vel_new**2, axis=1)) # cm/s

    window_size = int(2.0 * fps)
    if window_size > len(acc_mags): window_size = len(acc_mags)

    max_pga = 0.0
    max_pgv = 0.0
    for i in range(len(acc_mags) - window_size + 1):
        max_pga = max(max_pga, np.max(acc_mags[i : i + window_size]))
        max_pgv = max(max_pgv, np.max(vel_mags[i : i + window_size]))

    p95_pga = np.percentile(acc_mags, 95)
    
    print("\n" + "="*70)
    print("最新计算结果 (带有 4.0Hz 镜头抵消及 3.0Hz PGA 过滤)")
    print("="*70)
    print(f"最大水平加速度 (PGA): {max_pga:.2f} Gal -> 烈度 {map_to_gbt17742(max_pga, max_pgv)} 度")
    print(f"P95 加速度:           {p95_pga:.2f} Gal -> 烈度 {map_to_gbt17742(p95_pga, 0)} 度")
    print("="*70 + "\n")

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else 'output/sample_output_cancel/sample_raw_surface_waveform.csv'
    ref = sys.argv[2] if len(sys.argv) > 2 else 'output/sample_output_cancel/sample_reference_surface_waveform.csv'
    if not os.path.exists(target) or not os.path.exists(ref):
        print("CSV files not found, please provide valid correct target and ref csv paths.")
    else:
        process_and_calculate(target, ref)

