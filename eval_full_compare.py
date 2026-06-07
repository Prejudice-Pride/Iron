import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate, coherence, welch
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, rfftfreq
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

# ================= Configuration =================

# Video axis mapping configurations based on video name
VIDEO_AXIS_CONFIGS = {
    "sample": {"x_map": "NS", "y_map": "WE"},
    "sample3": {"x_map": "SN", "y_map": "WE"},
    "qample": {"x_map": "NS", "y_map": "WE"},
    "pample2": {"x_map": "NS", "y_map": "WE"},
    "qample2": {"x_map": "SN", "y_map": "EW"},
}

# Setup Argument Parser for Automation
parser = argparse.ArgumentParser(description="Evaluate visual tracking vs seismic data.")
parser.add_argument("--seismic_txt", type=str, default=r"txt_home\qample_206b.txt", help="Path to ground truth seismic txt")
parser.add_argument("--visual_csv", type=str, default=r"output\qample_output_cancel\qample_raw_surface_waveform.csv", help="Path to visual tracking csv")
parser.add_argument("--out_img", type=str, default="output/visual_vs_seismic_alignment_with_fft.png", help="Path to save output comparison image")
parser.add_argument("--visual_x_maps_to", type=str, default=None, help="Axis mapping for X (optional, auto-detected from filename if not provided)")
parser.add_argument("--visual_y_maps_to", type=str, default=None, help="Axis mapping for Y (optional, auto-detected from filename if not provided)")
args = parser.parse_args()

SEISMIC_TXT_PATH = args.seismic_txt
VISUAL_CSV_PATH = args.visual_csv
OUT_IMG_PATH = args.out_img

def get_video_name_from_path(path):
    """Extract video name (without extension) from path."""
    import os
    filename = os.path.basename(path)
    return os.path.splitext(filename)[0]

def auto_detect_axis_mapping(csv_path, x_override=None, y_override=None):
    video_name = get_video_name_from_path(csv_path)
    detected_x, detected_y = "NS", "WE"
    
    for video_key, mapping in VIDEO_AXIS_CONFIGS.items():
        if video_key.lower() in video_name.lower():
            detected_x = mapping["x_map"]
            detected_y = mapping["y_map"]
            print(f"✓ Auto-detected axis mapping from video '{video_name}': X→{detected_x}, Y→{detected_y}")
            break
            
    final_x = x_override if x_override is not None else detected_x
    final_y = y_override if y_override is not None else detected_y
    if x_override is not None or y_override is not None:
        print(f"  (Command-line override applied)")
    return final_x, final_y

VISUAL_X_MAPS_TO, VISUAL_Y_MAPS_TO = auto_detect_axis_mapping(
    VISUAL_CSV_PATH,
    x_override=args.visual_x_maps_to,
    y_override=args.visual_y_maps_to
)

SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2
VISUAL_FS = 25.0                     # 25 Hz from sample.mp4

# Filter Settings
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz)
LP_CUTOFF = 5.0                      # Low-pass filter cutoff (Hz)

# Shift search range (seconds)
SHIFT_MIN_SEC = None
SHIFT_MAX_SEC = None

# ✅ [关键修改 1] 评估时长：不截断，使用完整波形比较
EVAL_DURATION_SEC = None
MIN_EVAL_DURATION_SEC = 5.0

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def double_integrate(acc, dt, cutoff, fs):
    acc = acc - np.mean(acc)
    acc_filt = highpass_filter(acc, cutoff, fs)
    vel = cumulative_trapezoid(acc_filt, dx=dt, initial=0)
    vel_filt = highpass_filter(vel, cutoff, fs)
    disp = cumulative_trapezoid(vel_filt, dx=dt, initial=0)
    disp_filt = highpass_filter(disp, cutoff, fs)
    return disp_filt

def align_and_evaluate(ref_signal, test_signal, forced_shift=None, shift_min=None, shift_max=None, eval_samples=None, min_eval_samples=None):
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test_norm = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    
    if forced_shift is None:
        cc = correlate(ref_norm, test_norm, mode='full')
        lags = np.arange(-(len(test_norm) - 1), len(ref_norm))

        valid_mask = np.ones(len(lags), dtype=bool)
        
        if shift_min is not None:
            valid_mask &= (lags >= shift_min)
        if shift_max is not None:
            valid_mask &= (lags <= shift_max)
            
        if min_eval_samples is not None:
            overlaps = np.minimum(len(ref_norm) - np.maximum(lags, 0), len(test_norm) - np.maximum(-lags, 0))
            valid_mask &= (overlaps >= min_eval_samples)

        if not np.any(valid_mask):
            raise ValueError(f"No valid shift found. Range constraints or min_eval_samples ({min_eval_samples}) too strict.")

        cc_valid = cc[valid_mask]
        lags_valid = lags[valid_mask]
        shift = lags_valid[np.argmax(cc_valid)]
    else:
        shift = forced_shift

    if shift > 0:
        ref_aligned = ref_norm[shift:]
        test_aligned = test_norm[:-shift]
    elif shift < 0:
        ref_aligned = ref_norm[:shift]
        test_aligned = test_norm[-shift:]
    else:
        ref_aligned = ref_norm
        test_aligned = test_norm
        
    min_len = min(len(ref_aligned), len(test_aligned))
    if eval_samples is not None:
        min_len = min(min_len, eval_samples)
        
    if min_eval_samples is not None and min_len < min_eval_samples:
        raise ValueError(f"Aligned segment is too short ({min_len} samples). Expected at least {min_eval_samples} samples.")
        
    ref_aligned = ref_aligned[:min_len]
    test_aligned = test_aligned[:min_len]
    
    pearsons_r = np.corrcoef(ref_aligned, test_aligned)[0, 1]
    rmse = np.sqrt(np.mean((ref_aligned - test_aligned)**2))
    mae = np.mean(np.abs(ref_aligned - test_aligned))
    euclid_dist = euclidean(ref_aligned, test_aligned)
    cos_sim = 1.0 - cosine(ref_aligned, test_aligned)
    wass_dist = wasserstein_distance(ref_aligned, test_aligned)
    
    return shift, pearsons_r, mae, rmse, euclid_dist, cos_sim, wass_dist, ref_aligned, test_aligned

def compute_fft(signal_data, fs):
    n = len(signal_data)
    nperseg = max(64, n // 3)
    xf, pxx = welch(signal_data, fs=fs, nperseg=nperseg)
    amp = pxx / np.max(pxx)
    peak_idx = np.argmax(amp[1:]) + 1
    peak_freq = xf[peak_idx]
    return xf, amp, peak_freq

def compute_energy_bands(freqs, amp):
    """Calculate energy distribution across specific frequency bands."""
    bands = [(0.5, 2.0), (2.0, 5.0), (5.0, 10.0)]
    total_energy = np.sum(amp**2)
    if total_energy == 0:
        return [0] * len(bands)
    
    percentages = []
    for (f_low, f_high) in bands:
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = np.sum(amp[mask]**2)
        percentages.append((band_energy / total_energy) * 100)
    return percentages

# ================= Main Script =================
def main():
    print("1. Loading Seismometer Data (100Hz)...")
    try:
        seis_df = pd.read_csv(SEISMIC_TXT_PATH, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
    except Exception as e:
        print(f"Error loading TXT: {e}")
        return

    def get_mapped_acc(df, mapping):
        mapping = mapping.upper()
        if mapping == 'NS': return df['NS'].values
        elif mapping == 'SN': return -df['NS'].values
        elif mapping == 'EW': return df['EW'].values
        elif mapping == 'WE': return -df['EW'].values
        else: raise ValueError(f"Unknown axis mapping: {mapping}.")

    acc_x = get_mapped_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM
    acc_y = get_mapped_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM

    print("2. Integrating Acceleration to Displacement...")
    disp_seis_y = double_integrate(acc_y, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    disp_seis_x = double_integrate(acc_x, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    time_seis = np.arange(len(disp_seis_y)) * SEISMIC_DT

    print("3. Loading Visual Tracking Data (25Hz)...")
    try:
        vis_df = pd.read_csv(VISUAL_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    time_vis = vis_df['time'].values
    disp_vis_x = vis_df['mean_x'].values
    disp_vis_y = vis_df['mean_y'].values
    
    disp_vis_x = highpass_filter(disp_vis_x - np.mean(disp_vis_x), HP_CUTOFF, VISUAL_FS)
    disp_vis_y = highpass_filter(disp_vis_y - np.mean(disp_vis_y), HP_CUTOFF, VISUAL_FS)
    disp_vis_x = lowpass_filter(disp_vis_x, LP_CUTOFF, VISUAL_FS)
    disp_vis_y = lowpass_filter(disp_vis_y, LP_CUTOFF, VISUAL_FS)

    print("4. Upsampling Visual Data to match Seismometer (100Hz)...")
    disp_vis_x_100hz = np.interp(time_seis, time_vis, disp_vis_x)
    disp_vis_y_100hz = np.interp(time_seis, time_vis, disp_vis_y)

    print("5. Aligning timelines & calculating correlation...")
    shift_min_samples = int(SHIFT_MIN_SEC * SEISMIC_FS) if SHIFT_MIN_SEC is not None else None
    shift_max_samples = int(SHIFT_MAX_SEC * SEISMIC_FS) if SHIFT_MAX_SEC is not None else None
    eval_samples = int(EVAL_DURATION_SEC * SEISMIC_FS) if EVAL_DURATION_SEC is not None else None
    min_eval_samples = int(MIN_EVAL_DURATION_SEC * SEISMIC_FS) if MIN_EVAL_DURATION_SEC is not None else None

    try:
        shift_x, r_x, mae_x, rmse_x, euclid_x, cos_sim_x, wass_dist_x, ref_x, test_x = align_and_evaluate(
            disp_seis_x,
            disp_vis_x_100hz,
            shift_min=shift_min_samples,
            shift_max=shift_max_samples,
            eval_samples=eval_samples,
            min_eval_samples=min_eval_samples,
        )
        # ✅ 【回归物理本源】：X和Y来自同一个视频，必须具有完全相同的时间延迟(时差)。
        # 不允许为了刷高分而独立滑动，强行绑定Y使用X算出的物理时间差。
        shift_y, r_y, mae_y, rmse_y, euclid_y, cos_sim_y, wass_dist_y, ref_y, test_y = align_and_evaluate(
            disp_seis_y,
            disp_vis_y_100hz,
            forced_shift=shift_x,
            eval_samples=eval_samples,
            min_eval_samples=min_eval_samples,
        )
        
        # 为了计算互相关峰值指标（Peak Cross-Correlation），独立算一下允许最优滑动的Y的最大r值
        _, max_r_y, _, _, _, _, _, _, _ = align_and_evaluate(
            disp_seis_y,
            disp_vis_y_100hz,
            shift_min=shift_min_samples,
            shift_max=shift_max_samples,
            eval_samples=eval_samples,
            min_eval_samples=min_eval_samples,
        )
        max_r_x = r_x # X轴的最优滑动已经在前面算过了

    except ValueError as e:
        print(f"Skipping evaluation: {e}")
        return

    print("6. Computing FFT for Frequency Domain Analysis...")
    freq_seis_x, amp_seis_x, peak_f_seis_x = compute_fft(ref_x, SEISMIC_FS)
    freq_vis_x, amp_vis_x, peak_f_vis_x = compute_fft(test_x, SEISMIC_FS)
    freq_seis_y, amp_seis_y, peak_f_seis_y = compute_fft(ref_y, SEISMIC_FS)
    freq_vis_y, amp_vis_y, peak_f_vis_y = compute_fft(test_y, SEISMIC_FS)
    
    f_err_x = abs(peak_f_seis_x - peak_f_vis_x) / peak_f_seis_x * 100
    f_err_y = abs(peak_f_seis_y - peak_f_vis_y) / peak_f_seis_y * 100

    print("7. Computing Coherence (MSC), Energy Bands, and Spectral Correlation...")
    # Compute Magnitude-Squared Coherence
    # 注意: coherence(Welch's method)必须划段。如果 nperseg 等于信号总长度，
    # 数学上退化为单个periodogram，两个任何随机信号由于无平均操作，其MSC都会强行等于 1.0!
    # 因此这里将 nperseg 设为总数据的 1/3 或 1/4 左右，确保能够进行多段平均。
    nperseg = max(64, len(ref_x) // 3) 
    f_coh_x, Cxy_x = coherence(ref_x, test_x, fs=SEISMIC_FS, nperseg=nperseg)
    f_coh_y, Cxy_y = coherence(ref_y, test_y, fs=SEISMIC_FS, nperseg=nperseg)
    
    def get_mean_coh(f_coh, Cxy, f_min, f_max):
        mask = (f_coh >= f_min) & (f_coh <= f_max)
        return np.mean(Cxy[mask]) if np.any(mask) else 0.0

    msc_05_2_x = get_mean_coh(f_coh_x, Cxy_x, 0.5, 2.0)
    msc_2_5_x = get_mean_coh(f_coh_x, Cxy_x, 2.0, 5.0)
    msc_5_10_x = get_mean_coh(f_coh_x, Cxy_x, 5.0, 10.0)

    msc_05_2_y = get_mean_coh(f_coh_y, Cxy_y, 0.5, 2.0)
    msc_2_5_y = get_mean_coh(f_coh_y, Cxy_y, 2.0, 5.0)
    msc_5_10_y = get_mean_coh(f_coh_y, Cxy_y, 5.0, 10.0)

    # Compute Energy Bands (0.5-2Hz, 2-5Hz, 5-10Hz)
    bands_ref_x = compute_energy_bands(freq_seis_x, amp_seis_x)
    bands_test_x = compute_energy_bands(freq_vis_x, amp_vis_x)
    bands_ref_y = compute_energy_bands(freq_seis_y, amp_seis_y)
    bands_test_y = compute_energy_bands(freq_vis_y, amp_vis_y)
    
    # Spectral Correlation
    r_spec_x = np.corrcoef(amp_seis_x, amp_vis_x)[0, 1]
    r_spec_y = np.corrcoef(amp_seis_y, amp_vis_y)[0, 1]

    print("\n" + "="*80)
    print("--- 📊 综合评价指标体系（详细展开报告） ---")
    print("1. 时域指标 (Time Domain Metrics):")
    print(f"  - 皮尔逊相关系数 (r)       : X轴 = {r_x:.4f} | Y轴 = {r_y:.4f}")
    print(f"  - 互相关峰值 (Peak X-Corr) : X轴 = {max_r_x:.4f} | Y轴 = {max_r_y:.4f}")
    print(f"  - RMSE (Z-score)           : X轴 = {rmse_x:.4f} | Y轴 = {rmse_y:.4f}")
    print(f"  - MAE (Z-score)            : X轴 = {mae_x:.4f} | Y轴 = {mae_y:.4f}")
    
    print("\n2. 频域指标 (Frequency Domain Metrics):")
    print(f"  - 频段相干性 (MSC) 0.5-2.0 Hz : X轴 = {msc_05_2_x:.4f} | Y轴 = {msc_05_2_y:.4f}")
    print(f"  - 频段相干性 (MSC) 2.0-5.0 Hz : X轴 = {msc_2_5_x:.4f} | Y轴 = {msc_2_5_y:.4f}")
    print(f"  - 频段相干性 (MSC) 5.0-10  Hz : X轴 = {msc_5_10_x:.4f} | Y轴 = {msc_5_10_y:.4f}")
    print(f"  - 频谱整体相关系数 (r_spec)   : X轴 = {r_spec_x:.4f} | Y轴 = {r_spec_y:.4f}")
    print(f"  - 主频漂移误差 (Δf_dom)     : X轴 = {f_err_x:.1f}%   | Y轴 = {f_err_y:.1f}%")
    
    print("\n3. 统计分布指标 (Statistical Metrics):")
    print(f"  - Wasserstein距离 : X轴 = {wass_dist_x:.4f} | Y轴 = {wass_dist_y:.4f}")

    print("\n4. 能量结构验证 (Band Energy Ratio):")
    print(f"  ■ X轴 / {VISUAL_X_MAPS_TO}:")
    print(f"     0.5-2.0 Hz : {bands_test_x[0]:4.1f}% (Vis) vs {bands_ref_x[0]:4.1f}% (GT)  -> 余量偏差: {bands_test_x[0]-bands_ref_x[0]:+5.1f}%")
    print(f"     2.0-5.0 Hz : {bands_test_x[1]:4.1f}% (Vis) vs {bands_ref_x[1]:4.1f}% (GT)  -> 余量偏差: {bands_test_x[1]-bands_ref_x[1]:+5.1f}%")
    print(f"     5.0-10  Hz : {bands_test_x[2]:4.1f}% (Vis) vs {bands_ref_x[2]:4.1f}% (GT)  -> 余量偏差: {bands_test_x[2]-bands_ref_x[2]:+5.1f}%")
    print(f"  ■ Y轴 / {VISUAL_Y_MAPS_TO}:")
    print(f"     0.5-2.0 Hz : {bands_test_y[0]:4.1f}% (Vis) vs {bands_ref_y[0]:4.1f}% (GT)  -> 余量偏差: {bands_test_y[0]-bands_ref_y[0]:+5.1f}%")
    print(f"     2.0-5.0 Hz : {bands_test_y[1]:4.1f}% (Vis) vs {bands_ref_y[1]:4.1f}% (GT)  -> 余量偏差: {bands_test_y[1]-bands_ref_y[1]:+5.1f}%")
    print(f"     5.0-10  Hz : {bands_test_y[2]:4.1f}% (Vis) vs {bands_ref_y[2]:4.1f}% (GT)  -> 余量偏差: {bands_test_y[2]-bands_ref_y[2]:+5.1f}%")

    print("\n" + "-"*80)
    print("--- 📑 工程与交叉核验综合汇总表 ---")
    print(f"┌{'-'*21}┬{'-'*25}┬{'-'*6}┬{'-'*6}┬{'-'*55}┐")
    print(f"| {'类别':<19} | {'指标':<23} | {'X轴':<4} | {'Y轴':<4} | {'说明':<53} |")
    print(f"├{'-'*21}┼{'-'*25}┼{'-'*6}┼{'-'*6}┼{'-'*55}┤")
    print(f"| 时域-相关性         | 皮尔逊相关系数 r        | {r_x:.2f} | {r_y:.2f} | 受土-结构耦合传递函数及信噪比限制，数值偏低     |")
    print(f"| 时域-相关性         | 互相关峰值              | {max_r_x:.2f} | {max_r_y:.2f} | 允许独立最优时移补偿后的最大共振相关度          |")
    print(f"| 时域-偏差           | RMSE (Z-score)          | {rmse_x:.2f} | {rmse_y:.2f} | 反映标准化后波形的整体形态变异度，非测量误差    |")
    print(f"| 时域-偏差           | MAE (Z-score)           | {mae_x:.2f} | {mae_y:.2f} | 同上                                            |")
    print(f"├{'-'*21}┼{'-'*25}┼{'-'*6}┼{'-'*6}┼{'-'*55}┤")
    print(f"| 频域-相干 (核心)    | MSC (0.5-2 Hz)主频段    | {msc_05_2_x:.2f} | {msc_05_2_y:.2f} | 天然容忍跨物理系统信号间的未知线性传递函数关系  |")
    print(f"| 频域-相干 (辅助)    | MSC (2-5 Hz)亚频段      | {msc_2_5_x:.2f} | {msc_2_5_y:.2f} |                                                 |")
    print(f"├{'-'*21}┼{'-'*25}┼{'-'*6}┼{'-'*6}┼{'-'*55}┤")
    print(f"| 频域-结构           | 频谱相关系数 r_spec     | {r_spec_x:.2f} | {r_spec_y:.2f} | 反映整体频谱起伏形态的相似度，不关注绝对幅值    |")
    print(f"| 频域-结构           | 主频误差 Δf_dom       | {f_err_x:4.1f}%| {f_err_y:4.1f}%| 受频率分辨率限制，且对双峰不敏感，作次要参考    |")
    print(f"├{'-'*21}┼{'-'*25}┼{'-'*6}┼{'-'*6}┼{'-'*55}┤")
    print(f"| 统计分布            | Wasserstein 距离        | {wass_dist_x:.2f} | {wass_dist_y:.2f} | 衡量概率密度分布相似度，不依赖逐点对齐          |")
    print(f"└{'-'*21}┴{'-'*25}┴{'-'*6}┴{'-'*6}┴{'-'*55}┘")
    print("="*80 + "\n")

    print("8. Plotting comparisons...")
    plt.figure(figsize=(16, 18))

    t_plot_x = np.arange(len(ref_x)) * SEISMIC_DT
    t_plot_y = np.arange(len(ref_y)) * SEISMIC_DT

    plt.subplot(6, 1, 1)
    plt.title(f"X-Axis Time Domain")
    plt.plot(t_plot_x, ref_x, label=f"Seismometer {VISUAL_X_MAPS_TO}", color='black', alpha=0.7)
    plt.plot(t_plot_x, test_x, label="Visual X", color='blue', alpha=0.8, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Z-score)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(6, 1, 2)
    plt.title(f"Y-Axis Time Domain")
    plt.plot(t_plot_y, ref_y, label=f"Seismometer {VISUAL_Y_MAPS_TO}", color='black', alpha=0.7)
    plt.plot(t_plot_y, test_y, label="Visual Y", color='red', alpha=0.8, linestyle='--')
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (Z-score)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(6, 1, 3)
    plt.title("X-Axis Frequency Spectrum (Welch PSD)")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seis {VISUAL_X_MAPS_TO}", color='black', alpha=0.7)
    plt.plot(freq_vis_x, amp_vis_x, label="Visual X", color='blue', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(6, 1, 4)
    plt.title("Y-Axis Frequency Spectrum (Welch PSD)")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seis {VISUAL_Y_MAPS_TO}", color='black', alpha=0.7)
    plt.plot(freq_vis_y, amp_vis_y, label="Visual Y", color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized PSD")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(6, 1, 5)
    plt.title("X-Axis Magnitude-Squared Coherence (MSC)")
    plt.plot(f_coh_x, Cxy_x, color='blue', alpha=0.8)
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(6, 1, 6)
    plt.title("Y-Axis Magnitude-Squared Coherence (MSC)")
    plt.plot(f_coh_y, Cxy_y, color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.savefig(OUT_IMG_PATH, dpi=300)
    print(f"Comparison plot saved to {OUT_IMG_PATH}")
    print(f"__RESULT_R_X__: {r_x:.4f}")
    print(f"__RESULT_R_Y__: {r_y:.4f}")

if __name__ == "__main__":
    main()
