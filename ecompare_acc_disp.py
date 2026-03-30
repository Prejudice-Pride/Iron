import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
from scipy.fft import rfft, rfftfreq
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

# ================= Configuration =================
# qample.mp4
# SEISMIC_TXT_PATH = r"txt_home/qample_trimed206b.txt"
# VISUAL_CSV_PATH = r"output/qample_output_cancel/qample_corrected_surface_waveform.csv"

# qample2.mp4
SEISMIC_TXT_PATH = r"txt_home\sample_p1206b.txt"
VISUAL_CSV_PATH = r"output\sample_output_cancel\sample_corrected_surface_waveform.csv"

# Axis Mapping (Camera's perspective vs Seismometer)
# Camera visual axes mapped to Seismograph axes.
# Options: 'NS' (North-South), 'SN' (South-North, -NS), 'EW' (East-West), 'WE' (West-East, -EW)

# Default
VISUAL_X_MAPS_TO = 'NS'
VISUAL_Y_MAPS_TO = 'EW'

# sample.mp4
VISUAL_X_MAPS_TO = 'NS'
VISUAL_Y_MAPS_TO = 'WE'

# # qample.mp4
# VISUAL_X_MAPS_TO = 'NS'
# VISUAL_Y_MAPS_TO = 'WE'

# # qample2.mp4
# VISUAL_X_MAPS_TO = 'SN'
# VISUAL_Y_MAPS_TO = 'EW'

# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz from sample.mp4

# If visual displacement is not in mm, set proper scale here.
# e.g., pixel to mm conversion factor.
VISUAL_DISP_SCALE_TO_MM = 1.0

# Differentiation filter settings (for disp -> acc)
HP_CUTOFF = 0.5                      # remove low-frequency drift before differentiation
LP_CUTOFF = 8.0                      # suppress high-frequency noise after differentiation

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    """Fourth-order Butterworth high-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 0.999)
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    """Fourth-order Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = np.clip(cutoff / nyq, 1e-6, 0.999)
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def displacement_to_acceleration(disp, fs, hp_cutoff, lp_cutoff):
    """Convert displacement to acceleration by second-order differentiation."""
    dt = 1.0 / fs
    disp = disp - np.mean(disp)
    disp_hp = highpass_filter(disp, hp_cutoff, fs)

    vel = np.gradient(disp_hp, dt)
    acc = np.gradient(vel, dt)
    acc_lp = lowpass_filter(acc, lp_cutoff, fs)
    return acc_lp

def safe_zscore(signal):
    std = np.std(signal)
    if std < 1e-12:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std

def align_and_evaluate(ref_signal, test_signal):
    """
    Synchronize two signals using cross-correlation and calculate metrics.
    Both signals must be at the same sampling rate.
    """
    ref_norm = safe_zscore(ref_signal)
    test_norm = safe_zscore(test_signal)

    cc = correlate(ref_norm, test_norm, mode='full')
    shift = np.argmax(cc) - (len(test_norm) - 1)

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
    ref_aligned = ref_aligned[:min_len]
    test_aligned = test_aligned[:min_len]

    pearsons_r = np.corrcoef(ref_aligned, test_aligned)[0, 1]
    mae = np.mean(np.abs(ref_aligned - test_aligned))
    rmse = np.sqrt(np.mean((ref_aligned - test_aligned) ** 2))
    euclid_dist = euclidean(ref_aligned, test_aligned)
    cos_sim = 1.0 - cosine(ref_aligned, test_aligned)
    wass_dist = wasserstein_distance(ref_aligned, test_aligned)

    return shift, pearsons_r, mae, rmse, euclid_dist, cos_sim, wass_dist, ref_aligned, test_aligned

def compute_fft(signal_data, fs):
    """
    Compute FFT for a given 1D signal.
    Returns frequencies and normalized amplitudes.
    """
    n = len(signal_data)
    # Apply Hanning window to reduce spectral leakage
    window = np.hanning(n)
    signal_windowed = signal_data * window
    
    # Compute Real FFT
    yf = rfft(signal_windowed)
    xf = rfftfreq(n, 1 / fs)
    
    # Get Amplitude
    amp = np.abs(yf)
    max_amp = np.max(amp)
    if max_amp > 0:
        amp = amp / max_amp  # Normalize amplitude to 0-1 for easy comparison
    
    # Find dominant frequency
    peak_idx = np.argmax(amp[1:]) + 1 # Skip DC component
    peak_freq = xf[peak_idx]
    
    return xf, amp, peak_freq

# ================= Main Script =================
def main():
    print("1. Loading Seismometer Data (100Hz)...")
    try:
        seis_df = pd.read_csv(SEISMIC_TXT_PATH, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
    except Exception as e:
        print(f"Error loading TXT: {e}")
        return

    # Helper function to extract and invert axes if strictly needed
    def get_mapped_acc(df, mapping):
        mapping = mapping.upper()
        if mapping == 'NS':
            return df['NS'].values
        elif mapping == 'SN':
            return -df['NS'].values
        elif mapping == 'EW':
            return df['EW'].values
        elif mapping == 'WE':
            return -df['EW'].values
        else:
            raise ValueError(f"Unknown axis mapping: {mapping}. Must be one of NS, SN, EW, WE.")

    # User mappings: visual x/y -> seismometer axes
    # Unit: g -> mm/s^2
    acc_seis_x = get_mapped_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM
    acc_seis_y = get_mapped_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM

    time_seis = np.arange(len(acc_seis_y)) * SEISMIC_DT

    print("3. Loading Visual Tracking Data (25Hz)...")
    try:
        vis_df = pd.read_csv(VISUAL_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    if 'mean_x' not in vis_df.columns or 'mean_y' not in vis_df.columns:
        raise ValueError("Visual CSV must contain columns: mean_x, mean_y")

    if 'time' in vis_df.columns:
        time_vis = vis_df['time'].values
    else:
        time_vis = np.arange(len(vis_df)) / VISUAL_FS

    disp_vis_x = vis_df['mean_x'].values * VISUAL_DISP_SCALE_TO_MM
    disp_vis_y = vis_df['mean_y'].values * VISUAL_DISP_SCALE_TO_MM

    print("4. Converting visual displacement to acceleration (second derivative)...")
    acc_vis_x_25hz = displacement_to_acceleration(disp_vis_x, VISUAL_FS, HP_CUTOFF, LP_CUTOFF)
    acc_vis_y_25hz = displacement_to_acceleration(disp_vis_y, VISUAL_FS, HP_CUTOFF, LP_CUTOFF)

    print("5. Upsampling visual acceleration to match Seismometer (100Hz)...")
    acc_vis_x_100hz = np.interp(time_seis, time_vis, acc_vis_x_25hz)
    acc_vis_y_100hz = np.interp(time_seis, time_vis, acc_vis_y_25hz)

    print("6. Aligning timelines & calculating correlation...")
    shift_x, r_x, mae_x, rmse_x, euclid_x, cos_sim_x, wass_dist_x, ref_x, test_x = align_and_evaluate(acc_seis_x, acc_vis_x_100hz)
    shift_y, r_y, mae_y, rmse_y, euclid_y, cos_sim_y, wass_dist_y, ref_y, test_y = align_and_evaluate(acc_seis_y, acc_vis_y_100hz)
    
    print("7. Computing FFT for Frequency Domain Analysis...")
    freq_seis_x, amp_seis_x, peak_f_seis_x = compute_fft(ref_x, SEISMIC_FS)
    freq_vis_x, amp_vis_x, peak_f_vis_x = compute_fft(test_x, SEISMIC_FS)
    
    freq_seis_y, amp_seis_y, peak_f_seis_y = compute_fft(ref_y, SEISMIC_FS)
    freq_vis_y, amp_vis_y, peak_f_vis_y = compute_fft(test_y, SEISMIC_FS)
    
    # Error in dominant frequency
    f_err_x = abs(peak_f_seis_x - peak_f_vis_x) / peak_f_seis_x * 100 if peak_f_seis_x > 0 else 0.0
    f_err_y = abs(peak_f_seis_y - peak_f_vis_y) / peak_f_seis_y * 100 if peak_f_seis_y > 0 else 0.0

    print("\n" + "="*50)
    print(f"--- 📊 加速度对比指标 (视觉位移二阶导 vs 地震仪加速度) ---")
    print(f"X-Axis (Visual Disp→Acc) vs {VISUAL_X_MAPS_TO} (Seismometer Acc):")
    print(f"  [1] 时域对齐波形对比")
    print(f"  - Peak Time Alignment Shift: {shift_x * SEISMIC_DT:.3f} seconds")
    print(f"  - Pearson Correlation (r):   {r_x:.4f}  (1.0 is perfect)")
    print(f"  - MAE (平均绝对误差):          {mae_x:.4f}")
    print(f"  - RMSE (均方根误差):           {rmse_x:.4f}")
    
    print(f"\n  [2] 统计学特征相似度")
    print(f"  - 欧氏距离 (Euclid Dist):      {euclid_x:.4f}")
    print(f"  - 余弦相似度 (Cosine Sim):     {cos_sim_x:.4f}  (1.0 is perfect)")
    print(f"  - Wasserstein 距离:           {wass_dist_x:.4f}")
    
    print(f"\n  [3] 频域分析")
    print(f"  - Seismometer Dominant Freq: {peak_f_seis_x:.3f} Hz")
    print(f"  - Visual Output Dominant Freq: {peak_f_vis_x:.3f} Hz")
    print(f"  - Frequency Error:           {f_err_x:.2f} %")
    print("-" * 50)
    
    print(f"Y-Axis (Visual Disp→Acc) vs {VISUAL_Y_MAPS_TO} (Seismometer Acc):")
    print(f"  [1] 时域对齐波形对比")
    print(f"  - Peak Time Alignment Shift: {shift_y * SEISMIC_DT:.3f} seconds")
    print(f"  - Pearson Correlation (r):   {r_y:.4f}  (1.0 is perfect)")
    print(f"  - MAE (平均绝对误差):          {mae_y:.4f}")
    print(f"  - RMSE (均方根误差):           {rmse_y:.4f}")
    
    print(f"\n  [2] 统计学特征相似度")
    print(f"  - 欧氏距离 (Euclid Dist):      {euclid_y:.4f}")
    print(f"  - 余弦相似度 (Cosine Sim):     {cos_sim_y:.4f}  (1.0 is perfect)")
    print(f"  - Wasserstein 距离:           {wass_dist_y:.4f}")
    
    print(f"\n  [3] 频域分析")
    print(f"  - Seismometer Dominant Freq: {peak_f_seis_y:.3f} Hz")
    print(f"  - Visual Output Dominant Freq: {peak_f_vis_y:.3f} Hz")
    print(f"  - Frequency Error:           {f_err_y:.2f} %")
    print("="*50 + "\n")

    print("8. Plotting comparisons...")
    plt.figure(figsize=(16, 12))
    
    # Timeline generation for aligned segment
    t_plot_x = np.arange(len(ref_x)) * SEISMIC_DT
    t_plot_y = np.arange(len(ref_y)) * SEISMIC_DT

    # Plot 1: X-Axis Time Domain
    plt.subplot(4, 1, 1)
    plt.title(f"X-Axis Acceleration (Visual Disp→Acc vs Seis {VISUAL_X_MAPS_TO}) | R: {r_x:.4f}")
    plt.plot(t_plot_x, ref_x, label=f"Seismometer {VISUAL_X_MAPS_TO} (Acc)", color='black', alpha=0.7)
    plt.plot(t_plot_x, test_x, label="Visual Disp→Acc X", color='blue', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Acc)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Y-Axis Time Domain
    plt.subplot(4, 1, 2)
    plt.title(f"Y-Axis Acceleration (Visual Disp→Acc vs Seis {VISUAL_Y_MAPS_TO}) | R: {r_y:.4f}")
    plt.plot(t_plot_y, ref_y, label=f"Seismometer {VISUAL_Y_MAPS_TO} (Acc)", color='black', alpha=0.7)
    plt.plot(t_plot_y, test_y, label="Visual Disp→Acc Y", color='red', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Acc)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: X-Axis Frequency Domain
    plt.subplot(4, 1, 3)
    plt.title(f"X-Axis Frequency Spectrum | Dominant Freq Error: {f_err_x:.2f}%")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seis {VISUAL_X_MAPS_TO} ({peak_f_seis_x:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_x, amp_vis_x, label=f"Visual Disp→Acc X ({peak_f_vis_x:.2f}Hz)", color='blue', alpha=0.8)
    plt.xlim(0, 10)  # Usually structural vibrations are < 10Hz
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 4: Y-Axis Frequency Domain
    plt.subplot(4, 1, 4)
    plt.title(f"Y-Axis Frequency Spectrum | Dominant Freq Error: {f_err_y:.2f}%")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seis {VISUAL_Y_MAPS_TO} ({peak_f_seis_y:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_y, amp_vis_y, label=f"Visual Disp→Acc Y ({peak_f_vis_y:.2f}Hz)", color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    out_img = "output/visual_disp2acc_vs_seismic_acc.png"
    os.makedirs(os.path.dirname(out_img), exist_ok=True)
    plt.savefig(out_img, dpi=300)
    print(f"Comparison plot saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
