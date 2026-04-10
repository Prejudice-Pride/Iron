import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, rfftfreq
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

# ================= Configuration =================

# Video axis mapping configurations based on video name
VIDEO_AXIS_CONFIGS = {
    "sample": {"x_map": "NS", "y_map": "WE"},
    "qample": {"x_map": "NS", "y_map": "WE"},
    "pample2": {"x_map": "NS", "y_map": "WE"},
    "qample2": {"x_map": "SN", "y_map": "EW"},
}

# Setup Argument Parser for Automation
parser = argparse.ArgumentParser(description="Evaluate visual tracking vs seismic data.")
parser.add_argument("--seismic_txt", type=str, default=r"txt_home\sample_p2_1301.txt", help="Path to ground truth seismic txt")
parser.add_argument("--visual_csv", type=str, default=r"output\sample_output_cancel\sample_corrected_surface_waveform.csv", help="Path to visual tracking csv")
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
    """
    Auto-detect axis mapping based on CSV filename.
    If x_override or y_override are provided, use those instead.
    """
    video_name = get_video_name_from_path(csv_path)
    
    # Try to find matching config
    detected_x, detected_y = "NS", "WE"  # Default fallback
    
    for video_key, mapping in VIDEO_AXIS_CONFIGS.items():
        if video_key.lower() in video_name.lower():
            detected_x = mapping["x_map"]
            detected_y = mapping["y_map"]
            print(f"✓ Auto-detected axis mapping from video '{video_name}': X→{detected_x}, Y→{detected_y}")
            break
    
    # Override with command-line args if provided
    final_x = x_override if x_override is not None else detected_x
    final_y = y_override if y_override is not None else detected_y
    
    if x_override is not None or y_override is not None:
        print(f"  (Command-line override applied)")
    
    return final_x, final_y

# Auto-detect or use provided mappings
VISUAL_X_MAPS_TO, VISUAL_Y_MAPS_TO = auto_detect_axis_mapping(
    VISUAL_CSV_PATH,
    x_override=args.visual_x_maps_to,
    y_override=args.visual_y_maps_to
)


# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz from sample.mp4

# Integration Filter Settings
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz) to remove baseline drift
LP_CUTOFF = 5.0                      # Low-pass filter cutoff (Hz) to remove high-frequency tracking noise (毛刺)

# Shift search range (seconds)
# Set to None for no limits (search full signal)
SHIFT_MIN_SEC = None
SHIFT_MAX_SEC = None

# Max evaluation duration (seconds)
# Set to a fixed number (e.g., 15.0) to only compare N seconds after alignment, avoiding long noisy tails.
# Set to None to use full overlapping duration.
EVAL_DURATION_SEC = 6.0
MIN_EVAL_DURATION_SEC = 3.5

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    """Fourth-order Butterworth high-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    """Fourth-order Butterworth low-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    # 避免截止频率超过尼奎斯特频率
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def double_integrate(acc, dt, cutoff, fs):
    """
    Integrate acceleration twice to get displacement, 
    with high-pass filtering at each step to prevent drift.
    """
    # 1. Detrend and filter acceleration
    acc = acc - np.mean(acc)
    acc_filt = highpass_filter(acc, cutoff, fs)
    
    # 2. Integrate to velocity
    vel = cumulative_trapezoid(acc_filt, dx=dt, initial=0)
    vel_filt = highpass_filter(vel, cutoff, fs)
    
    # 3. Integrate to displacement
    disp = cumulative_trapezoid(vel_filt, dx=dt, initial=0)
    disp_filt = highpass_filter(disp, cutoff, fs)
    
    return disp_filt

def align_and_evaluate(ref_signal, test_signal, forced_shift=None, shift_min=None, shift_max=None, eval_samples=None, min_eval_samples=None):
    """
    Synchronize two signals using cross-correlation and calculate metrics.
    Both signals must be at the same sampling rate.
    If forced_shift is given, it enforces that shift for alignment.
    """
    # Normalize signals for correlation
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test_norm = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    
    if forced_shift is None:
        # Cross-correlation to find time shift
        cc = correlate(ref_norm, test_norm, mode='full')
        lags = np.arange(-(len(test_norm) - 1), len(ref_norm))

        if shift_min is not None or shift_max is not None:
            if shift_min is None:
                shift_min = lags.min()
            if shift_max is None:
                shift_max = lags.max()

            valid_mask = (lags >= shift_min) & (lags <= shift_max)
            if not np.any(valid_mask):
                raise ValueError(f"No valid shift found in range [{shift_min}, {shift_max}].")

            cc_valid = cc[valid_mask]
            lags_valid = lags[valid_mask]
            shift = lags_valid[np.argmax(cc_valid)]
        else:
            shift = lags[np.argmax(cc)]
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
        
    # Trim to same length
    min_len = min(len(ref_aligned), len(test_aligned))
    if eval_samples is not None:
        min_len = min(min_len, eval_samples)
        
    if min_eval_samples is not None and min_len < min_eval_samples:
        raise ValueError(f"Aligned segment is too short ({min_len} samples). Expected at least {min_eval_samples} samples.")
        
    ref_aligned = ref_aligned[:min_len]
    test_aligned = test_aligned[:min_len]
    
    # Calculate Correlation
    pearsons_r = np.corrcoef(ref_aligned, test_aligned)[0, 1]
    
    # Calculate RMSE on normalized waveforms (shape error)
    rmse = np.sqrt(np.mean((ref_aligned - test_aligned)**2))
    
    # Calculate MAE (Mean Absolute Error)
    mae = np.mean(np.abs(ref_aligned - test_aligned))
    
    # Calculate Euclidean Distance
    euclid_dist = euclidean(ref_aligned, test_aligned)
    
    # Calculate Cosine Similarity (scipy's cosine is 1 - similarity)
    # We want similarity, so 1 - cosine_distance
    cos_sim = 1.0 - cosine(ref_aligned, test_aligned)
    
    # Calculate Wasserstein Distance
    # Note: Wasserstein measures distance between probability distributions.
    # We treat the amplitude values as 1D distributions to assess energy dispersion similarity.
    # To match the paper's comparison style:
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
    amp = amp / np.max(amp)  # Normalize amplitude to 0-1 for easy comparison
    
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
        if mapping == 'NS': return df['NS'].values
        elif mapping == 'SN': return -df['NS'].values
        elif mapping == 'EW': return df['EW'].values
        elif mapping == 'WE': return -df['EW'].values
        else: raise ValueError(f"Unknown axis mapping: {mapping}. Must be one of NS, SN, EW, WE.")

    # User mappings: visual x/y -> seismometer axes
    # Unit: g -> mm/s^2
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
    
    # High-pass filter visual data just in case there's any slow drift
    disp_vis_x = highpass_filter(disp_vis_x - np.mean(disp_vis_x), HP_CUTOFF, VISUAL_FS)
    disp_vis_y = highpass_filter(disp_vis_y - np.mean(disp_vis_y), HP_CUTOFF, VISUAL_FS)
    
    # Low-pass filter structural tracking data to remove high-frequency noise (尖峰毛刺)
    disp_vis_x = lowpass_filter(disp_vis_x, LP_CUTOFF, VISUAL_FS)
    disp_vis_y = lowpass_filter(disp_vis_y, LP_CUTOFF, VISUAL_FS)

    print("4. Upsampling Visual Data to match Seismometer (100Hz) for precise alignment...")
    # Resample 25Hz visual to 100Hz (4x upsampling) 
    # Use linear interpolation based on time
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
        # y 轴严格使用 x 轴的 shift 对齐
        _, r_y, mae_y, rmse_y, euclid_y, cos_sim_y, wass_dist_y, ref_y, test_y = align_and_evaluate(
            disp_seis_y,
            disp_vis_y_100hz,
            forced_shift=shift_x,
            eval_samples=eval_samples,
            min_eval_samples=min_eval_samples,
        )
        shift_y = shift_x
    except ValueError as e:
        print(f"Skipping evaluation: {e}")
        return

    print("6. Computing FFT for Frequency Domain Analysis...")
    # Calculate FFT for X (NS) and Y (EW) aligned segments
    freq_seis_x, amp_seis_x, peak_f_seis_x = compute_fft(ref_x, SEISMIC_FS)
    freq_vis_x, amp_vis_x, peak_f_vis_x = compute_fft(test_x, SEISMIC_FS)
    
    freq_seis_y, amp_seis_y, peak_f_seis_y = compute_fft(ref_y, SEISMIC_FS)
    freq_vis_y, amp_vis_y, peak_f_vis_y = compute_fft(test_y, SEISMIC_FS)
    
    # Error in dominant frequency
    f_err_x = abs(peak_f_seis_x - peak_f_vis_x) / peak_f_seis_x * 100
    f_err_y = abs(peak_f_seis_y - peak_f_vis_y) / peak_f_seis_y * 100

    print("\n" + "="*50)
    print(f"--- 📊 EVALUATION METRICS (对标论文 4.1 & 4.2) ---")
    print(f"X-Axis (Visual) vs {VISUAL_X_MAPS_TO} (Seismometer):")
    print(f"  [1] 时域对齐波形对比 (4.1 手动追踪对标)")
    print(f"  - Peak Time Alignment Shift: {shift_x * SEISMIC_DT:.3f} seconds")
    print(f"  - Pearson Correlation (r):   {r_x:.4f}  (1.0 is perfect)")
    print(f"  - MAE (平均绝对误差):          {mae_x:.4f}")
    print(f"  - RMSE (均方根误差):           {rmse_x:.4f}")
    
    print(f"\n  [2] 统计学特征相似度 (4.2 强震仪数据对标)")
    print(f"  - 欧氏距离 (Euclid Dist):      {euclid_x:.4f}")
    print(f"  - 余弦相似度 (Cosine Sim):     {cos_sim_x:.4f}  (1.0 is perfect)")
    print(f"  - Wasserstein 距离:           {wass_dist_x:.4f}")
    
    print(f"\n  [3] 频域分析")
    print(f"  - Seismometer Dominant Freq: {peak_f_seis_x:.3f} Hz")
    print(f"  - Visual Output Dominant Freq: {peak_f_vis_x:.3f} Hz")
    print(f"  - Frequency Error:           {f_err_x:.2f} %")
    print("-" * 50)
    
    print(f"Y-Axis (Visual) vs {VISUAL_Y_MAPS_TO} (Seismometer):")
    print(f"  [1] 时域对齐波形对比 (4.1 手动追踪对标)")
    print(f"  - Peak Time Alignment Shift: {shift_y * SEISMIC_DT:.3f} seconds")
    print(f"  - Pearson Correlation (r):   {r_y:.4f}  (1.0 is perfect)")
    print(f"  - MAE (平均绝对误差):          {mae_y:.4f}")
    print(f"  - RMSE (均方根误差):           {rmse_y:.4f}")
    
    print(f"\n  [2] 统计学特征相似度 (4.2 强震仪数据对标)")
    print(f"  - 欧氏距离 (Euclid Dist):      {euclid_y:.4f}")
    print(f"  - 余弦相似度 (Cosine Sim):     {cos_sim_y:.4f}  (1.0 is perfect)")
    print(f"  - Wasserstein 距离:           {wass_dist_y:.4f}")
    
    print(f"\n  [3] 频域分析")
    print(f"  - Seismometer Dominant Freq: {peak_f_seis_y:.3f} Hz")
    print(f"  - Visual Output Dominant Freq: {peak_f_vis_y:.3f} Hz")
    print(f"  - Frequency Error:           {f_err_y:.2f} %")
    print("="*50 + "\n")

    print("7. Plotting comparisons...")
    # Change figure size to accommodate 4 subplots
    plt.figure(figsize=(16, 12))
    
    # Timeline generation for aligned segment
    t_plot_x = np.arange(len(ref_x)) * SEISMIC_DT
    t_plot_y = np.arange(len(ref_y)) * SEISMIC_DT

    # Plot 1: X-Axis Time Domain
    plt.subplot(4, 1, 1)
    plt.title(f"X-Axis Time Domain (Visual vs Seismometer {VISUAL_X_MAPS_TO}) | R: {r_x:.4f}")
    plt.plot(t_plot_x, ref_x, label=f"Seismometer {VISUAL_X_MAPS_TO} (GT)", color='black', alpha=0.7)
    plt.plot(t_plot_x, test_x, label="Visual Output X", color='blue', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Disp)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Y-Axis Time Domain
    plt.subplot(4, 1, 2)
    plt.title(f"Y-Axis Time Domain (Visual vs Seismometer {VISUAL_Y_MAPS_TO}) | R: {r_y:.4f}")
    plt.plot(t_plot_y, ref_y, label=f"Seismometer {VISUAL_Y_MAPS_TO} (GT)", color='black', alpha=0.7)
    plt.plot(t_plot_y, test_y, label="Visual Output Y", color='red', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Disp)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: X-Axis Frequency Domain
    plt.subplot(4, 1, 3)
    plt.title(f"X-Axis Frequency Spectrum | Dominant Freq Error: {f_err_x:.2f}%")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seis {VISUAL_X_MAPS_TO} ({peak_f_seis_x:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_x, amp_vis_x, label=f"Visual X ({peak_f_vis_x:.2f}Hz)", color='blue', alpha=0.8)
    plt.xlim(0, 10)  # Usually structural vibrations are < 10Hz
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 4: Y-Axis Frequency Domain
    plt.subplot(4, 1, 4)
    plt.title(f"Y-Axis Frequency Spectrum | Dominant Freq Error: {f_err_y:.2f}%")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seis {VISUAL_Y_MAPS_TO} ({peak_f_seis_y:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_y, amp_vis_y, label=f"Visual Y ({peak_f_vis_y:.2f}Hz)", color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    out_img = OUT_IMG_PATH
    plt.savefig(out_img, dpi=300)
    print(f"Comparison plot saved to {out_img}")
    # OUTPUT_FOR_PARSE
    print(f"__RESULT_R_X__: {r_x:.4f}")
    print(f"__RESULT_R_Y__: {r_y:.4f}")
    # plt.show()  # Disable blocking show for automation

if __name__ == "__main__":
    main()
