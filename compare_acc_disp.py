import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, rfftfreq
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import euclidean, cosine

# ================= Configuration =================
SEISMIC_TXT_PATH = r"txt_home/ID210711_room_trimed206.txt"
# Assuming we want to compare with the corrected cancellation outputs:
VISUAL_CSV_PATH = r"output/sample_output_cancel/sample_corrected_surface_waveform.csv"

# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz from sample.mp4

# Integration Filter Settings
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz) to remove baseline drift

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    """Fourth-order Butterworth high-pass filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
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

def align_and_evaluate(ref_signal, test_signal):
    """
    Synchronize two signals using cross-correlation and calculate metrics.
    Both signals must be at the same sampling rate.
    """
    # Normalize signals for correlation
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test_norm = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    
    # Cross-correlation to find time shift
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
        
    # Trim to same length
    min_len = min(len(ref_aligned), len(test_aligned))
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

    # User mappings: EW -> Y-axis, NS -> X-axis
    # Unit: g -> mm/s^2
    acc_y = seis_df['EW'].values * G_TO_MM
    acc_x = seis_df['NS'].values * G_TO_MM

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

    print("4. Upsampling Visual Data to match Seismometer (100Hz) for precise alignment...")
    # Resample 25Hz visual to 100Hz (4x upsampling) 
    # Use linear interpolation based on time
    disp_vis_x_100hz = np.interp(time_seis, time_vis, disp_vis_x)
    disp_vis_y_100hz = np.interp(time_seis, time_vis, disp_vis_y)

    print("5. Aligning timelines & calculating correlation...")
    shift_x, r_x, mae_x, rmse_x, euclid_x, cos_sim_x, wass_dist_x, ref_x, test_x = align_and_evaluate(disp_seis_x, disp_vis_x_100hz)
    shift_y, r_y, mae_y, rmse_y, euclid_y, cos_sim_y, wass_dist_y, ref_y, test_y = align_and_evaluate(disp_seis_y, disp_vis_y_100hz)
    
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
    print(f"X-Axis (Visual) vs NS (Seismometer):")
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
    
    print(f"Y-Axis (Visual) vs EW (Seismometer):")
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
    plt.title(f"X-Axis Time Domain (Visual vs Seismometer NS) | R: {r_x:.4f}")
    plt.plot(t_plot_x, ref_x, label="Seismometer NS (GT)", color='black', alpha=0.7)
    plt.plot(t_plot_x, test_x, label="Visual Output X", color='blue', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Disp)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Y-Axis Time Domain
    plt.subplot(4, 1, 2)
    plt.title(f"Y-Axis Time Domain (Visual vs Seismometer EW) | R: {r_y:.4f}")
    plt.plot(t_plot_y, ref_y, label="Seismometer EW (GT)", color='black', alpha=0.7)
    plt.plot(t_plot_y, test_y, label="Visual Output Y", color='red', alpha=0.8, linestyle='--')
    plt.ylabel("Z-Score (Disp)")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 3: X-Axis Frequency Domain
    plt.subplot(4, 1, 3)
    plt.title(f"X-Axis Frequency Spectrum | Dominant Freq Error: {f_err_x:.2f}%")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seis NS ({peak_f_seis_x:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_x, amp_vis_x, label=f"Visual X ({peak_f_vis_x:.2f}Hz)", color='blue', alpha=0.8)
    plt.xlim(0, 10)  # Usually structural vibrations are < 10Hz
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 4: Y-Axis Frequency Domain
    plt.subplot(4, 1, 4)
    plt.title(f"Y-Axis Frequency Spectrum | Dominant Freq Error: {f_err_y:.2f}%")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seis EW ({peak_f_seis_y:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_y, amp_vis_y, label=f"Visual Y ({peak_f_vis_y:.2f}Hz)", color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    out_img = "output/visual_vs_seismic_alignment_with_fft.png"
    plt.savefig(out_img, dpi=300)
    print(f"Comparison plot saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
