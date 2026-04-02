import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, correlate
from scipy.fft import rfft, rfftfreq

# ================= Configuration =================

# Data Paths
SEISMIC_TXT_PATH = r"txt_home\sample_p2_910.txt"
VISUAL_CSV_PATH = r"output\sample_output_cancel\sample_corrected_surface_waveform.csv"

# Axis Mapping (Camera's perspective vs Seismometer)
VISUAL_X_MAPS_TO = 'NS'
VISUAL_Y_MAPS_TO = 'WE'

# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz
VISUAL_DT = 1.0 / VISUAL_FS

# Filter Settings
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz)
LP_CUTOFF = 5.0                      # Low-pass filter cutoff (Hz)

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
    if normal_cutoff >= 1.0:
        normal_cutoff = 0.99
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def double_differentiate(disp, dt, lp_cutoff, fs):
    """
    Differentiate displacement twice to get acceleration,
    with low-pass filtering to suppress amplified high-frequency noise.
    """
    vel = np.gradient(disp, dt)
    vel_filt = lowpass_filter(vel, lp_cutoff, fs)
    
    acc = np.gradient(vel_filt, dt)
    acc_filt = lowpass_filter(acc, lp_cutoff, fs)
    
    return acc_filt

def compute_fft(signal_data, fs):
    """
    Compute FFT for a given 1D signal.
    Returns frequencies and amplitudes.
    """
    n = len(signal_data)
    # Apply Hanning window
    window = np.hanning(n)
    signal_windowed = signal_data * window
    
    yf = rfft(signal_windowed)
    xf = rfftfreq(n, 1 / fs)
    
    amp = np.abs(yf)
    amp = amp / np.max(amp)  # Normalize amplitude to 0-1
    
    peak_idx = np.argmax(amp[1:]) + 1
    peak_freq = xf[peak_idx]
    
    return xf, amp, peak_freq

def calculate_spectral_metrics(amp_ref, amp_test):
    """
    Calculate similarity metrics between two normalized frequency spectra.
    """
    # 1. Spectral Correlation Coefficient (SCC)
    scc = np.corrcoef(amp_ref, amp_test)[0, 1]
    
    # 2. Mean Absolute Spectral Error (MASE)
    mase = np.mean(np.abs(amp_ref - amp_test))
    
    # 3. Spectral Cosine Similarity
    from scipy.spatial.distance import cosine
    cos_sim = 1.0 - cosine(amp_ref, amp_test)
    
    return scc, mase, cos_sim

def align_signals(ref_signal, test_signal):
    """
    Synchronize two signals using cross-correlation.
    """
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test_norm = (test_signal - np.mean(test_signal)) / np.std(test_signal)
    
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
    return shift, ref_aligned[:min_len], test_aligned[:min_len]

# ================= Main Script =================
def main():
    print("1. Loading Seismometer Acceleration Data (100Hz)...")
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
        else: raise ValueError(f"Unknown axis mapping: {mapping}")

    # Unit: g -> mm/s^2
    acc_seis_x = get_mapped_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM
    acc_seis_y = get_mapped_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM

    # Filter seismic acceleration
    acc_seis_x = highpass_filter(acc_seis_x - np.mean(acc_seis_x), HP_CUTOFF, SEISMIC_FS)
    acc_seis_y = highpass_filter(acc_seis_y - np.mean(acc_seis_y), HP_CUTOFF, SEISMIC_FS)
    
    acc_seis_x = lowpass_filter(acc_seis_x, LP_CUTOFF, SEISMIC_FS)
    acc_seis_y = lowpass_filter(acc_seis_y, LP_CUTOFF, SEISMIC_FS)

    time_seis = np.arange(len(acc_seis_y)) * SEISMIC_DT

    print("2. Loading Visual Tracking Data (25Hz) and Differentiating...")
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

    acc_vis_x = double_differentiate(disp_vis_x, VISUAL_DT, LP_CUTOFF, VISUAL_FS)
    acc_vis_y = double_differentiate(disp_vis_y, VISUAL_DT, LP_CUTOFF, VISUAL_FS)

    print("3. Upsampling Visual Acceleration to match Seismometer (100Hz)...")
    acc_vis_x_100hz = np.interp(time_seis, time_vis, acc_vis_x)
    acc_vis_y_100hz = np.interp(time_seis, time_vis, acc_vis_y)

    print("4. Aligning timelines...")
    shift_x, ref_x, test_x = align_signals(acc_seis_x, acc_vis_x_100hz)
    
    # 统一使用 x 轴的 shift 对 y 轴进行对齐
    ref_norm_y = (acc_seis_y - np.mean(acc_seis_y)) / np.std(acc_seis_y)
    test_norm_y = (acc_vis_y_100hz - np.mean(acc_vis_y_100hz)) / np.std(acc_vis_y_100hz)
    
    if shift_x > 0:
        ref_y = ref_norm_y[shift_x:]
        test_y = test_norm_y[:-shift_x]
    elif shift_x < 0:
        ref_y = ref_norm_y[:shift_x]
        test_y = test_norm_y[-shift_x:]
    else:
        ref_y = ref_norm_y
        test_y = test_norm_y
        
    min_len = min(len(ref_y), len(test_y))
    ref_y = ref_y[:min_len]
    test_y = test_y[:min_len]
    
    # Trim x just in case
    min_len_x = min(len(ref_x), len(test_x))
    ref_x = ref_x[:min_len_x]
    test_x = test_x[:min_len_x]

    print("5. Computing FFT for Frequency Domain Analysis...")
    freq_seis_x, amp_seis_x, peak_f_seis_x = compute_fft(ref_x, SEISMIC_FS)
    freq_vis_x, amp_vis_x, peak_f_vis_x = compute_fft(test_x, SEISMIC_FS)
    
    freq_seis_y, amp_seis_y, peak_f_seis_y = compute_fft(ref_y, SEISMIC_FS)
    freq_vis_y, amp_vis_y, peak_f_vis_y = compute_fft(test_y, SEISMIC_FS)

    # --- Spectral Similarity Evaluation ---
    # Focus on 0.5 - 10Hz range for structural dynamics
    mask_x = (freq_seis_x >= 0.5) & (freq_seis_x <= 10.0)
    scc_x, mase_x, spec_cos_x = calculate_spectral_metrics(amp_seis_x[mask_x], amp_vis_x[mask_x])

    mask_y = (freq_seis_y >= 0.5) & (freq_seis_y <= 10.0)
    scc_y, mase_y, spec_cos_y = calculate_spectral_metrics(amp_seis_y[mask_y], amp_vis_y[mask_y])

    f_err_x = abs(peak_f_seis_x - peak_f_vis_x) / peak_f_seis_x * 100
    f_err_y = abs(peak_f_seis_y - peak_f_vis_y) / peak_f_seis_y * 100

    print("\n" + "="*50)
    print(f"--- 📊 ACCELERATION FREQUENCY METRICS ---")
    print(f"X-Axis (Visual vs {VISUAL_X_MAPS_TO}):")
    print(f"  [1] Peak Frequency Analysis")
    print(f"    - Seismometer Dominant Freq:   {peak_f_seis_x:.3f} Hz")
    print(f"    - Visual Output Dominant Freq: {peak_f_vis_x:.3f} Hz")
    print(f"    - Frequency Error:             {f_err_x:.2f} %")
    print(f"  [2] Spectrum Similarity (0.5-10Hz)")
    print(f"    - Spectral Correlation (SCC):  {scc_x:.4f} (1.0 is perfect)")
    print(f"    - Spectral Cosine Sim:         {spec_cos_x:.4f}")
    print(f"    - Mean Spectral Error (MASE):  {mase_x:.4f}")

    print(f"\nY-Axis (Visual vs {VISUAL_Y_MAPS_TO}):")
    print(f"  [1] Peak Frequency Analysis")
    print(f"    - Seismometer Dominant Freq:   {peak_f_seis_y:.3f} Hz")
    print(f"    - Visual Output Dominant Freq: {peak_f_vis_y:.3f} Hz")
    print(f"    - Frequency Error:             {f_err_y:.2f} %")
    print(f"  [2] Spectrum Similarity (0.5-10Hz)")
    print(f"    - Spectral Correlation (SCC):  {scc_y:.4f} (1.0 is perfect)")
    print(f"    - Spectral Cosine Sim:         {spec_cos_y:.4f}")
    print(f"    - Mean Spectral Error (MASE):  {mase_y:.4f}")
    print("="*50 + "\n")

    print("6. Plotting comparisons...")
    plt.figure(figsize=(16, 8))

    # Plot 1: X-Axis Frequency Domain
    plt.subplot(2, 1, 1)
    plt.title(f"X-Axis Acceleration Frequency Spectrum | Freq Error: {f_err_x:.2f}%")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seis Acc {VISUAL_X_MAPS_TO} ({peak_f_seis_x:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_x, amp_vis_x, label=f"Visual Acc X ({peak_f_vis_x:.2f}Hz)", color='blue', alpha=0.8)
    plt.xlim(0, 10)
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Y-Axis Frequency Domain
    plt.subplot(2, 1, 2)
    plt.title(f"Y-Axis Acceleration Frequency Spectrum | Freq Error: {f_err_y:.2f}%")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seis Acc {VISUAL_Y_MAPS_TO} ({peak_f_seis_y:.2f}Hz)", color='black', alpha=0.7)
    plt.plot(freq_vis_y, amp_vis_y, label=f"Visual Acc Y ({peak_f_vis_y:.2f}Hz)", color='red', alpha=0.8)
    plt.xlim(0, 10)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amp")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    out_img = "output/visual_vs_seismic_acceleration_fft.png"
    plt.savefig(out_img, dpi=300)
    print(f"Comparison plot saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
