import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq

# ================= Configuration =================
VISUAL_CSV_PATH = r"output\sample3_output_cancel\sample3_corrected_surface_waveform.csv"
VISUAL_FS = 25.0                     # 25 Hz

# ================= Helper Functions =================

def compute_fft(signal_data, fs):
    """计算直接测得的位移数据的FFT"""
    n = len(signal_data)
    signal_data = detrend(signal_data)
    window = np.hanning(n)
    signal_windowed = signal_data * window

    yf = rfft(signal_windowed)
    xf = rfftfreq(n, 1 / fs)

    amp = np.abs(yf)
    if np.max(amp[1:]) > 0:
        amp = amp / np.max(amp[1:])
    return xf, amp

# ================= Main Script =================
def main():
    print("1. Loading Visual Tracking Data (25Hz)...")
    try:
        vis_df = pd.read_csv(VISUAL_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    disp_vis_x = vis_df['mean_x'].values
    disp_vis_y = vis_df['mean_y'].values

    print("2. Computing Frequency Spectra for visual displacement...")
    freq_vis_x, amp_vis_x = compute_fft(disp_vis_x, VISUAL_FS)
    freq_vis_y, amp_vis_y = compute_fft(disp_vis_y, VISUAL_FS)

    peak_x = freq_vis_x[np.argmax(amp_vis_x)]
    peak_y = freq_vis_y[np.argmax(amp_vis_y)]

    print("\n" + "="*50)
    print("---  Visual Frequency Analysis ---")
    print(f"X-Axis Peak Frequency: {peak_x:.3f} Hz")
    print(f"Y-Axis Peak Frequency: {peak_y:.3f} Hz")
    print("="*50 + "\n")

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.title("X-Axis Frequency Histogram")
    plt.hist(freq_vis_x, bins=30, weights=amp_vis_x, color='blue', alpha=0.7)
    plt.xlim(0, VISUAL_FS / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Weighted Amplitude")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.title("Y-Axis Frequency Histogram")
    plt.hist(freq_vis_y, bins=30, weights=amp_vis_y, color='red', alpha=0.7)
    plt.xlim(0, VISUAL_FS / 2)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Weighted Amplitude")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
