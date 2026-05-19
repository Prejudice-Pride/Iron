import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid
from scipy.fft import rfft, rfftfreq

# ================= Configuration =================
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz)

FLOORS = ["206a", "206b", "901", "910", "1301"]
COLORS = ['black', 'blue', 'green', 'orange', 'red']

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def double_integrate(acc, dt, cutoff, fs):
    acc = acc - np.mean(acc)
    acc_filt = highpass_filter(acc, cutoff, fs)
    vel = cumulative_trapezoid(acc_filt, dx=dt, initial=0)
    vel_filt = highpass_filter(vel, cutoff, fs)
    disp = cumulative_trapezoid(vel_filt, dx=dt, initial=0)
    disp_filt = highpass_filter(disp, cutoff, fs)
    return disp_filt

def compute_fft(signal_data, fs):
    n = len(signal_data)
    window = np.hanning(n)
    signal_windowed = signal_data * window
    yf = rfft(signal_windowed)
    xf = rfftfreq(n, 1 / fs)
    amp = np.abs(yf)
    amp = amp / np.max(amp)
    return xf, amp

def main():
    parser = argparse.ArgumentParser(description="Plot multi-floor seismic ground truth.")
    parser.add_argument("--txt_dir", type=str, default="txt_home", help="Directory containing txt files")
    parser.add_argument("--video_name", type=str, default="sample_full", help="Base name of the video/experiment (e.g., sample_p2)")
    parser.add_argument("--axis", type=str, choices=["NS", "EW"], default="NS", help="Axis to plot (NS or EW)")
    parser.add_argument("--out_img", type=str, default="output/seismic_multi_floor.png", help="Path to save output image")
    args = parser.parse_args()

    plt.figure(figsize=(14, 10))

    # Time domain subplot
    ax1 = plt.subplot(2, 1, 1)
    ax1.set_title(f"Seismic Ground Truth - Time Domain Displacement ({args.axis})")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Displacement (mm)")
    ax1.grid(True, linestyle=':', alpha=0.7)

    # Frequency domain subplot
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_title(f"Seismic Ground Truth - Frequency Domain ({args.axis})")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Normalized Amplitude")
    ax2.set_xlim(0, 10)
    ax2.grid(True, linestyle=':', alpha=0.7)

    for floor, color in zip(FLOORS, COLORS):
        file_path = os.path.join(args.txt_dir, f"{args.video_name}_{floor}.txt")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}. Skipping floor {floor}.")
            continue
            
        try:
            df = pd.read_csv(file_path, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        acc = df[args.axis].values * G_TO_MM
        disp = double_integrate(acc, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
        time_arr = np.arange(len(disp)) * SEISMIC_DT

        # Plot Time Domain
        ax1.plot(time_arr, disp, label=f"Floor {floor}", color=color, alpha=0.8)

        # Plot Frequency Domain
        freqs, amps = compute_fft(disp, SEISMIC_FS)
        ax2.plot(freqs, amps, label=f"Floor {floor}", color=color, alpha=0.8)

    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    
    os.makedirs(os.path.dirname(args.out_img), exist_ok=True)
    plt.savefig(args.out_img, dpi=300)
    print(f"Plot saved to {args.out_img}")

if __name__ == "__main__":
    main()
