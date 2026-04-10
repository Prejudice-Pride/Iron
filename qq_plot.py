import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.signal import butter, filtfilt
from scipy.integrate import cumulative_trapezoid

# ================= Configuration (参考 compare_acc_disp.py) =================

# Data Paths
SEISMIC_TXT_PATH = r"txt_home\sample_p1_910.txt"
VISUAL_CSV_PATH = r"output\sample_output_cancel\sample_reference_surface_waveform.csv"

# Axis Mapping
VISUAL_X_MAPS_TO = 'SN'
VISUAL_Y_MAPS_TO = 'EW'

# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65                    # 1g = 9806.65 mm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz

# Integration Filter Settings
HP_CUTOFF = 0.5                      # High-pass filter cutoff (Hz)
LP_CUTOFF = 5.0                      # Low-pass filter cutoff (Hz)

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

def get_mapped_acc(df, mapping):
    mapping = mapping.upper()
    if mapping == 'NS': return df['NS'].values
    elif mapping == 'SN': return -df['NS'].values
    elif mapping == 'EW': return df['EW'].values
    elif mapping == 'WE': return -df['EW'].values
    else: raise ValueError(f"Unknown axis mapping: {mapping}")

# ================= Main Script =================
def main():
    print("1. Loading Seismometer Data...")
    seis_df = pd.read_csv(SEISMIC_TXT_PATH, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
    
    acc_x = get_mapped_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM
    acc_y = get_mapped_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM

    print("2. Integrating Acceleration to Displacement...")
    disp_seis_x = double_integrate(acc_x, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    disp_seis_y = double_integrate(acc_y, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    
    print("3. Loading Visual Tracking Data...")
    vis_df = pd.read_csv(VISUAL_CSV_PATH)
    disp_vis_x = highpass_filter(vis_df['mean_x'].values - np.mean(vis_df['mean_x'].values), HP_CUTOFF, VISUAL_FS)
    disp_vis_y = highpass_filter(vis_df['mean_y'].values - np.mean(vis_df['mean_y'].values), HP_CUTOFF, VISUAL_FS)
    
    disp_vis_x = lowpass_filter(disp_vis_x, LP_CUTOFF, VISUAL_FS)
    disp_vis_y = lowpass_filter(disp_vis_y, LP_CUTOFF, VISUAL_FS)

    # Standardize data for Q-Q plot (Z-score normalization)
    def standardize(data):
        return (data - np.mean(data)) / np.std(data)

    seis_x_std = standardize(disp_seis_x)
    seis_y_std = standardize(disp_seis_y)
    vis_x_std = standardize(disp_vis_x)
    vis_y_std = standardize(disp_vis_y)

    print("4. Plotting Q-Q Plots...")
    plt.figure(figsize=(10, 12))

    # X-Axis Q-Q Plot
    plt.subplot(2, 1, 1)
    # Calculate quantiles for both datasets
    percs = np.linspace(0, 100, 1000)
    qn_seis_x = np.percentile(seis_x_std, percs)
    qn_vis_x = np.percentile(vis_x_std, percs)
    
    plt.scatter(qn_seis_x, qn_vis_x, color='blue', alpha=0.5, s=10, label='Data Points')
    # Identity line
    min_val = min(qn_seis_x.min(), qn_vis_x.min())
    max_val = max(qn_seis_x.max(), qn_vis_x.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Identity Line (Ideal)')
    
    plt.title(f'Q-Q Plot: X-Axis (Visual vs Seis {VISUAL_X_MAPS_TO})')
    plt.xlabel(f'Seismometer {VISUAL_X_MAPS_TO} Quantiles (Standardized)')
    plt.ylabel('Visual Tracking X Quantiles (Standardized)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    # Y-Axis Q-Q Plot
    plt.subplot(2, 1, 2)
    qn_seis_y = np.percentile(seis_y_std, percs)
    qn_vis_y = np.percentile(vis_y_std, percs)
    
    plt.scatter(qn_seis_y, qn_vis_y, color='red', alpha=0.5, s=10, label='Data Points')
    min_val = min(qn_seis_y.min(), qn_vis_y.min())
    max_val = max(qn_seis_y.max(), qn_vis_y.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linestyle='--', label='Identity Line (Ideal)')
    
    plt.title(f'Q-Q Plot: Y-Axis (Visual vs Seis {VISUAL_Y_MAPS_TO})')
    plt.xlabel(f'Seismometer {VISUAL_Y_MAPS_TO} Quantiles (Standardized)')
    plt.ylabel('Visual Tracking Y Quantiles (Standardized)')
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    out_img = "output/visual_vs_seismic_qq_plot.png"
    plt.savefig(out_img, dpi=300)
    print(f"Q-Q plot saved to {out_img}")
    plt.show()

if __name__ == "__main__":
    main()
