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

VIDEO_AXIS_CONFIGS = {
    "sample": {"x_map": "NS", "y_map": "WE"},
    "qample": {"x_map": "NS", "y_map": "WE"},
    "pample2": {"x_map": "NS", "y_map": "WE"},
    "qample2": {"x_map": "SN", "y_map": "EW"},
}

parser = argparse.ArgumentParser(description="Ablation study: Evaluate corrected vs raw tracking.")
parser.add_argument("--seismic_txt", type=str, default=r"txt_home\sample3_206a.txt")
parser.add_argument("--corrected_csv", type=str, default=r"output\sample3_output_cancel\sample3_corrected_surface_waveform.csv")
parser.add_argument("--raw_csv", type=str, default=r"output\sample3_output_cancel\sample3_raw_surface_waveform.csv")
parser.add_argument("--out_img", type=str, default="output/ablation_visual_vs_seismic.png")
parser.add_argument("--visual_x_maps_to", type=str, default=None)
parser.add_argument("--visual_y_maps_to", type=str, default=None)
args = parser.parse_args()

def get_video_name_from_path(path):
    return os.path.splitext(os.path.basename(path))[0]

def auto_detect_axis_mapping(csv_path, x_override=None, y_override=None):
    video_name = get_video_name_from_path(csv_path)
    detected_x, detected_y = "NS", "WE"
    for video_key, mapping in VIDEO_AXIS_CONFIGS.items():
        if video_key.lower() in video_name.lower():
            detected_x = mapping["x_map"]
            detected_y = mapping["y_map"]
            break
    return x_override or detected_x, y_override or detected_y

VISUAL_X_MAPS_TO, VISUAL_Y_MAPS_TO = auto_detect_axis_mapping(
    args.corrected_csv, args.visual_x_maps_to, args.visual_y_maps_to
)

SEISMIC_FS = 100.0
SEISMIC_DT = 1.0 / SEISMIC_FS
G_TO_MM = 9806.65
VISUAL_FS = 25.0
HP_CUTOFF = 0.5
LP_CUTOFF = 5.0
SHIFT_MIN_SEC, SHIFT_MAX_SEC = None, None
EVAL_DURATION_SEC = 6.0
MIN_EVAL_DURATION_SEC = 3.5

# ================= Helper Functions =================
def highpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, cutoff / nyq, btype='high', analog=False)
    return filtfilt(b, a, data)

def lowpass_filter(data, cutoff, fs, order=4):
    nyq = 0.5 * fs
    nc = cutoff / nyq
    if nc >= 1.0: nc = 0.99
    b, a = butter(order, nc, btype='low', analog=False)
    return filtfilt(b, a, data)

def double_integrate(acc, dt, cutoff, fs):
    acc = acc - np.mean(acc)
    acc_filt = highpass_filter(acc, cutoff, fs)
    vel_filt = highpass_filter(cumulative_trapezoid(acc_filt, dx=dt, initial=0), cutoff, fs)
    return highpass_filter(cumulative_trapezoid(vel_filt, dx=dt, initial=0), cutoff, fs)

def align_and_evaluate(ref_signal, test_signal, forced_shift=None):
    ref_norm = (ref_signal - np.mean(ref_signal)) / np.std(ref_signal)
    test_norm = (test_signal - np.mean(test_signal)) / (np.std(test_signal) + 1e-9)
    
    if forced_shift is None:
        cc = correlate(ref_norm, test_norm, mode='full')
        lags = np.arange(-(len(test_norm) - 1), len(ref_norm))
        shift = lags[np.argmax(cc)]
    else:
        shift = forced_shift

    if shift > 0:
        ref_aligned, test_aligned = ref_norm[shift:], test_norm[:-shift]
    elif shift < 0:
        ref_aligned, test_aligned = ref_norm[:shift], test_norm[-shift:]
    else:
        ref_aligned, test_aligned = ref_norm, test_norm
        
    eval_samples = int(EVAL_DURATION_SEC * SEISMIC_FS)
    min_len = min(len(ref_aligned), len(test_aligned), eval_samples)
    ref_aligned, test_aligned = ref_aligned[:min_len], test_aligned[:min_len]
    
    r = np.corrcoef(ref_aligned, test_aligned)[0, 1]
    rmse = np.sqrt(np.mean((ref_aligned - test_aligned)**2))
    mae = np.mean(np.abs(ref_aligned - test_aligned))
    wass_dist = wasserstein_distance(ref_aligned, test_aligned)
    return shift, r, mae, rmse, wass_dist, ref_aligned, test_aligned

def compute_fft(signal_data, fs):
    window = np.hanning(len(signal_data))
    yf = rfft(signal_data * window)
    xf = rfftfreq(len(signal_data), 1 / fs)
    amp = np.abs(yf)
    amp = amp / (np.max(amp) + 1e-9)
    peak_f = xf[np.argmax(amp[1:]) + 1]
    return xf, amp, peak_f

# ================= Main Script =================
def process_visual_df(df_path, time_seis):
    df = pd.read_csv(df_path)
    time_vis, v_x, v_y = df['time'].values, df['mean_x'].values, df['mean_y'].values
    v_x = lowpass_filter(highpass_filter(v_x - np.mean(v_x), HP_CUTOFF, VISUAL_FS), LP_CUTOFF, VISUAL_FS)
    v_y = lowpass_filter(highpass_filter(v_y - np.mean(v_y), HP_CUTOFF, VISUAL_FS), LP_CUTOFF, VISUAL_FS)
    return np.interp(time_seis, time_vis, v_x), np.interp(time_seis, time_vis, v_y)

def main():
    seis_df = pd.read_csv(args.seismic_txt, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
    def get_acc(df, m):
        m = m.upper()
        return df['NS'].values if m=='NS' else -df['NS'].values if m=='SN' else df['EW'].values if m=='EW' else -df['EW'].values
    
    acc_x, acc_y = get_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM, get_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM
    seis_x = double_integrate(acc_x, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    seis_y = double_integrate(acc_y, SEISMIC_DT, HP_CUTOFF, SEISMIC_FS)
    time_seis = np.arange(len(seis_y)) * SEISMIC_DT

    corr_x, corr_y = process_visual_df(args.corrected_csv, time_seis)
    raw_x, raw_y = process_visual_df(args.raw_csv, time_seis)

    # Align corrected to ground truth
    shift_x, r_cx, mae_cx, rmse_cx, w_cx, ref_x, test_cx = align_and_evaluate(seis_x, corr_x)
    shift_y, r_cy, mae_cy, rmse_cy, w_cy, ref_y, test_cy = align_and_evaluate(seis_y, corr_y, forced_shift=shift_x)

    # Force RAW to use the EXACT SAME SHIFT as corrected relative to ground truth
    _, r_rx, mae_rx, rmse_rx, w_rx, _, test_rx = align_and_evaluate(seis_x, raw_x, forced_shift=shift_x)
    _, r_ry, mae_ry, rmse_ry, w_ry, _, test_ry = align_and_evaluate(seis_y, raw_y, forced_shift=shift_x)

    # FFTs
    fx, amp_ref_x, pf_ref_x = compute_fft(ref_x, SEISMIC_FS)
    _, amp_cx, pf_cx = compute_fft(test_cx, SEISMIC_FS)
    _, amp_rx, pf_rx = compute_fft(test_rx, SEISMIC_FS)
    
    fy, amp_ref_y, pf_ref_y = compute_fft(ref_y, SEISMIC_FS)
    _, amp_cy, pf_cy = compute_fft(test_cy, SEISMIC_FS)
    _, amp_ry, pf_ry = compute_fft(test_ry, SEISMIC_FS)

    err_cx = abs(pf_ref_x - pf_cx) / pf_ref_x * 100 if pf_ref_x != 0 else 0
    err_rx = abs(pf_ref_x - pf_rx) / pf_ref_x * 100 if pf_ref_x != 0 else 0
    err_cy = abs(pf_ref_y - pf_cy) / pf_ref_y * 100 if pf_ref_y != 0 else 0
    err_ry = abs(pf_ref_y - pf_ry) / pf_ref_y * 100 if pf_ref_y != 0 else 0

    print("\n" + "="*50)
    print("=== ABLATION STUDY RESULTS: CORRECTED vs RAW ===")
    print(f"X-Axis (VS Seismometer {VISUAL_X_MAPS_TO})")
    print(f"  [Corrected (Ours)]")
    print(f"    - Pearson R:       {r_cx:.4f}")
    print(f"    - RMSE:            {rmse_cx:.4f}")
    print(f"    - MAE:             {mae_cx:.4f}")
    print(f"    - Frequency Error: {err_cx:.2f}% (Peak: {pf_cx:.2f}Hz)")
    print(f"    - Wasserstein:     {w_cx:.4f}")
    print(f"  [Raw / Baseline (No Cancel)]")
    print(f"    - Pearson R:       {r_rx:.4f}")
    print(f"    - RMSE:            {rmse_rx:.4f}")
    print(f"    - MAE:             {mae_rx:.4f}")
    print(f"    - Frequency Error: {err_rx:.2f}% (Peak: {pf_rx:.2f}Hz)")
    print(f"    - Wasserstein:     {w_rx:.4f}")
    print("-" * 50)
    print(f"Y-Axis (VS Seismometer {VISUAL_Y_MAPS_TO})")
    print(f"  [Corrected (Ours)]")
    print(f"    - Pearson R:       {r_cy:.4f}")
    print(f"    - RMSE:            {rmse_cy:.4f}")
    print(f"    - MAE:             {mae_cy:.4f}")
    print(f"    - Frequency Error: {err_cy:.2f}% (Peak: {pf_cy:.2f}Hz)")
    print(f"    - Wasserstein:     {w_cy:.4f}")
    print(f"  [Raw / Baseline (No Cancel)]")
    print(f"    - Pearson R:       {r_ry:.4f}")
    print(f"    - RMSE:            {rmse_ry:.4f}")
    print(f"    - MAE:             {mae_ry:.4f}")
    print(f"    - Frequency Error: {err_ry:.2f}% (Peak: {pf_ry:.2f}Hz)")
    print(f"    - Wasserstein:     {w_ry:.4f}")
    print("="*50 + "\n")
    
    # Plotting
    plt.figure(figsize=(16, 12))
    t_plot_x = np.arange(len(ref_x)) * SEISMIC_DT
    
    plt.subplot(4, 1, 1)
    plt.title(f"X-Axis Time Domain | Corrected r={r_cx:.4f} vs Raw r={r_rx:.4f}")
    plt.plot(t_plot_x, ref_x, 'k-', alpha=0.7, label='Seismometer (GT)')
    plt.plot(t_plot_x, test_rx, 'gray', alpha=0.8, linestyle=':', label='Raw (Before Cancel)')
    plt.plot(t_plot_x, test_cx, 'b--', alpha=0.8, label='Corrected (Ours)')
    plt.legend(); plt.grid(True, ls=':')

    plt.subplot(4, 1, 2)
    plt.title(f"Y-Axis Time Domain | Corrected r={r_cy:.4f} vs Raw r={r_ry:.4f}")
    plt.plot(t_plot_x, ref_y, 'k-', alpha=0.7, label='Seismometer (GT)')
    plt.plot(t_plot_x, test_ry, 'orange', alpha=0.8, linestyle=':', label='Raw (Before Cancel)')
    plt.plot(t_plot_x, test_cy, 'r--', alpha=0.8, label='Corrected (Ours)')
    plt.legend(); plt.grid(True, ls=':')

    plt.subplot(4, 1, 3)
    plt.title(f"X-Axis Frequency | Err: Corr={err_cx:.1f}%, Raw={err_rx:.1f}%")
    plt.plot(fx, amp_ref_x, 'k-', alpha=0.7, label=f'GT ({pf_ref_x:.2f}Hz)')
    plt.plot(fx, amp_rx, 'gray', alpha=0.8, label=f'Raw ({pf_rx:.2f}Hz)')
    plt.plot(fx, amp_cx, 'b-', alpha=0.8, label=f'Corrected ({pf_cx:.2f}Hz)')
    plt.xlim(0, 10); plt.legend(); plt.grid(True, ls=':')

    plt.subplot(4, 1, 4)
    plt.title(f"Y-Axis Frequency | Err: Corr={err_cy:.1f}%, Raw={err_ry:.1f}%")
    plt.plot(fy, amp_ref_y, 'k-', alpha=0.7, label=f'GT ({pf_ref_y:.2f}Hz)')
    plt.plot(fy, amp_ry, 'orange', alpha=0.8, label=f'Raw ({pf_ry:.2f}Hz)')
    plt.plot(fy, amp_cy, 'r-', alpha=0.8, label=f'Corrected ({pf_cy:.2f}Hz)')
    plt.xlim(0, 10); plt.legend(); plt.grid(True, ls=':')

    plt.tight_layout()
    plt.savefig(args.out_img, dpi=300)
    print(f"Plot saved to {args.out_img}")
    plt.show()

if __name__ == "__main__":
    main()
