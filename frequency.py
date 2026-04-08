import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import detrend
from scipy.fft import rfft, rfftfreq

# ================= Configuration =================
# 你可以根据需要修改这部分路径和映射
SEISMIC_TXT_PATH = r"txt_home\sample_p1_910.txt"
VISUAL_CSV_PATH = r"output\sample_output_cancel\sample_corrected_surface_waveform.csv"

# Axis Mapping (Camera's perspective vs Seismometer)
# Default
VISUAL_X_MAPS_TO = 'NS'
VISUAL_Y_MAPS_TO = 'EW'

# sample.mp4
VISUAL_X_MAPS_TO = 'NS'
VISUAL_Y_MAPS_TO = 'WE'

# # qample.mp4
# VISUAL_X_MAPS_TO = 'NS'
# VISUAL_Y_MAPS_TO = 'WE'

# # pample2.mp4
# VISUAL_X_MAPS_TO = 'NS'
# VISUAL_Y_MAPS_TO = 'WE'

# # qample2.mp4
# VISUAL_X_MAPS_TO = 'SN'
# VISUAL_Y_MAPS_TO = 'EW'

# Seismometer specs
SEISMIC_FS = 100.0                   # 100 Hz
G_TO_MM = 980.665                    # 1g = 980.665 cm/s^2

# Visual tracking specs
VISUAL_FS = 25.0                     # 25 Hz

# Frequency integration cutoff (to avoid dividing by near-zero frequencies which blows up the spectrum)
HP_CUTOFF = 0.5                      # High-pass equivalent (Hz)

# ================= Helper Functions =================
def get_mapped_acc(df, mapping):
    """根据映射关系提取地震仪的加速度列并处理符号"""
    mapping = mapping.upper()
    if mapping == 'NS': return df['NS'].values
    elif mapping == 'SN': return -df['NS'].values
    elif mapping == 'EW': return df['EW'].values
    elif mapping == 'WE': return -df['EW'].values
    else: raise ValueError(f"Unknown axis mapping: {mapping}. Must be one of NS, SN, EW, WE.")

def compute_disp_fft_from_acc(acc_data, fs, hp_cutoff):
    """计算加速度数据的FFT，并在频域中将其积分两次得到位移频谱"""
    n = len(acc_data)
    # 去直流
    acc_data = detrend(acc_data)
    # 汉宁窗减少频谱泄露
    window = np.hanning(n)
    acc_windowed = acc_data * window
    
    # FFT
    yf_acc = rfft(acc_windowed)
    xf = rfftfreq(n, 1 / fs)
    
    # 频域积分： 位移(w) = 加速度(w) / -(w^2) ，其中 w = 2*pi*f
    yf_disp = np.zeros_like(yf_acc, dtype=complex)
    for i, f in enumerate(xf):
        if f >= hp_cutoff:  # 跳过低频分量，避免除以0或造成低频极度放大
            omega = 2.0 * np.pi * f
            yf_disp[i] = yf_acc[i] / (-(omega ** 2))
            
    # 计算幅值并归一化以便于对比
    amp = np.abs(yf_disp)
    if np.max(amp) > 0:
        amp = amp / np.max(amp)
    return xf, amp

def compute_fft(signal_data, fs):
    """计算直接测得的位移数据的FFT"""
    n = len(signal_data)
    signal_data = detrend(signal_data)
    window = np.hanning(n)
    signal_windowed = signal_data * window
    
    yf = rfft(signal_windowed)
    xf = rfftfreq(n, 1 / fs)
    
    # 计算幅值并归一化
    amp = np.abs(yf)
    # 跳过直流量[0]进行归一化
    if np.max(amp[1:]) > 0:
        amp = amp / np.max(amp[1:])
    return xf, amp

def evaluate_spectral_similarity(freq_ref, amp_ref, freq_test, amp_test, max_f=12.5):
    """
    量化评估频谱相似度
    """
    # 1. 裁剪频谱至指定的评估区间 (默认 12.5Hz)
    mask_ref = freq_ref <= max_f
    mask_test = freq_test <= max_f
    
    f_ref_cut = freq_ref[mask_ref]
    a_ref_cut = amp_ref[mask_ref]
    
    f_test_cut = freq_test[mask_test]
    a_test_cut = amp_test[mask_test]

    # 2. 主频误差
    peak_f_ref = f_ref_cut[np.argmax(a_ref_cut)]
    peak_f_test = f_test_cut[np.argmax(a_test_cut)]
    f_err = abs(peak_f_ref - peak_f_test)
    
    # 3. 对齐频率轴以便计算统计指标
    # 将测试频谱插值到参考频谱的格点上
    amp_test_interp = np.interp(f_ref_cut, f_test_cut, a_test_cut)
    
    # 4. 皮尔逊相关度 (Spectral Correlation)
    correlation = np.corrcoef(a_ref_cut, amp_test_interp)[0, 1]
    
    # 5. RMSE
    rmse = np.sqrt(np.mean((a_ref_cut - amp_test_interp)**2))
    
    return {
        "Ref Peak Freq": peak_f_ref,
        "Test Peak Freq": peak_f_test,
        "Freq Error (Hz)": f_err,
        "Spectral Correlation": correlation,
        "Spectral RMSE": rmse
    }

# ================= Main Script =================
def main():
    print("1. Loading Seismometer Data (100Hz)...")
    try:
        seis_df = pd.read_csv(SEISMIC_TXT_PATH, sep=r'\s+', skiprows=6, names=['Date', 'Time', 'EW', 'NS', 'UD'])
    except Exception as e:
        print(f"Error loading TXT: {e}")
        return

    # 提取加速度并转换单位 g -> mm/s^2
    acc_x = get_mapped_acc(seis_df, VISUAL_X_MAPS_TO) * G_TO_MM
    acc_y = get_mapped_acc(seis_df, VISUAL_Y_MAPS_TO) * G_TO_MM

    print("2. Loading Visual Tracking Data (25Hz)...")
    try:
        vis_df = pd.read_csv(VISUAL_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return
        
    disp_vis_x = vis_df['mean_x'].values
    disp_vis_y = vis_df['mean_y'].values

    print("3. Computing Frequency Spectra (FFT Integration for acceleration)...")
    # 计算 X 轴
    freq_seis_x, amp_seis_x = compute_disp_fft_from_acc(acc_x, SEISMIC_FS, HP_CUTOFF)
    freq_vis_x, amp_vis_x = compute_fft(disp_vis_x, VISUAL_FS)
    
    # 计算 Y 轴
    freq_seis_y, amp_seis_y = compute_disp_fft_from_acc(acc_y, SEISMIC_FS, HP_CUTOFF)
    freq_vis_y, amp_vis_y = compute_fft(disp_vis_y, VISUAL_FS)

    # 4. 量化评估
    print("\n" + "="*50)
    print("--- 📊 SPECTRAL SIMILARITY EVALUATION ---")
    
    metrics_x = evaluate_spectral_similarity(freq_seis_x, amp_seis_x, freq_vis_x, amp_vis_x)
    print(f"X-Axis (Visual X vs Seismometer {VISUAL_X_MAPS_TO}):")
    print(f"  - Seismometer Peak Freq:   {metrics_x['Ref Peak Freq']:.3f} Hz")
    print(f"  - Visual Output Peak Freq: {metrics_x['Test Peak Freq']:.3f} Hz")
    print(f"  - Peak Frequency Error:    {metrics_x['Freq Error (Hz)']:.3f} Hz")
    print(f"  - Spectral Correlation:    {metrics_x['Spectral Correlation']:.4f}")
    print(f"  - Spectral RMSE:           {metrics_x['Spectral RMSE']:.4f}")
    
    metrics_y = evaluate_spectral_similarity(freq_seis_y, amp_seis_y, freq_vis_y, amp_vis_y)
    print(f"\nY-Axis (Visual Y vs Seismometer {VISUAL_Y_MAPS_TO}):")
    print(f"  - Seismometer Peak Freq:   {metrics_y['Ref Peak Freq']:.3f} Hz")
    print(f"  - Visual Output Peak Freq: {metrics_y['Test Peak Freq']:.3f} Hz")
    print(f"  - Peak Frequency Error:    {metrics_y['Freq Error (Hz)']:.3f} Hz")
    print(f"  - Spectral Correlation:    {metrics_y['Spectral Correlation']:.4f}")
    print(f"  - Spectral RMSE:           {metrics_y['Spectral RMSE']:.4f}")
    print("="*50 + "\n")

    # 5. Plotting comparisons...
    plt.figure(figsize=(14, 10))
    
    # 限制X轴最大频率，因为25Hz采样率奈奎斯特频率为12.5Hz
    max_plot_freq = 12.5 
    
    # Plot 1: X-Axis
    plt.subplot(2, 1, 1)
    plt.title(f"X-Axis Displacement Frequency Spectrum (Visual vs Seismometer {VISUAL_X_MAPS_TO})")
    plt.plot(freq_seis_x, amp_seis_x, label=f"Seismometer {VISUAL_X_MAPS_TO} (FFT Integration)", color='black', alpha=0.7, linewidth=1.5)
    plt.plot(freq_vis_x, amp_vis_x, label="Visual Output X (Direct FFT)", color='blue', alpha=0.8, linewidth=1.5)
    plt.xlim(0, max_plot_freq)
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    # Plot 2: Y-Axis
    plt.subplot(2, 1, 2)
    plt.title(f"Y-Axis Displacement Frequency Spectrum (Visual vs Seismometer {VISUAL_Y_MAPS_TO})")
    plt.plot(freq_seis_y, amp_seis_y, label=f"Seismometer {VISUAL_Y_MAPS_TO} (FFT Integration)", color='black', alpha=0.7, linewidth=1.5)
    plt.plot(freq_vis_y, amp_vis_y, label="Visual Output Y (Direct FFT)", color='red', alpha=0.8, linewidth=1.5)
    plt.xlim(0, max_plot_freq)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized Amplitude")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()