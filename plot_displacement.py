import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# python plot_displacement.py output/cond1_output/cond1_tracked_surface_waveform.csv --method detrend
# python plot_displacement.py output/cond2_output/cond2_tracked_surface_waveform.csv --method detrend
# python plot_displacement.py output/cond3_output/cond3_tracked_surface_waveform.csv --method detrend

def main():
    parser = argparse.ArgumentParser(description="Plot total displacement from tracking waveform CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the waveform CSV file")
    parser.add_argument("--out", type=str, default=None, help="Output plot image path")
    
    # 新增参数
    parser.add_argument("--method", type=str, default="detrend",
                        choices=["detrend", "highpass"],
                        help="Baseline correction method")
    parser.add_argument("--cutoff", type=float, default=0.5,
                        help="High-pass cutoff frequency (Hz), only for highpass")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return
    
    print(f"Loading data from {args.csv_file}...")
    try:
        data = np.genfromtxt(args.csv_file, delimiter=',', names=True)
        time = data['time']
        mean_x = data['mean_x']
        mean_y = data['mean_y']
    except ValueError as e:
        print(f"Error parsing CSV format: {e}")
        return

    # 原始位移
    displacement = np.sqrt(mean_x**2 + mean_y**2)

    # ====== 去基线漂移 ======
    if args.method == "detrend":
        displacement_corrected = signal.detrend(displacement)

    elif args.method == "highpass":
        dt = np.mean(np.diff(time))
        fs = 1.0 / dt  # sampling frequency
        
        cutoff = args.cutoff
        b, a = signal.butter(4, cutoff / (0.5 * fs), btype='high')
        displacement_corrected = signal.filtfilt(b, a, displacement)

    # ====== 输出路径 ======
    if args.out:
        out_path = args.out
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        out_path = f"{base_name}_displacement_plot.png"

    # ====== 画图 ======
    plt.figure(figsize=(12, 6))

    # 原始信号（淡色）
    plt.plot(time, displacement, label='Original', alpha=0.4)

    # 去漂移信号
    plt.plot(time, displacement_corrected,
             label='Corrected', color='purple', linewidth=1.5)

    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Total Displacement (cm)', fontsize=12)
    plt.title('Displacement (Baseline Drift Removed)', fontsize=14)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()

    plt.savefig(out_path, dpi=300)
    print(f"Displacement plot saved to: {out_path}")

if __name__ == "__main__":
    main()