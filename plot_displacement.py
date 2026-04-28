import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# python plot_displacement.py output/cond1_output/cond1_tracked_surface_waveform.csv
# python plot_displacement.py output/cond2_output/cond2_tracked_surface_waveform.csv
# python plot_displacement.py output/cond3_output/cond3_tracked_surface_waveform.csv

def main():
    parser = argparse.ArgumentParser(description="Plot total displacement from tracking waveform CSV.")
    parser.add_argument("csv_file", type=str, help="Path to the waveform CSV file (e.g., cond1_tracked_surface_waveform.csv)")
    parser.add_argument("--out", type=str, default=None, help="Output plot image path. Defaults to same directory as CSV.")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.csv_file):
        print(f"Error: File not found: {args.csv_file}")
        return
    
    # Load data using numpy (assumes header is first row)
    print(f"Loading data from {args.csv_file}...")
    try:
        # 尝试读取带表头的 CSV 数据
        data = np.genfromtxt(args.csv_file, delimiter=',', names=True)
        time = data['time']
        mean_x = data['mean_x']
        mean_y = data['mean_y']
    except ValueError as e:
        print(f"Error parsing CSV format: {e}")
        return

    # Calculate total displacement: sqrt(mean_x^2 + mean_y^2)
    displacement = np.sqrt(mean_x**2 + mean_y**2)
    
    # Output path
    if args.out:
        out_path = args.out
    else:
        base_name = os.path.splitext(args.csv_file)[0]
        out_path = f"{base_name}_displacement_plot.png"

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time, displacement, label=r'$\sqrt{X^2 + Y^2}$', color='purple', linewidth=1.5)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Total Displacement (cm)', fontsize=12)
    plt.title('Surface Tracking Total Displacement Over Time', fontsize=14)
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.legend(fontsize=12)
    plt.tight_layout()
    
    # Save and show
    plt.savefig(out_path, dpi=300)
    print(f"Displacement plot saved successfully to: {out_path}")
    # plt.show() # Uncomment to display window if running interactively

if __name__ == "__main__":
    main()
