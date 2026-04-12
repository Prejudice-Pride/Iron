import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from compare_acc_disp import highpass_filter, lowpass_filter

# ================= 1. Configuration (Path Only) =================

# 视觉追踪结果 CSV 路径
VISUAL_CSV_PATH = r"best\sample3_0.5273_0.1487_16.67%_0.00%\sample3_output_cancel\sample3_corrected_surface_waveform.csv"

# 视觉采样参数
VISUAL_FS = 25.0         # 25 Hz
HP_CUTOFF = 0.5          # 高通滤波（去除慢速漂移/趋势）
LP_CUTOFF = 5.0          # 低通滤波（去除高频毛刺/追踪噪声）

# 输出目录
OUTPUT_DIR = "output/stats"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 2. Plotting Function =================

def plot_single_stat(data, axis_label, filename):
    """
    绘制单信号直方图 (Histogram) 与 核密度估计 (KDE)
    """
    plt.figure(figsize=(10, 6))
    
    # 颜色定义
    plot_color = "royalblue" if "X" in axis_label else "crimson"
    
    # 绘制直方图和 KDE
    # bw_adjust 控制平滑度，1.2-1.5 可以让曲线更温和
    sns.histplot(data, kde=True, stat="density", color=plot_color, 
                 label=f"Visual Displacement ({axis_label})", 
                 alpha=0.4, bins=50, kde_kws={'bw_adjust': 1.2})

    # 绘制正态分布拟合曲线（展示信号的平稳性）
    mu, std = norm.fit(data)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    plt.plot(x, norm.pdf(x, mu, std), color='black', linestyle='--', lw=1.5, 
             label=f"Gaussian Fit (μ={mu:.3f}, σ={std:.3f})")

    plt.title(f"Statistical Distribution Analysis: {axis_label}", fontsize=14)
    plt.xlabel("Displacement (cm)", fontsize=12)
    plt.ylabel("Probability Density", fontsize=12)
    plt.legend(loc="upper right", frameon=True)
    plt.grid(axis='y', linestyle=':', alpha=0.6)
    
    save_path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Successfully saved plot to: {save_path}")

# ================= 3. Main Logic =================

def main():
    print(f"Analyzing Visual Tracking Data: {VISUAL_CSV_PATH}")

    # 1. 加载视觉数据
    try:
        vis_df = pd.read_csv(VISUAL_CSV_PATH)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. 预处理：去均值 -> 高通 -> 低通
    def process_signal(raw_data):
        # 移除直流分量（将平衡位置归零）
        d = raw_data - np.mean(raw_data)
        # 滤波处理
        d = highpass_filter(d, HP_CUTOFF, VISUAL_FS)
        d = lowpass_filter(d, LP_CUTOFF, VISUAL_FS)
        return d

    disp_x = process_signal(vis_df['mean_x'].values)
    disp_y = process_signal(vis_df['mean_y'].values)

    # 3. 统计分析绘图
    print("Generating Histogram and KDE Analysis for the full video...")

    # 5. 绘图输出（合并为一张上下子图，便于论文展示）
    print("Generating combined Histogram and KDE Analysis for X (top) and Y (bottom)...")
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))

    def plot_on_ax(ax, data, axis_label):
        plot_color = "royalblue" if "X" in axis_label else "crimson"
        sns.histplot(data, kde=True, stat="density", color=plot_color,
                     label=f"Visual Displacement ({axis_label})", alpha=0.4, bins=50,
                     kde_kws={'bw_adjust': 1.2}, ax=ax)

        mu, std = norm.fit(data)
        xmin, xmax = ax.get_xlim()
        x = np.linspace(xmin, xmax, 200)
        ax.plot(x, norm.pdf(x, mu, std), color='black', linestyle='--', lw=1.5,
                label=f"Gaussian Fit (μ={mu:.3f}, σ={std:.3f})")

        ax.set_title(f"Statistical Distribution: {axis_label}")
        ax.set_xlabel("Displacement (mm)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')
        ax.grid(axis='y', linestyle=':', alpha=0.6)

    plot_on_ax(axes[0], disp_x, f"X-Axis")
    plot_on_ax(axes[1], disp_y, f"Y-Axis")

    plt.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, "visual_stats_xy_full.png")
    plt.savefig(outpath, dpi=300)
    print(f"Saved combined plot to: {outpath}")
    plt.show()

if __name__ == "__main__":
    main()