# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# === 全局科研绘图风格（优化版）===
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["SimHei", "Arial"],  # 中文黑体 + 英文 Arial
    "font.size": 11,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix"
})

def generate_x_ground_truth(t_end=23.0, sample_rate=100):
    """
    X方向 Ground Truth：无运动（恒为0）
    """
    t = np.linspace(0, t_end, int(t_end * sample_rate))
    x = np.zeros_like(t)
    return t, x


def plot_waveforms_x(csv_files, labels=None, save_path=None):
    """
    绘制多个CSV的X方向波形，并与 Ground Truth (0) 对比
    """
    colors = [
    '#2E6F95',  # 蓝（稍亮）
    '#B03A2E',  # 红
    '#239B56',  # 绿
    '#6C3483',  # 紫
    '#AF601A'   # 棕
    ]
    gt_color = '#5A5A5A'

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)

    # === 1. Ground Truth（X=0）===
    t_gt, x_gt = generate_x_ground_truth()
    ax.plot(t_gt, x_gt,
            label='Ground Truth (X=0)',
            color=gt_color,
            linewidth=1.5,
            linestyle='-',
            zorder=1)

    # === 2. 绘制实验数据 ===
    for i, file_path in enumerate(csv_files):
        if not os.path.exists(file_path):
            continue

        df = pd.read_csv(file_path)

        time_col = 'time' if 'time' in df.columns else df.columns[0]

        # === 自动选择 X 列（优先 mean_x）===
        if 'mean_x' in df.columns:
            x_col = 'mean_x'
        elif 'median_x' in df.columns:
            x_col = 'median_x'
        else:
            # fallback（防止格式不一致）
            x_col = df.columns[3]

        label = labels[i] if labels and i < len(labels) else os.path.basename(file_path)

        ax.plot(df[time_col],
                df[x_col],
                label=label,
                color=colors[i % len(colors)],
                linewidth=1.8,
                zorder=2)

    # === 3. 标题 & 坐标 ===
    ax.set_title('X轴位移波形比较',
                 fontsize=14,
                 fontweight='bold',
                 fontfamily='SimHei',
                 pad=12)

    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Displacement (cm)', fontsize=12)

    # === 4. 全框（论文推荐）===
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#444444')

    # === 5. 网格 ===
    ax.grid(True, linestyle='--', alpha=0.4)

    # === 6. 基准线 ===
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)

    # === 7. 图例 ===
    ax.legend(frameon=True,
              facecolor='white',
              framealpha=0.9,
              loc='upper right',
              fontsize=9)

    plt.tight_layout()

    # === 8. 保存 ===
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")

    plt.show()


if __name__ == "__main__":

    files_to_compare = [
        "output/cond1_output/cond1_tracked_surface_waveform.csv",
        "output/cond3_p1_output/cond3_tracked_surface_waveform.csv",
        "output/cond3_p2_output/cond3_tracked_surface_waveform.csv",
        "output/cond3_p3_output/cond3_tracked_surface_waveform.csv",
    ]

    legend_labels = [
        "Tracker Result (实验一)",
        "Tracker Result (实验三 P1)",
        "Tracker Result (实验三 P2)",
        "Tracker Result (实验三 P3)",
    ]

    output_path = "output/x_waveform_comparison.png"

    plot_waveforms_x(files_to_compare,
                     labels=legend_labels,
                     save_path=output_path)