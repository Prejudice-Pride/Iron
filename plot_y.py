# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置全局科研绘图风格，并使用常见中文字体避免中文方块
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "sans-serif"],
    "font.size": 11,
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix"
})

def generate_ground_truth(t_start=2.0, freq=1.0, normal_cycles=18, sample_rate=100, end_time=23.0):
    """
    生成符合物理振动台限制的Ground Truth波形
    :param t_start: 起始震动时间 (s)
    :param freq: 正常运动的频率 (Hz)
    :param normal_cycles: 正常周期数（不含首个受限周期，默认 18，合计 19 个周期）
    :param sample_rate: 采样率
    :param end_time: 总时长(s)
    """
    t = np.linspace(0, end_time, int(end_time * sample_rate))
    y = np.zeros_like(t)
    
    T = 1.0 / freq  # 单个周期时长
    
    idx = 0
    for idx_t, current_t in enumerate(t):
        if current_t < t_start:
            y[idx_t] = 0.0
            
        # 第一个周期（受到物理限制）
        elif t_start <= current_t < t_start + T:
            phase = 2 * np.pi * freq * (current_t - t_start)
            y[idx_t] = 0.2 * np.sin(phase) if phase < np.pi else 1.0 * np.sin(phase)
                
        # 正常周期
        elif t_start + T <= current_t < t_start + T + normal_cycles * T:
            phase = 2 * np.pi * freq * (current_t - t_start - T)
            y[idx_t] = 2.0 * np.sin(phase)      # 正常幅值 2.0cm
            
        # 结束时的额外周期（反方向重复第一个周期）
        elif t_start + T + normal_cycles * T <= current_t < t_start + 2*T + normal_cycles * T:
            t_extra = current_t - (t_start + T + normal_cycles * T)
            phase = 2 * np.pi * freq * t_extra
            y[idx_t] = 1.0 * np.sin(phase) if phase < np.pi else 0.2 * np.sin(phase)
        
        # 之后时间停下
        else:
            y[idx_t] = 0.0
            
    return t, y

def plot_waveforms(csv_files, labels=None, save_path=None):
    """
    绘制多个CSV波形与Ground Truth对比
    :param csv_files: waveform.csv 路径列表
    :param labels: 图例标签列表
    :param save_path: 保存图像的文件路径
    """
    # 使用更加沉稳、专业的科研配色 (深蓝, 砖红, 深绿, 靛紫, 棕褐)
    colors = [
        '#2E6F95',  # 蓝（稍亮）
        '#B03A2E',  # 红
        '#239B56',  # 绿
        '#6C3483',  # 紫
        '#AF601A'   # 棕
    ]
    # 基准线颜色改为更显眼但仍低对比的深灰色实线
    gt_color = '#5A5A5A'
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    
    # 1. 绘制 Ground Truth (换成实线，zorder设低一些作为底衬)
    t_gt, y_gt = generate_ground_truth(t_start=1.5, freq=1.0, normal_cycles=18, end_time=23.0)
    ax.plot(t_gt, y_gt, label='Ground Truth', color=gt_color, linewidth=1.5, linestyle='-', zorder=1)
    
    # 2. 绘制数据线
    for i, file_path in enumerate(csv_files):
        if not os.path.exists(file_path): continue
        df = pd.read_csv(file_path)
        time_col = 'time' if 'time' in df.columns else df.columns[0]
        y_col = 'y' if 'y' in df.columns else df.columns[2]
        
        label = labels[i] if labels and i < len(labels) else os.path.basename(file_path)
        # 增加 zorder 确保数据线在虚线之上，使用 linewidth=1.8 增加质感
        ax.plot(df[time_col], df[y_col], label=label, color=colors[i % len(colors)], linewidth=1.8, zorder=2)

    # 3. 细节美化
    ax.set_title('Y轴位移波形比较',
                fontsize=14,
                fontweight='bold',
                fontfamily='SimHei',
                pad=12)
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Displacement (cm)', fontsize=12)
    
    # === 全框（优化版，不厚重） ===
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_color('#444444')  # 深灰比纯黑更高级
    
    # 将网格设为非常淡的虚线
    ax.grid(True, linestyle='--', alpha=0.4, which='both')
    
    # 强制设置水平基准线 y=0
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.3)
    
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, loc='upper right', fontsize=9)
    
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight') # 300dpi 满足绝大部分期刊要求
        print(f"Fine quality plot saved to: {save_path}")

    plt.show()

if __name__ == "__main__":
    # 替换为您需要比对的波形文件路径
    files_to_compare = [
        "output/cond1_output/cond1_tracked_surface_waveform.csv",
        # "output/cond2_output/waveform.csv" # 增加其他条件
        "output/cond3_p1_output/cond3_tracked_surface_waveform.csv",
        "output/cond3_p2_output/cond3_tracked_surface_waveform.csv",
        "output/cond3_p3_output/cond3_tracked_surface_waveform.csv",
    ]
    
    # 给定好读的图例名
    legend_labels = [
        "Tracker Result (实验一)",
        "Tracker Result (实验三 P1)",
        "Tracker Result (实验三 P2)",
        "Tracker Result (实验三 P3)",
    ]
    
    output_path = "output/discussion_waveform_comparison.png"
    plot_waveforms(files_to_compare, labels=legend_labels, save_path=output_path)