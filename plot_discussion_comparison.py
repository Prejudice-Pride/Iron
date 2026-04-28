import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend, butter, filtfilt, correlate
from scipy.interpolate import PchipInterpolator

# python plot_discussion_comparison.py output/cond1_output/cond1_tracked_surface_waveform.csv output/cond2_output/cond2_tracked_surface_waveform.csv output/cond3_output/cond3_tracked_surface_waveform.csv --labels "Ideal Condition" "Calibration Z-Jump" "Tracking Surface Deformation"
# python plot_discussion_comparison.py output/cond1_output/cond1_tracked_surface_waveform.csv output/cond2_output/cond2_tracked_surface_waveform.csv --labels "Ideal Condition" "Calibration Z-Jump"
# python plot_discussion_comparison.py output/cond1_output/cond1_tracked_surface_waveform.csv  --labels "Ideal Condition"

def butter_highpass_filter(data, cutoff, fs, order=4):
    """零相位高通滤波器，用于极大地消除追踪过程中的基线漂移"""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def extract_principal_displacement(mean_x, mean_y):
    """
    使用PCA主成分投影提取总位移，保留波形的振荡正负号
    （比 sqrt(x^2+y^2) 更符合物理实际，防止负半周被翻转重叠）
    """
    # 去均值
    x_c = mean_x - np.mean(mean_x)
    y_c = mean_y - np.mean(mean_y)
    
    # 构造协方差矩阵并求特征值/特征向量
    coords = np.vstack([x_c, y_c])
    cov_mat = np.cov(coords)
    evals, evecs = np.linalg.eig(cov_mat)
    
    # 找到最大方差对应的特征向量（即主运动方向）
    principal_evec = evecs[:, np.argmax(evals)]
    
    # 投影到主方向
    disp = x_c * principal_evec[0] + y_c * principal_evec[1]
    return disp

def extract_total_displacement(mean_x, mean_y):
    """
    直接计算总位移幅值（无方向信息）
    disp = sqrt(x^2 + y^2)

    注意：先去均值，避免静态偏移影响幅值
    """
    x_c = mean_x - np.mean(mean_x)
    y_c = mean_y - np.mean(mean_y)
    
    disp = np.sqrt(x_c**2 + y_c**2)
    return disp

def generate_ground_truth(fs=60, freq=1.0, amp=2.0, cycles=20):
    """
    生成贴合你振动台特性的绝对物理真值 (Ground Truth)
    启动周期: 正向 1/8 幅值 (amp/8), 负向 1/2 幅值 (amp/2)
    中间周期: 标准 2cm 振幅正弦波
    结束额外周期: 逆向重复, 正向 1/2 幅值, 负向 1/8 幅值
    """
    # 1. 启动周期 (第 1 秒)
    t_start = np.arange(0, 1.0, 1/fs)
    # 使用 Pchip 保证插值平滑且不越界
    y_start = PchipInterpolator([0, 0.25, 0.75, 1.0], [0, amp/8, -amp/2, 0])(t_start)
    
    # 2. 正常周期 (持续 cycles 秒)
    t_mid = np.arange(1.0, 1.0 + cycles, 1/fs)
    y_mid = amp * np.sin(2 * np.pi * freq * (t_mid - 1.0))
    
    # 3. 结束额外周期 (第 21.0 到 22.0 秒)
    t_end = np.arange(1.0 + cycles, 2.0 + cycles, 1/fs)
    y_end = PchipInterpolator([t_end[0], t_end[0]+0.25, t_end[0]+0.75, t_end[-1]], 
                              [0, amp/2, -amp/8, 0])(t_end)
    
    # 组合
    t_total = np.concatenate([t_start, t_mid, t_end])
    y_total = np.concatenate([y_start, y_mid, y_end])
    
    return t_total, y_total

def align_signals(ref_y, target_y):
    """利用互相关函数自动寻找最佳对齐点截取/填充波形"""
    correlation = correlate(target_y, ref_y, mode='full')
    lag = np.argmax(np.abs(correlation)) - (len(ref_y) - 1)
    
    # 互相关可能匹配到反相（如果PCA特征向量方向和GT反了）
    if correlation[np.argmax(np.abs(correlation))] < 0:
        target_y = -target_y
        
    aligned_y = np.zeros_like(ref_y)
    
    if lag > 0:
        valid_len = min(len(ref_y), len(target_y) - lag)
        aligned_y[:valid_len] = target_y[lag:lag+valid_len]
    else:
        valid_len = min(len(target_y), len(ref_y) + lag)
        aligned_y[-lag:-lag+valid_len] = target_y[:valid_len]
        
    return aligned_y

def main():
    parser = argparse.ArgumentParser(description="综合对比 Discussion 波形图")
    parser.add_argument("csv_files", nargs='+', help="输入的CSV文件列表 (支持多个)")
    parser.add_argument("--labels", nargs='+', default=[], help="图例标签 (例如 Baseline Cond2 Cond3)")
    parser.add_argument("--cutoff", type=float, default=0.2, help="消除基线漂移的高通滤波截止频率 (Hz)")
    args = parser.parse_args()

    fs = 60.0
    t_gt, y_gt = generate_ground_truth(fs=fs, freq=1.0, amp=2.0, cycles=20)
    
    plt.figure(figsize=(14, 6))
    
    # 绘制 Ground Truth
    plt.plot(t_gt, y_gt, label="Ground Truth (Shake Table)", color="black", linestyle="--", linewidth=2.5, zorder=10)

    colors = ['#1f77b4', '#ff7f0e', '#d62728', '#2ca02c', '#9467bd']
    
    for i, csv_file in enumerate(args.csv_files):
        if not os.path.exists(csv_file):
            print(f"警告：找不到文件 {csv_file}")
            continue
            
        data = np.genfromtxt(csv_file, delimiter=',', names=True)
        mean_x, mean_y = data['mean_x'], data['mean_y']
        
        # # 1. 提取主位移 (PCA)
        # raw_disp = extract_principal_displacement(mean_x, mean_y)
        # 1. 提取位移
        raw_disp = extract_total_displacement(mean_x, mean_y)
        
        # 2. 去特征漂移 (Detrend + HP Filter)
        detrended_disp = detrend(raw_disp)
        filtered_disp = butter_highpass_filter(detrended_disp, cutoff=args.cutoff, fs=fs)
        
        # 3. 自动时间轴同步与截取
        aligned_disp = align_signals(y_gt, filtered_disp)
        
        label = args.labels[i] if i < len(args.labels) else os.path.basename(csv_file)
        plt.plot(t_gt, aligned_disp, label=label, color=colors[i % len(colors)], linewidth=1.5, alpha=0.8)

    # 布局美化
    plt.title("Physical Displacement Synthesis & Limitations Validation", fontsize=16, pad=15)
    plt.xlabel("Time (s)", fontsize=14)
    plt.ylabel("Synthesized Displacement (cm)", fontsize=14)
    plt.axhline(0, color='gray', linewidth=1, alpha=0.5)
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.legend(loc="upper right", fontsize=11, framealpha=0.9)
    
    # 针对启动和结尾阶段高亮背景
    plt.axvspan(0, 1.0, color='yellow', alpha=0.1, label='Init Cycle')
    plt.axvspan(21.0, 22.0, color='pink', alpha=0.1, label='Tail Cycle')

    plt.tight_layout()
    out_file = "discussion_waveforms_comparison.png"
    plt.savefig(out_file, dpi=400)
    print(f"\n✅ 绘图完成！已保存为: {out_file}")
    plt.show()

if __name__ == "__main__":
    main()
