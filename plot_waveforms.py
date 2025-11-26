import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import csv

def plot_waveforms(csv_path, output_image_path=None, title="Displacement Waveforms", chunk_len_frames=100, show_stats=True):
    """
    读取由 SeismicTracker 生成的 waveforms.csv 文件并绘制波形图。
    在图上标注分段（chunk）边界，并计算每个边界处的跳变（delta）统计量以便检测断点处的不连续性。

    Args:
        csv_path (str): CSV 文件路径
        output_image_path (str, optional): 输出图片路径。如果不传，默认保存为同名的 .png 文件。
        title (str): 图表标题
        chunk_len_frames (int): 每段的帧长度（用于在时间轴上标注边界）
        show_stats (bool): 是否在图上显示每个边界的跳变统计值
    """
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}")
        return

    print(f"Reading waveforms from: {csv_path}")

    # 读取 CSV 头部以获取列名
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

    # 使用 numpy 读取数据（跳过标题行）
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.size == 0:
        print("Error: CSV file is empty or invalid.")
        return

    # 第一列是时间，后面是各个点的位移
    time = data[:, 0]
    traces = data[:, 1:]
    num_points = traces.shape[1]
    n_steps = len(time)
    print(f"Found {num_points} traces, {n_steps} time steps.")

    # 估算帧率（由 time 间隔近似）
    if n_steps >= 2:
        dt = np.median(np.diff(time))
        fps_est = 1.0 / dt if dt > 0 else None
    else:
        dt = None
        fps_est = None

    if fps_est is None:
        print("Warning: cannot estimate fps from time vector; chunk boundary markers may be incorrect.")

    # 计算边界对应的时间点
    if fps_est is not None and chunk_len_frames > 0:
        # chunk boundaries in frames: chunk_len_frames, 2*chunk_len_frames, ... (zero-indexed frame numbers)
        max_frame = int(np.round(time[-1] * fps_est))
        boundaries_frames = list(range(chunk_len_frames, max_frame + 1, chunk_len_frames))
        boundaries_times = [bf / fps_est for bf in boundaries_frames]
        # Only keep boundaries inside recorded time range
        boundaries_times = [t for t in boundaries_times if t > time[0] and t < time[-1]]
    else:
        boundaries_times = []

    # 绘图主图
    plt.figure(figsize=(14, 7))
    for i in range(num_points):
        if not np.all(np.isnan(traces[:, i])):
            plt.plot(time, traces[:, i], label=f"Point {i}", alpha=0.5, linewidth=1)

    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Relative Y Displacement (pixels)")
    plt.grid(True, linestyle='--', alpha=0.6)

    # 标注分段边界并计算跳变
    deltas_per_boundary = []  # 每个边界的各点 delta 列表
    for bt in boundaries_times:
        # 找到最接近的时间索引
        idx = np.searchsorted(time, bt)
        if idx <= 0 or idx >= n_steps:
            continue
        t_prev = idx - 1
        t_next = idx
        prev_vals = traces[t_prev, :]
        next_vals = traces[t_next, :]
        # delta = next - prev
        delta = next_vals - prev_vals
        deltas_per_boundary.append(delta)
        # 在主图上绘制竖线
        plt.axvline(x=bt, color='k', linestyle='--', linewidth=0.8, alpha=0.7)
        if show_stats:
            # 统计：均值，绝对均值，最大值
            mean_delta = np.nanmean(delta)
            mean_abs = np.nanmean(np.abs(delta))
            max_abs = np.nanmax(np.abs(delta))
            plt.text(bt, plt.gca().get_ylim()[1]*0.9, f"Δμ={mean_delta:.3f}\n|Δ|μ={mean_abs:.3f}\n|Δ|max={max_abs:.3f}",
                     rotation=90, va='top', ha='right', fontsize=8, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    # 如果点太少可以显示图例
    if num_points <= 10:
        plt.legend()

    plt.tight_layout()

    # 若有边界，画一个小的热图/条形图展示每个边界的绝对均值（侧边）
    if len(deltas_per_boundary) > 0:
        # 新建一个 figure 的 inset（在右边）来显示边界统计
        import matplotlib.gridspec as gridspec
        fig = plt.gcf()
        gs = gridspec.GridSpec(1, 4, figure=fig)
        ax_main = fig.axes[0]
        ax_inset = fig.add_subplot(gs[0, 3])
        # 计算每个边界的 mean abs delta
        mean_abs_list = [float(np.nanmean(np.abs(d))) for d in deltas_per_boundary]
        ax_inset.bar(range(len(mean_abs_list)), mean_abs_list)
        ax_inset.set_title('Mean |Δ| per boundary')
        ax_inset.set_xlabel('Boundary #')
        ax_inset.set_ylabel('pixels')
        plt.subplots_adjust(right=0.8)

    # 确定保存路径
    if output_image_path is None:
        output_image_path = os.path.splitext(csv_path)[0] + "_chunks.png"

    plt.savefig(output_image_path, dpi=150)
    print(f"Plot saved to: {output_image_path}")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_waveforms.py <video_base_name_or_csv_path> [chunk_len_frames]")
        print("Example 1: python plot_waveforms.py sample 100")
        print("Example 2: python plot_waveforms.py output/sample_output_auto/sample_waveforms.csv 150")
        sys.exit(1)

    input_arg = sys.argv[1]
    # optional chunk length argument
    chunk_len = int(sys.argv[2]) if len(sys.argv) > 2 else 100

    # 1. 如果输入的是一个存在的 CSV 文件，直接绘制
    if os.path.isfile(input_arg) and input_arg.endswith(".csv"):
        plot_waveforms(input_arg, chunk_len_frames=chunk_len)
        sys.exit(0)

    # 2. 否则，将其视为视频的基础名称 (base_name)，去 output 文件夹下查找
    base_name = input_arg
    # 如果用户输入了 sample.mp4，去掉后缀
    if base_name.endswith(".mp4"):
        base_name = base_name[:-4]

    modes = ["auto", "manual"]
    found_any = False

    print(f"Searching for waveforms for base name: '{base_name}'...")

    for mode in modes:
        # 路径结构: output/{base}_output_{mode}/{base}_waveforms.csv
        csv_path = os.path.join("output", f"{base_name}_output_{mode}", f"{base_name}_waveforms.csv")
        
        if os.path.exists(csv_path):
            print(f"--> Found {mode} mode file: {csv_path}")
            plot_waveforms(csv_path, title=f"Waveforms: {base_name} ({mode})", chunk_len_frames=chunk_len)
            found_any = True

    if not found_any:
        print(f"Error: No waveform files found in output/ for '{base_name}'.")
        print(f"Expected paths like: output/{base_name}_output_auto/{base_name}_waveforms.csv")
