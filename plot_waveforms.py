import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def find_waveform_csv(path_arg=None):
    if path_arg:
        if os.path.exists(path_arg):
            return path_arg
        raise FileNotFoundError(f"指定文件不存在: {path_arg}")
    # 自动在 output/ 及其子目录中递归查找 *_waveforms.csv
    pattern = os.path.join("output", "**", "*_waveforms.csv")
    candidates = sorted(glob.glob(pattern, recursive=True))
    if not candidates:
        raise FileNotFoundError("未在 output/ 目录找到任何 *_waveforms.csv 文件。")
    return candidates[-1]  # 默认使用最后一个

def plot_all_waveforms(csv_path, out_png=None, normalize=False, subtract_median=False):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise RuntimeError("CSV 文件列数不足，预期第一列为 time，其余为波形列。")
    time = df.iloc[:,0].values
    cols = df.columns.tolist()[1:]
    data = df[cols].values  # shape (T, N)
    # mask columns that are entirely NaN
    valid_cols = [i for i in range(data.shape[1]) if not np.all(np.isnan(data[:,i]))]
    if len(valid_cols) == 0:
        raise RuntimeError("没有有效的波形列可绘制（全部为 NaN）。")
    data = data[:, valid_cols]
    cols = [cols[i] for i in valid_cols]

    if subtract_median:
        # subtract per-track median of valid samples to center traces
        for j in range(data.shape[1]):
            col = data[:,j]
            valid = ~np.isnan(col)
            if valid.any():
                data[valid,j] = col[valid] - np.median(col[valid])

    if normalize:
        # z-score normalize each track (only valid samples)
        for j in range(data.shape[1]):
            col = data[:,j]
            valid = ~np.isnan(col)
            if valid.sum() > 1:
                mu = np.nanmean(col[valid])
                sigma = np.nanstd(col[valid])
                if sigma > 0:
                    data[valid,j] = (col[valid] - mu) / sigma

    plt.figure(figsize=(12,7))
    cmap = plt.get_cmap("tab20")
    N = data.shape[1]
    for j in range(N):
        col = data[:,j]
        valid = ~np.isnan(col)
        plt.plot(time[valid], col[valid], color=cmap(j % 20), linewidth=0.9, alpha=0.8)

    # plot ensemble median (忽略 NaN)
    median_trace = np.nanmedian(data, axis=1)
    plt.plot(time, median_trace, color="k", linewidth=2.0, label="median", alpha=0.95)

    plt.xlabel("Time (s)")
    plt.ylabel("Relative Y displacement (pixels)")
    plt.title(os.path.basename(csv_path))
    plt.grid(alpha=0.3)
    plt.legend(loc='upper right')
    plt.tight_layout()

    if out_png is None:
        out_dir = os.path.dirname(csv_path) or "."
        base = os.path.splitext(os.path.basename(csv_path))[0]
        out_png = os.path.join(out_dir, base + "_all_waveforms.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png

if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        csv_path = find_waveform_csv(arg)
    except Exception as e:
        print("ERROR:", e)
        sys.exit(1)
    print("Using waveform CSV:", csv_path)
    outpng = plot_all_waveforms(csv_path, normalize=False, subtract_median=False)
    print("Saved plot to:", outpng)