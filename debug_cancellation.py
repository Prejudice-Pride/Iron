import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from scipy.signal import detrend
from scipy.ndimage import gaussian_filter1d


def process_cancellation(target_csv, ref_csv, fps=30.0, output_dir="debug_output"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading files:\nTarget: {target_csv}\nReference: {ref_csv}")

    df_target = pd.read_csv(target_csv)
    df_ref = pd.read_csv(ref_csv)

    # ----------------------------
    # CSV -> tracks (N, T, 2)
    # ----------------------------
    def csv_to_tracks(df):
        num_points = df['id'].max() + 1
        num_frames = df['frame'].max() + 1
        tracks = np.full((num_points, num_frames, 2), np.nan)

        for _, row in df.iterrows():
            tracks[int(row['id']), int(row['frame']), 0] = row['x']
            tracks[int(row['id']), int(row['frame']), 1] = row['y']
        return tracks

    tracks_target = csv_to_tracks(df_target)
    tracks_ref = csv_to_tracks(df_ref)
    T = tracks_target.shape[1]

    # ----------------------------
    # 工具：NaN 插值
    # ----------------------------
    def fill_nans(arr):
        for i in range(arr.shape[1]):
            mask = np.isnan(arr[:, i])
            if mask.any() and not mask.all():
                x = np.where(~mask)[0]
                y = arr[~mask, i]
                arr[mask, i] = np.interp(np.where(mask)[0], x, y)
            elif mask.all():
                arr[:, i] = 0
        return arr

    # ----------------------------
    # 1. Reference (Camera)
    # ----------------------------
    ref_initial = np.nanmedian(tracks_ref[:, :5, :], axis=1, keepdims=True)
    ref_shifts = tracks_ref - ref_initial

    avg_camera_shift = np.nanmedian(ref_shifts, axis=0)
    avg_camera_shift = fill_nans(avg_camera_shift)

    # ⚠️ 只做轻微平滑，不破坏物理趋势
    avg_camera_shift_smooth = gaussian_filter1d(avg_camera_shift, sigma=1.0, axis=0)

    # ----------------------------
    # 2. Target
    # ----------------------------
    target_initial = np.nanmedian(tracks_target[:, :5, :], axis=1, keepdims=True)
    target_shifts = tracks_target - target_initial

    avg_target_shift = np.nanmedian(target_shifts, axis=0)
    avg_target_shift = fill_nans(avg_target_shift)

    # ✅ 关键：target 去趋势（消除漂移源）
    avg_target_shift[:, 0] = detrend(avg_target_shift[:, 0], type='linear')
    avg_target_shift[:, 1] = detrend(avg_target_shift[:, 1], type='linear')

    # ----------------------------
    # 3. 相关性 + 权重
    # ----------------------------
    def safe_corr(a, b):
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            return 0.0
        return np.corrcoef(a, b)[0, 1]

    corr_x = safe_corr(avg_target_shift[:, 0], avg_camera_shift_smooth[:, 0])
    corr_y = safe_corr(avg_target_shift[:, 1], avg_camera_shift_smooth[:, 1])

    def get_axis_weight(corr):
        if corr > 0.6:
            return 1.0
        elif corr > 0.2:
            return (corr - 0.2) / 0.4
        else:
            return 0.0

    weight_x = get_axis_weight(corr_x)
    weight_y = get_axis_weight(corr_y)

    print(f"Correlation: X={corr_x:.4f}, Y={corr_y:.4f}")
    print(f"Weights: X={weight_x:.2f}, Y={weight_y:.2f}")

    # ----------------------------
    # 4. 执行抵消
    # ----------------------------
    cancel_weight_matrix = np.array([weight_x, weight_y])
    final_cancellation = avg_camera_shift_smooth * cancel_weight_matrix[None, :]

    corrected_tracks = tracks_target - final_cancellation[None, :, :]

    # ----------------------------
    # 5. 计算最终波形
    # ----------------------------
    mean_target_raw = np.nanmedian(target_shifts, axis=0)
    mean_target_corrected = np.nanmedian(corrected_tracks - target_initial, axis=0)

    # ✅ 最终强制去趋势（保证围绕0）
    mean_target_corrected[:, 0] = detrend(mean_target_corrected[:, 0], type='linear')
    mean_target_corrected[:, 1] = detrend(mean_target_corrected[:, 1], type='linear')

    # ----------------------------
    # 6. 可视化
    # ----------------------------
    time_axis = np.arange(T) / fps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.plot(time_axis, mean_target_raw[:, 1], label='Raw Target (Y)', alpha=0.4)
    ax1.plot(time_axis, final_cancellation[:, 1], '--', label='Camera Motion (Y)')
    ax1.plot(time_axis, mean_target_corrected[:, 1], label='Corrected (Y)', linewidth=2)
    ax1.set_title("Y-Axis")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(time_axis, mean_target_raw[:, 0], label='Raw Target (X)', alpha=0.4)
    ax2.plot(time_axis, final_cancellation[:, 0], '--', label='Camera Motion (X)')
    ax2.plot(time_axis, mean_target_corrected[:, 0], label='Corrected (X)', linewidth=2)
    ax2.set_title("X-Axis")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "correction_debug_plot.png")
    plt.savefig(plot_path)
    print(f"Debug plot saved to {plot_path}")

    # ----------------------------
    # 7. 保存 CSV
    # ----------------------------
    out_csv = os.path.join(output_dir, "corrected_output.csv")

    results = []
    N = corrected_tracks.shape[0]

    for i in range(N):
        for t in range(T):
            results.append([
                i, t,
                corrected_tracks[i, t, 0],
                corrected_tracks[i, t, 1]
            ])

    pd.DataFrame(results, columns=['id', 'frame', 'x', 'y']).to_csv(out_csv, index=False)
    print(f"Corrected CSV saved to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--out", default="debug_correction_result")
    args = parser.parse_args()

    process_cancellation(args.target, args.ref, args.fps, args.out)