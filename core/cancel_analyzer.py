import os
import numpy as np
import matplotlib.pyplot as plt
import csv

class CancelAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_comparison(self, base_name, tracks_target_raw, tracks_ref, tracks_target_corrected, fps):
        """
        绘制三条曲线对比分析：
        - 原始目标轨迹 (Target Raw)
        - 纯参考抖动 (Reference Shift)
        - 抵消后的轨迹 (Corrected Target)
        """
        T = tracks_target_raw.shape[1]
        times = np.arange(T) / float(fps)
        
        def get_avg_displacement(tracks):
            # 获取 ROI 内所有点相对于各自起始位置的平均位移向量 (T, 2)
            n_points = tracks.shape[0]
            all_disps = []
            for i in range(n_points):
                pts = tracks[i]
                valid = ~np.isnan(pts[:, 0])
                if not any(valid): continue
                # 相对于第一帧的相对位移
                all_disps.append(pts - pts[np.where(valid)[0][0]])
            
            if not all_disps: return np.zeros((T, 2))
            return np.nanmean(np.array(all_disps), axis=0)

        # 获取各组的平均位移矢量
        disp_target_raw = get_avg_displacement(tracks_target_raw)
        disp_ref = get_avg_displacement(tracks_ref)
        disp_target_corr = get_avg_displacement(tracks_target_corrected)

        # 核心改进：在绘图前对齐基准线 (Subtract mean offset) 
        # 确保三条曲线都从 0 附近开始，消除追踪初始位置差异导致的常数偏置
        # 但是对 reference 不进行任何处理，以真实显示摄像机的位移
        disp_target_raw -= np.nanmean(disp_target_raw[:5], axis=0)
        disp_target_corr -= np.nanmean(disp_target_corr[:5], axis=0)

        # 计算模长用于绘图
        mag_target_raw = np.sqrt(np.sum(disp_target_raw**2, axis=1))
        mag_ref = np.sqrt(np.sum(disp_ref**2, axis=1))
        mag_target_corr = np.sqrt(np.sum(disp_target_corr**2, axis=1))

        # 绘图逻辑
        plt.figure(figsize=(12, 7))
        plt.plot(times, mag_target_raw, label="Target (Raw: Vibration + Camera)", color='red', alpha=0.4, linestyle='--')
        plt.plot(times, mag_ref, label="Reference (Camera Shift)", color='green', alpha=0.6, linestyle='-.')
        plt.plot(times, mag_target_corr, label="Target (Corrected: Actual)", color='blue', linewidth=2.5)
        
        plt.title(f"Dynamic Cancellation Analysis - {base_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement Magnitude (pixels)")
        plt.legend(loc='upper right', frameon=True, shadow=True)
        plt.grid(True, linestyle=':', alpha=0.7)
        
        # 添加标注：显示最大缩减量
        peak_reduction = np.max(mag_target_raw) - np.max(mag_target_corr)
        if peak_reduction > 0:
            plt.annotate(f'Peak Reduction: {peak_reduction:.2f} pixels', 
                         xy=(times[np.argmax(mag_target_raw)], np.max(mag_target_raw)),
                         xytext=(20, 20), textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', color='black'))

        plot_path = os.path.join(self.output_dir, f"{base_name}_cancel_analysis.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()
        
        # 保存详细对比 CSV
        csv_path = os.path.join(self.output_dir, f"{base_name}_cancel_comparison_triple.csv")
        data = np.column_stack([times, mag_target_raw, mag_ref, mag_target_corr])
        np.savetxt(csv_path, data, delimiter=",", 
                   header="time,mag_target_raw,mag_reference,mag_target_corrected", comments="")
        
        return plot_path
