import os
import numpy as np
import csv
import warnings
import matplotlib.pyplot as plt

class Analyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.homography_matrix = None
        self.pixels_to_mm_fallback = 10.0 # Default if no calibration

    def setup_calibration(self, src_pts, dst_pts):
        """
        Calculate Homography matrix to map pixels to physical coordinates.
        Supports robust estimation using RANSAC if more than 4 points are provided.
        
        src_pts: List of pixel coordinates [x, y] from the video (N >= 4)
        dst_pts: List of corresponding physical coordinates [x_mm, y_mm] (N >= 4)
        """
        import cv2
        src_pts = np.array(src_pts, dtype=np.float32)
        dst_pts = np.array(dst_pts, dtype=np.float32)
        
        if len(src_pts) < 4:
            print("[Analyzer] Error: Need at least 4 points for calibration.")
            return

        # Use RANSAC robust estimation to handle outliers or slight manual errors
        # ransacReprojThreshold=5.0 means points with >5 pixel error are treated as outliers
        self.homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        inliers = np.sum(mask)
        print(f"[Analyzer] Calibration established using {inliers}/{len(src_pts)} points (RANSAC).")

    def pixel_to_physical(self, tracks):
        """
        Transform pixel tracks (N, T, 2) to physical cm tracks.
        (Note: Internally converts the calibrated output [which defaults to mm] to cm)
        """
        if self.homography_matrix is None:
            # Fallback is assumed to be 10.0 mm/pixel, so we return in cm:
            return (tracks * self.pixels_to_mm_fallback) / 10.0
            
        N, T, _ = tracks.shape
        # Reshape to (N*T, 1, 2) for cv2.perspectiveTransform
        reshaped_tracks = tracks.reshape(-1, 1, 2).astype(np.float32)
        import cv2
        phys_tracks = cv2.perspectiveTransform(reshaped_tracks, self.homography_matrix)
        # Convert from mm (which dst_pts uses) to cm
        phys_tracks = phys_tracks / 10.0
        return phys_tracks.reshape(N, T, 2)

    def save_tracks(self, base_name, tracks, fps):
        """
        Save tracks to npz and csv.
        tracks: (N, T, 2)
        """
        npz_path = os.path.join(self.output_dir, base_name + "_surface_tracks.npz")
        np.savez_compressed(npz_path, tracks=tracks, fps=fps)
        
        csv_path = os.path.join(self.output_dir, base_name + "_surface_tracks.csv")
        N, T, _ = tracks.shape
        
        # Writing large CSVs can be slow, user might want this optional?
        # Original script does it.
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["id", "frame", "x", "y"])
            for i in range(N):
                for t in range(T):
                    x = tracks[i, t, 0]
                    y = tracks[i, t, 1]
                    if np.isnan(x) or np.isnan(y):
                        w.writerow([i, t, "", ""])
                    else:
                        w.writerow([i, t, f"{x:.2f}", f"{y:.2f}"])
                        
        return npz_path, csv_path

    def compute_and_save_waveform(self, base_name, tracks, fps):
        """
        Calculate relative displacements and average them to get the object vibration.
        tracks: (N, T, 2)
        """
        N, T, _ = tracks.shape
        times = np.arange(T) / float(fps)
        
        # (T, N)
        displacements_y = np.full((T, N), np.nan, dtype=np.float32)
        displacements_x = np.full((T, N), np.nan, dtype=np.float32)
        
        for i in range(N):
            pts = tracks[i]
            valid = ~np.isnan(pts[:, 1])
            if valid.any():
                first_idx = np.where(valid)[0][0]
                y0 = pts[first_idx, 1]
                x0 = pts[first_idx, 0]
                displacements_y[valid, i] = pts[valid, 1] - y0
                displacements_x[valid, i] = pts[valid, 0] - x0
        
        # Robust averaging
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            warnings.filterwarnings('ignore', message='All-NaN slice encountered')
            mean_y = np.nanmean(displacements_y, axis=1)
            median_y = np.nanmedian(displacements_y, axis=1)
            mean_x = np.nanmean(displacements_x, axis=1)
            median_x = np.nanmedian(displacements_x, axis=1)

        # (已移除 scipy detrend 去漂移逻辑，纯粹展示运动)
        
        # Save to CSV
        csv_path = os.path.join(self.output_dir, base_name + "_surface_waveform.csv")
        
        # Header includes both X and Y for each point for full waveform recovery
        header = ["time", "mean_y", "median_y", "mean_x", "median_x"]
        for i in range(N):
            header.append(f"pt{i}_x")
            header.append(f"pt{i}_y")
        
        # Prepare data columns
        data_list = [times, mean_y, median_y, mean_x, median_x]
        for i in range(N):
            data_list.append(displacements_x[:, i])
            data_list.append(displacements_y[:, i])
            
        data = np.column_stack(data_list)
        
        fmt = ["%.6f"] * data.shape[1]
        np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments='', fmt=fmt)
        
        return csv_path

    def compute_and_save_spectrum(self, base_name, tracks, fps):
        """
        Compute FFT for the average displacement waveforms and save spectrum plots.
        """
        N, T, _ = tracks.shape
        
        # (T, N)
        displacements_y = np.full((T, N), np.nan, dtype=np.float32)
        displacements_x = np.full((T, N), np.nan, dtype=np.float32)
        
        # Calculate displacements and interpolate NaNs
        for i in range(N):
            pts = tracks[i]
            # Use original coordinates to compute displacement from start
            # Need to handle cases where start is NaN
            
            # Find first valid index for this track
            valid_mask = ~np.isnan(pts[:, 1])
            if not valid_mask.any():
                continue
                
            first_idx = np.where(valid_mask)[0][0]
            y0 = pts[first_idx, 1]
            x0 = pts[first_idx, 0]
            
            dy = pts[:, 1] - y0
            dx = pts[:, 0] - x0
            
            # Interpolate raw track NaNs for FFT continuity
            # (Note: pts can have NaNs in the middle if tracking lost momentarily)
            nans = np.isnan(dy)
            # Only interpolate if we have some data and some nans
            if nans.any() and (~nans).any():
                x_axis = np.arange(T)
                # Fill beginning/end NaNs with nearest valid or 0? 
                # np.interp uses constant extrapolation by default for points outside range? 
                # No, np.interp holds the end values. That is acceptable.
                dy[nans] = np.interp(x_axis[nans], x_axis[~nans], dy[~nans])
                dx[nans] = np.interp(x_axis[nans], x_axis[~nans], dx[~nans])
            elif nans.all():
                # Should not happen due to valid_mask check, but safety
                dy[:] = 0
                dx[:] = 0

            displacements_y[:, i] = dy
            displacements_x[:, i] = dx

        # Compute average waveform (ignoring fully-NaN tracks if any)
        # Use nanmean just in case some tracks were completely empty/skipped
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Mean of empty slice')
            mean_y = np.nanmean(displacements_y, axis=1)
            mean_x = np.nanmean(displacements_x, axis=1)
        
        # Fill any remaining NaNs in the mean signal (e.g. if all tracks were NaN at some frame)
        if np.isnan(mean_y).any():
            x_axis = np.arange(T)
            mask_y = np.isnan(mean_y)
            if (~mask_y).any():
                mean_y[mask_y] = np.interp(x_axis[mask_y], x_axis[~mask_y], mean_y[~mask_y])
            else:
                mean_y[:] = 0
        
        if np.isnan(mean_x).any():
            x_axis = np.arange(T)
            mask_x = np.isnan(mean_x)
            if (~mask_x).any():
                mean_x[mask_x] = np.interp(x_axis[mask_x], x_axis[~mask_x], mean_x[~mask_x])
            else:
                mean_x[:] = 0

        # FFT Calculation
        # Detrend (remove DC offset)
        mean_y_detrend = mean_y - np.mean(mean_y)
        mean_x_detrend = mean_x - np.mean(mean_x)
        
        n = len(mean_y_detrend)
        freqs = np.fft.rfftfreq(n, d=1.0/fps)
        
        # Normalized amplitude spectrum
        fft_y = np.abs(np.fft.rfft(mean_y_detrend)) / n * 2 
        fft_x = np.abs(np.fft.rfft(mean_x_detrend)) / n * 2
        
        # Save Spectrum Data
        spec_csv_path = os.path.join(self.output_dir, base_name + "_spectrum.csv")
        spec_data = np.column_stack((freqs, fft_x, fft_y))
        header = "frequency,amplitude_x,amplitude_y"
        np.savetxt(spec_csv_path, spec_data, delimiter=",", header=header, comments='', fmt="%.6f")
        
        # Plot Spectrum
        plot_path = os.path.join(self.output_dir, base_name + "_spectrum.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot X Spectrum (Top)
        ax1.plot(freqs, fft_x, 'r-', label='X Spectrum')
        ax1.set_title(f"Frequency Spectrum - X (Horizontal) | FPS={fps:.2f}")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.5)
        if len(freqs) > 1:
            peak_idx_x = np.argmax(fft_x[1:]) + 1
            peak_freq_x = freqs[peak_idx_x]
            peak_amp_x = fft_x[peak_idx_x]
            ax1.annotate(f'Peak: {peak_freq_x:.2f} Hz', 
                         xy=(peak_freq_x, peak_amp_x), 
                         xytext=(peak_freq_x, peak_amp_x * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        # Plot Y Spectrum (Bottom)
        ax2.plot(freqs, fft_y, 'b-', label='Y Spectrum')
        ax2.set_title(f"Frequency Spectrum - Y (Vertical) | FPS={fps:.2f}")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.5)
        # Annotate peak
        if len(freqs) > 1:
            peak_idx_y = np.argmax(fft_y[1:]) + 1 # ignore DC at index 0
            peak_freq_y = freqs[peak_idx_y]
            peak_amp_y = fft_y[peak_idx_y]
            ax2.annotate(f'Peak: {peak_freq_y:.2f} Hz', 
                         xy=(peak_freq_y, peak_amp_y), 
                         xytext=(peak_freq_y, peak_amp_y * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        return plot_path

    def compute_and_save_double_waveforms(self, base_name, tracks, fps):
        """
        Plot both X and Y average displacement waveforms.
        X is at the top, Y is at the bottom.
        """
        # We reuse the logic to get mean_y and mean_x
        N, T, _ = tracks.shape
        times = np.arange(T) / float(fps)
        
        displacements_y = np.full((T, N), np.nan, dtype=np.float32)
        displacements_x = np.full((T, N), np.nan, dtype=np.float32)
        
        for i in range(N):
            pts = tracks[i]
            valid = ~np.isnan(pts[:, 1])
            if valid.any():
                first_idx = np.where(valid)[0][0]
                y0 = pts[first_idx, 1]
                x0 = pts[first_idx, 0]
                displacements_y[valid, i] = pts[valid, 1] - y0
                displacements_x[valid, i] = pts[valid, 0] - x0
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            mean_y = np.nanmean(displacements_y, axis=1)
            mean_x = np.nanmean(displacements_x, axis=1)

        plot_path = os.path.join(self.output_dir, base_name + "_double_waveforms.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        ax1.plot(times, mean_x, color='red', label='Mean X Displacement')
        ax1.set_ylabel("Displacement X (cm)")
        ax1.set_title(f"Waveform Analysis ")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        ax2.plot(times, mean_y, color='blue', label='Mean Y Displacement')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement Y (cm)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def calculate_intensity(self, tracks, fps, pixels_to_mm=None):
        """
        理清烈度计算的全流程逻辑:
        1. (已提前处理) 轨迹映射：外部已完成单应矩阵转换，输入 tracks 直接为物理坐标 (cm)。
        2. 信号集成：对 ROI 内多个追踪点进行鲁棒平均，获得物体中心运动信号。
        3. 运动学推导：通过一阶和二阶数值微分获取速度 (cm/s) 和加速度 (cm/s^2 即 Gal)。
        4. 特征峰值提取：利用滑动窗口技术捕捉峰值水平加速度 (PGA) 和峰值水平速度 (PGV)。
        5. 烈度评估：基于 GB/T 17742-2020 标准映射计算地震烈度。
        """
        N, T, _ = tracks.shape
        dt = 1.0 / fps
        
        # --- Step 1: 物理尺度恢复 (Scale Restoration) ---
        # 外部已经调用了 pixel_to_physical 转换为了 cm
        phys_tracks_all = tracks # 结果单位: cm
        
        # --- Step 2: 信号集成 (Signal Aggregation) ---
        # 融合多个追踪点的位移，减少单点追踪噪声
        valid_masks = ~np.isnan(phys_tracks_all[:, :, 0])
        mean_phys_tracks = np.zeros((T, 2))
        for t in range(T):
            if valid_masks[:, t].any():
                mean_phys_tracks[t] = np.nanmean(phys_tracks_all[valid_masks[:, t], t], axis=0)
            else:
                mean_phys_tracks[t] = mean_phys_tracks[t-1] if t > 0 else np.array([0, 0])

        # 获取相对起始点的位移 (Relative Displacement)
        disp_mm = (mean_phys_tracks - mean_phys_tracks[0])
        
        # (已移除 scipy detrend 强制去漂移逻辑，纯粹展示运动)

        # --- (Optional) Signal Smoothing: 应用低通滤波以抑制求导放大的高频噪声 ---
        
        cutoff = 2.0
        
        from scipy.signal import butter, filtfilt
        def lowpass_filter(data, cutoff, fs=30.0, order=4):
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            # 处理 NaN: 仅对有效数据段滤波
            mask = ~np.isnan(data)
            if np.sum(mask) > 15: # 确保数据量足够滤波
                filtered = np.copy(data)
                filtered[mask] = filtfilt(b, a, data[mask])
                return filtered
            return data

        disp_mm_smooth = np.zeros_like(disp_mm)
        disp_mm_smooth[:, 0] = lowpass_filter(disp_mm[:, 0], cutoff, fs=fps)
        disp_mm_smooth[:, 1] = lowpass_filter(disp_mm[:, 1], cutoff, fs=fps)

        # --- Step 3: 运动学推导 (Kinematics Calculation) ---
        # 速度 v = ∆d / ∆t (cm/s)
        vel = np.diff(disp_mm_smooth, axis=0) / dt
        # 再次滤波：消除一阶导数产生的噪声
        vel[:, 0] = lowpass_filter(vel[:, 0], cutoff, fs=fps)
        vel[:, 1] = lowpass_filter(vel[:, 1], cutoff, fs=fps)
        
        # 加速度 a = ∆v / ∆t (cm/s^2 = Gal)
        acc = np.diff(vel, axis=0) / dt
        # 再次滤波：消除二阶导数极其严重的放大噪声
        acc[:, 0] = lowpass_filter(acc[:, 0], cutoff, fs=fps)
        acc[:, 1] = lowpass_filter(acc[:, 1], cutoff, fs=fps)
        
        # 对齐长度
        vel = np.vstack([vel, [0, 0]])
        acc = np.vstack([acc, [0, 0], [0, 0]])

        # --- Step 4: 峰值特征提取 (Peak Feature Extraction) ---
        # 计算全序列的合成加速度和速度模长
        acc_mags_all = np.sqrt(np.sum(acc**2, axis=1)) # cm/s^2 (Gal)
        vel_mags_all = np.sqrt(np.sum(vel**2, axis=1)) # cm/s

        # 1. 传统峰值法 (Traditional PGA)
        # 采用 2.0 秒滑动窗口捕捉最大震动特征 (地震工程常用窗口)
        window_size = int(2.0 * fps)
        if window_size > T: window_size = T
        
        max_pga = 0.0 # cm/s^2 (Gal)
        max_pgv = 0.0 # cm/s
        for i in range(T - window_size + 1):
            max_pga = max(max_pga, np.max(acc_mags_all[i : i + window_size]))
            max_pgv = max(max_pgv, np.max(vel_mags_all[i : i + window_size]))
        
        # 2. 统计百分位法 (Statistical P95 PGA)
        # 取第 95 百分位值，有效过滤单帧算法跳动带来的噪点
        p95_pga = np.percentile(acc_mags_all, 95) # cm/s^2

        # 3. 阿里亚斯烈度 (Arias Intensity - Ia)
        # 物理意义：代表震动总能量的积分指标。Ia = (pi/2g) * integral(a^2 dt)
        # 修正：将加速度从 cm/s^2 转换为 m/s^2 以符合标准国际单位制 (Standard units: m/s)
        acc_m_s2 = acc_mags_all / 100.0  # 100 cm = 1 m
        g_m_s2 = 9.80665                  # 标准重力加速度 (m/s^2)
        
        # 积分计算 (单位为 m/s)
        arias_intensity = (np.pi / (2 * g_m_s2)) * np.sum(acc_m_s2**2 * dt)

        # 4. 有效设计加速度 (Effective Design Acceleration)
        # 业内标准做法：取最大峰值的 75% 作为结构实际感受到的持续加速度
        effective_pga = 0.75 * max_pga # cm/s^2

        # --- Step 5: 烈度映射 (Intensity Mapping - GB/T 17742) ---
        # 此时已经均为 Gal (cm/s^2) 和 cm/s
        pga_gal = max_pga
        pgv_cm_s = max_pgv
        
        # 为四种方法分别计算烈度评估
        intensity_m1 = self._map_to_gbt17742(pga_gal, pgv_cm_s)
        intensity_m2 = self._map_to_gbt17742(p95_pga, 0) # 仅按加速度映射
        intensity_m4 = self._map_to_gbt17742(effective_pga, 0) # 仅按加速度映射
        
        # 针对 Arias Intensity (Ia)，采用学术界常用的经验转换（用于对比参考）
        # 注：Ia 为能量指标，此处将其映射到大致感知的烈度量级
        if arias_intensity > 0:
            # 这是一个参考映射：Ia 与烈度的对数关系
            log_ia = np.log10(arias_intensity)
            # 简化近似：Ia=0.01~1 对应 5-8 度
            if arias_intensity < 0.001: intensity_m3 = 1
            elif arias_intensity < 0.005: intensity_m3 = 3
            elif arias_intensity < 0.02: intensity_m3 = 5
            elif arias_intensity < 0.1: intensity_m3 = 6
            elif arias_intensity < 0.5: intensity_m3 = 7
            elif arias_intensity < 2.0: intensity_m3 = 8
            else: intensity_m3 = 9
        else:
            intensity_m3 = 0

        # 输出报告
        self._print_intensity_report(
            max_pga, pga_gal, max_pgv, pgv_cm_s, 
            intensity_m1, window_size, 
            p95_pga, arias_intensity, effective_pga,
            intensity_m2, intensity_m3, intensity_m4
        )
        
        return {
            "pga": max_pga,
            "pgv": max_pgv,
            "intensity_pga": intensity_m1,
            "intensity_p95": intensity_m2,
            "intensity_arias": intensity_m3,
            "intensity_effective": intensity_m4
        }

    def _map_to_gbt17742(self, pga, pgv):
        """依据中国地震烈度表 (GB/T 17742-2020) 进行简约映射"""
        if pga < 0.8: return 1
        if pga < 2.5: return 2
        if pga < 8:   return 3
        if pga < 22:  return 4
        if pga < 44:  return 5
        if pga < 89:  return 6   # 烈度 VI 度
        if pga < 177: return 7   # 烈度 VII 度
        if pga < 353: return 8   # 烈度 VIII 度
        if pga < 707: return 9
        if pga < 1414: return 10
        return 12

    def _print_intensity_report(self, max_pga, pga_gal, max_pgv, pgv_cm_s, intensity, window_size, p95_pga, arias_intensity, effective_pga, intensity_m2, intensity_m3, intensity_m4):
        print("\n" + "="*70)
        print("地震烈度多维度分析报告 (Multi-Dimensional Seismic Analysis)")
        print("="*70)
        cal_mode = "透视变换 (Homography)" if self.homography_matrix is not None else f"线性缩放 ({self.pixels_to_mm_fallback} mm/pix)"
        print(f"校准模式: {cal_mode}")
        print(f"分析窗口: 2.0s ({window_size} 帧)")
        print("-" * 70)
        
        print(f"[方法 1] 传统峰值法 (PGA - Absolute Peak):")
        print(f"   最大水平加速度: {max_pga:.2f} cm/s^2 ({pga_gal:.2f} Gal)")
        print(f"   最大水平速度:   {max_pgv:.2f} cm/s")
        print(f"   >>> 评估烈度: {intensity} 度")
        print("-" * 40)
        
        print(f"[方法 2] 统计百分位法 (P95 Robust Peak):")
        print(f"   P95 加速度:     {p95_pga:.2f} cm/s^2 ({p95_pga:.2f} Gal)")
        print(f"   >>> 评估烈度: {intensity_m2} 度 (更具抗噪性)")
        print("-" * 40)
        
        # print(f"[方法 3] 能量积分法 (Arias Intensity - Ia):")
        # print(f"   阿里亚斯烈度:   {arias_intensity:.4f} m/s")
        # print(f"   >>> 评估烈度: {intensity_m3} 度 (能量等效量级)")
        # print("-" * 40)
        
        print(f"[方法 3] 有效设计加速度 (Effective Acceleration):")
        print(f"   有效加速度:     {effective_pga:.2f} cm/s^2 ({effective_pga:.2f} Gal)")
        print(f"   >>> 评估烈度: {intensity_m4} 度 (结构设计实际感受)")
        
        print("="*70 + "\n")
