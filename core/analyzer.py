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
        Transform pixel tracks (N, T, 2) to physical mm tracks.
        """
        if self.homography_matrix is None:
            return tracks * self.pixels_to_mm_fallback
            
        N, T, _ = tracks.shape
        # Reshape to (N*T, 1, 2) for cv2.perspectiveTransform
        reshaped_tracks = tracks.reshape(-1, 1, 2).astype(np.float32)
        import cv2
        phys_tracks = cv2.perspectiveTransform(reshaped_tracks, self.homography_matrix)
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
        spec_data = np.column_stack((freqs, fft_y, fft_x))
        header = "frequency,amplitude_y,amplitude_x"
        np.savetxt(spec_csv_path, spec_data, delimiter=",", header=header, comments='', fmt="%.6f")
        
        # Plot Spectrum
        plot_path = os.path.join(self.output_dir, base_name + "_spectrum.png")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # Plot Y Spectrum
        ax1.plot(freqs, fft_y, 'b-', label='Y Spectrum')
        ax1.set_title(f"Frequency Spectrum - Y (Vertical) | FPS={fps:.2f}")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.5)
        # Annotate peak
        if len(freqs) > 1:
            peak_idx_y = np.argmax(fft_y[1:]) + 1 # ignore DC at index 0
            peak_freq_y = freqs[peak_idx_y]
            peak_amp_y = fft_y[peak_idx_y]
            ax1.annotate(f'Peak: {peak_freq_y:.2f} Hz', 
                         xy=(peak_freq_y, peak_amp_y), 
                         xytext=(peak_freq_y, peak_amp_y * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        # Plot X Spectrum
        ax2.plot(freqs, fft_x, 'r-', label='X Spectrum')
        ax2.set_title(f"Frequency Spectrum - X (Horizontal) | FPS={fps:.2f}")
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Amplitude")
        ax2.grid(True, alpha=0.5)
        if len(freqs) > 1:
            peak_idx_x = np.argmax(fft_x[1:]) + 1
            peak_freq_x = freqs[peak_idx_x]
            peak_amp_x = fft_x[peak_idx_x]
            ax2.annotate(f'Peak: {peak_freq_x:.2f} Hz', 
                         xy=(peak_freq_x, peak_amp_x), 
                         xytext=(peak_freq_x, peak_amp_x * 1.1),
                         arrowprops=dict(facecolor='black', shrink=0.05))

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close(fig)
        
        return plot_path

    def compute_and_save_double_waveforms(self, base_name, tracks, fps):
        """
        Plot both X and Y average displacement waveforms.
        Y is at the top because it's usually more critical.
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
        
        ax1.plot(times, mean_y, color='blue', label='Mean Y Displacement')
        ax1.set_ylabel("Displacement Y (pixels)")
        ax1.set_title(f"Waveform Analysis - {base_name}")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper right')

        ax2.plot(times, mean_x, color='red', label='Mean X Displacement')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Displacement X (pixels)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        return plot_path

    def calculate_intensity(self, tracks, fps, pixels_to_mm=None):
        """
        Calculate Seismic Intensity (simplified Chinese Intensity Scale GB/T 17742).
        Uses a sliding window for PGA/PGV.
        """
        N, T, _ = tracks.shape
        dt = 1.0 / fps
        
        # 1. Transform coordinates to physical (mm)
        if pixels_to_mm is not None:
             # Manual override if provided
             self.pixels_to_mm_fallback = pixels_to_mm
             
        phys_tracks_all = self.pixel_to_physical(tracks) # (N, T, 2)
        
        # Average multiple points to get a stable signal
        valid_masks = ~np.isnan(phys_tracks_all[:, :, 0])
        mean_phys_tracks = np.zeros((T, 2))
        for t in range(T):
            if valid_masks[:, t].any():
                mean_phys_tracks[t] = np.nanmean(phys_tracks_all[valid_masks[:, t], t], axis=0)
            else:
                mean_phys_tracks[t] = mean_phys_tracks[t-1] if t > 0 else np.array([0, 0])

        # Displacement in mm (relative to start)
        disp_mm = (mean_phys_tracks - mean_phys_tracks[0])
        
        # 2. Velocity (v = dx/dt) - mm/s
        vel = np.diff(disp_mm, axis=0) / dt
        # 3. Acceleration (a = dv/dt) - mm/s^2 (Gal = 10 mm/s^2)
        acc = np.diff(vel, axis=0) / dt
        
        # Pad with 0 to keep same length as T for windowing
        vel = np.vstack([vel, [0, 0]])
        acc = np.vstack([acc, [0, 0], [0, 0]])

        # 4. Sliding Window (2.0 seconds)
        window_size = int(2.0 * fps)
        if window_size > T: window_size = T
        
        max_pga = 0.0
        max_pgv = 0.0
        
        for i in range(T - window_size + 1):
            window_acc = acc[i : i + window_size]
            window_vel = vel[i : i + window_size]
            
            # Resultant vector magnitude: sqrt(ax^2 + ay^2)
            acc_mags = np.sqrt(np.sum(window_acc**2, axis=1))
            vel_mags = np.sqrt(np.sum(window_vel**2, axis=1))
            
            max_pga = max(max_pga, np.max(acc_mags))
            max_pgv = max(max_pgv, np.max(vel_mags))
        
        # 5. Mapping to Intensity (Simplified GB/T 17742-2020)
        # Intensity 6 is usually PGA ~ 0.05g (approx 50 Gal or 500 mm/s^2)
        # PGA in Gal (cm/s^2), here we have mm/s^2 so / 10
        pga_gal = max_pga / 10.0
        pgv_cm_s = max_pgv / 10.0
        
        def mock_intensity(pga, pgv):
            # GB/T 17742-2020 Mapping
            if pga < 0.8: return 1
            if pga < 2.5: return 2
            if pga < 8:   return 3
            if pga < 22:  return 4
            if pga < 44:  return 5
            if pga < 89:  return 6
            if pga < 177: return 7
            if pga < 353: return 8
            if pga < 707: return 9
            if pga < 1414: return 10
            return 12

        intensity = mock_intensity(pga_gal, pgv_cm_s)
        
        # Console Output
        print("\n" + "="*60)
        print("SEISMIC INTENSITY CALCULATION REPORT (ENGINEERING CALIBRATION)")
        print("="*60)
        cal_mode = "HOMOGRAPHY MATRIX" if self.homography_matrix is not None else f"LINEAR SCALE ({self.pixels_to_mm_fallback} mm/pix)"
        print(f"Calibration Mode: {cal_mode}")
        print(f"Formula: Velocity v=∆d/∆t, Accel a=∆v/∆t, PGA=max(sqrt(ax^2+ay^2))")
        print(f"Sliding Window: 2.0s ({window_size} frames)")
        print("-" * 60)
        print(f"Peak Ground Acceleration (PGA): {max_pga:.2f} mm/s^2 ({pga_gal:.2f} Gal)")
        print(f"Peak Ground Velocity (PGV):     {max_pgv:.2f} mm/s ({(pgv_cm_s):.2f} cm/s)")
        print(f"Resultant Intensity (GB/T 17742): {intensity} 度")
        print("="*60 + "\n")
        
        return {
            "pga": max_pga,
            "pgv": max_pgv,
            "intensity": intensity
        }
