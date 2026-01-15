import os
import numpy as np
import csv
import warnings

class Analyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

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
        
        header = ["time", "mean_y", "median_y", "mean_x", "median_x"] + [f"pt{i}_y" for i in range(N)]
        
        # Prepare data columns
        # Handle NaNs in mean/median if necessary (usually they just stay Nan)
        
        data_list = [times, mean_y, median_y, mean_x, median_x]
        for i in range(N):
            data_list.append(displacements_y[:, i])
            
        data = np.column_stack(data_list)
        
        fmt = ["%.6f"] * data.shape[1]
        np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments='', fmt=fmt)
        
        return csv_path
