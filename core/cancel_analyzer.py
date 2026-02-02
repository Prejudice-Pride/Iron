import os
import numpy as np
import matplotlib.pyplot as plt
import csv

class CancelAnalyzer:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_comparison(self, base_name, tracks_raw, tracks_corrected, fps):
        """
        Compare target movement before and after cancellation.
        Uses displacement magnitude: sqrt(dx^2 + dy^2)
        """
        T = tracks_raw.shape[1]
        times = np.arange(T) / float(fps)
        
        def get_avg_magnitude(tracks):
            # tracks: (N, T, 2)
            n_points = tracks.shape[0]
            magnitudes = []
            
            for i in range(n_points):
                pts = tracks[i]
                # Filter out NaNs
                valid_mask = ~np.isnan(pts[:, 0])
                if not any(valid_mask):
                    continue
                
                # Relative to first valid frame
                first_idx = np.where(valid_mask)[0][0]
                x0, y0 = pts[first_idx]
                
                dx = pts[:, 0] - x0
                dy = pts[:, 1] - y0
                
                # sqrt(x^2 + y^2)
                mag = np.sqrt(dx**2 + dy**2)
                magnitudes.append(mag)
                
            if len(magnitudes) == 0:
                return np.zeros(T)
            
            # Average across all points in the ROI
            return np.nanmean(np.array(magnitudes), axis=0)

        mag_raw = get_avg_magnitude(tracks_raw)
        mag_corr = get_avg_magnitude(tracks_corrected)

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(times, mag_raw, label="Before Cancellation (Raw Target)", color='red', alpha=0.5, linestyle='--')
        plt.plot(times, mag_corr, label="After Cancellation (Final Motion)", color='blue', linewidth=2)
        
        plt.title(f"Cancellation Effect Analysis - {base_name}")
        plt.xlabel("Time (s)")
        plt.ylabel("Displacement Magnitude (px)")
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.7)
        
        plot_path = os.path.join(self.output_dir, f"{base_name}_cancel_analysis.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()
        
        # Save CSV for the comparison
        csv_path = os.path.join(self.output_dir, f"{base_name}_cancel_comparison.csv")
        data = np.column_stack([times, mag_raw, mag_corr])
        np.savetxt(csv_path, data, delimiter=",", header="time,magnitude_raw,magnitude_corrected", comments="")
        
        return plot_path
