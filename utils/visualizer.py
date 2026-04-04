import cv2
import numpy as np
import imageio.v3 as iio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import io

import time
from datetime import datetime

# 打印当前时间节点
def print_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def visualize_tracks(frames, tracks, visibilities, out_path, fps, rois=None):
    """
    Visualizes tracking results by drawing points and trails on video frames.
    
    Args:
        frames: (T, H, W, 3) RGB video frames
        tracks: (N, T, 2) Coordinates of tracked points
        visibilities: (N, T) Vsibilities of tracked points
        out_path: Path to save the output video
        fps: Frames per second
        rois: Optional list of ROIs to draw [x1, y1, x2, y2]
    """
    N, T, _ = tracks.shape
    # N points
    rng = np.random.RandomState(42)
    # Different colors for points, or uniform? 
    # Original script used random colors.
    colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]
    
    vis = []
    print(f"Generating visualization for {T} frames...")
    
    for t in range(T):
        # Convert RGB to BGR for OpenCV drawing
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        
        # Draw ROIs
        if rois:
            for r in rois:
                cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)
                
        for i in range(N):
            pts = tracks[i]
            pts_vis = visibilities[i]
            # Draw trails
            valid = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid) >= 2:
                vp = pts[:t+1][valid].astype(np.int32)
                for k in range(len(vp)-1):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k+1]), colors[i], 1)
            
            # Draw current point
            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                if pts_vis[t]:
                    cv2.circle(img, (x,y), 3, colors[i], -1)
                else:
                    for angle in range(0, 360, 30):
                        cv2.ellipse(img, (x,y), (3,3), 0, angle, angle+15, colors[i], 1)
        
        # Convert back to RGB for imageio
        vis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Save video
    iio.imwrite(out_path, np.array(vis), plugin="FFMPEG", fps=fps, macro_block_size=1)
    return out_path


def visualize_tracks_with_plot(frames, tracks, visibilities, out_path, fps, rois=None, plot_width=None):
    """
    Generate a side-by-side video: left = video frames with tracks, right = time vs displacement plot up to current frame.

    Args:
        frames: (T, H, W, 3) RGB frames
        tracks: (N, T, 2)
        visibilities: (N, T)
        out_path: output path
        fps: frames per second
        rois: optional ROIs to draw
        plot_width: desired width of the plot panel in pixels (defaults to frame width)
    """
    N, T, _ = tracks.shape
    H = frames.shape[1]
    W = frames.shape[2]
    if plot_width is None:
        plot_width = W

    # Precompute mean displacement (Y) and X for plotting
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

    with np.errstate(invalid='ignore'):
        mean_y = np.nanmean(displacements_y, axis=1)
        mean_x = np.nanmean(displacements_x, axis=1)

    # Fill NaNs in mean signals
    def _fill_nan_signal(sig):
        if np.isnan(sig).any():
            x_axis = np.arange(len(sig))
            mask = np.isnan(sig)
            if (~mask).any():
                sig[mask] = np.interp(x_axis[mask], x_axis[~mask], sig[~mask])
            else:
                sig[:] = 0
        return sig

    mean_y = _fill_nan_signal(mean_y.copy())
    mean_x = _fill_nan_signal(mean_x.copy())

    # Colors
    rng = np.random.RandomState(42)
    colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]

    out_frames = []
    print(f"Generating side-by-side visualization for {T} frames...")

    for t in range(T):
        # Left: draw video frame with tracks (reuse logic)
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        if rois:
            for r in rois:
                cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)

        for i in range(N):
            pts = tracks[i]
            valid = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid) >= 2:
                vp = pts[:t+1][valid].astype(np.int32)
                for k in range(len(vp)-1):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k+1]), colors[i], 1)

            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                if visibilities[i, t]:
                    cv2.circle(img, (x,y), 3, colors[i], -1)
                else:
                    for angle in range(0, 360, 30):
                        cv2.ellipse(img, (x,y), (3,3), 0, angle, angle+15, colors[i], 1)

        left_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Right: generate plot image up to frame t
        fig = Figure(figsize=(plot_width/100.0, H/100.0), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.plot(times, mean_x, color='red', alpha=0.3)
        ax.plot(times[:t+1], mean_x[:t+1], color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement X (cm)')
        ax.grid(True, alpha=0.4)
        ax.axvline(times[t], color='black', linestyle='--', linewidth=1)
        ax.set_xlim(times[0], times[-1] if len(times)>1 else 1)
        # autoscale y with a margin
        ymin = np.nanmin(mean_x)
        ymax = np.nanmax(mean_x)
        if np.isfinite(ymin) and np.isfinite(ymax):
            if ymin == ymax:
                ymin -= 1
                ymax += 1
            pad = (ymax - ymin) * 0.1
            ax.set_ylim(ymin - pad, ymax + pad)

        canvas.draw()
        buf = canvas.buffer_rgba()
        plot_w, plot_h = canvas.get_width_height()
        plot_arr = np.frombuffer(buf, dtype=np.uint8).reshape(plot_h, plot_w, 4)
        # Convert RGBA to RGB
        plot_arr = plot_arr[:, :, :3]

        # Resize plot to match (H, plot_width)
        plot_resized = cv2.resize(plot_arr, (plot_width, H))

        # Concatenate left and right
        # Ensure left_rgb sized (H,W,3)
        if left_rgb.shape[0] != H or left_rgb.shape[1] != W:
            left_rgb = cv2.resize(left_rgb, (W, H))

        combined = np.hstack([left_rgb, plot_resized])
        out_frames.append(combined)
        plt.close(fig)


    # Save video
    # Use imageio v3 with imopen to avoid huge np.array allocations and potential OS pipe errors
    import imageio.v3 as iio_v3
    
    # 使用示例
    print_timestamp("开始处理")
    
    with iio_v3.imopen(out_path, "w", plugin="FFMPEG") as file:
        file.write(np.stack(out_frames), fps=fps, macro_block_size=1,output_params=['-c:v', 'h264_nvenc', '-preset', 'p4'])
    print_timestamp("处理结束")
    return out_path

    