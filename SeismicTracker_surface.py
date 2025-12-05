"""Run cotracker-based seismic extraction pipeline with Surface Tracking strategy.

This script implements a "Surface Tracking" strategy:
1. User selects a Region of Interest (ROI) covering the object to track.
2. A dense grid of points is generated within this surface (instead of random corners).
3. All points are tracked simultaneously.
4. The vibration of the object is calculated as the robust average (mean/median) of all point displacements.
   This mitigates the noise and instability of single-point tracking.

Usage: run from repo root.
  python SeismicTracker_surface.py [video_file] [output_dir]
"""
import os
import sys
import json
import numpy as np
import imageio.v3 as iio
import cv2
import torch
import torch.nn.functional as F

# Add co-tracker path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cotracker.predictor import CoTrackerPredictor

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"
ROI_DIR = "rois"
CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "scaled_offline.pth")
# Grid sampling spacing (px) for surface tracking
# Increased to 10 to avoid overly dense points and save compute
GRID_SPACING = 10
# Safety cap: if the generated grid yields more than this many points, subsample to avoid OOM
MAX_GRID_POINTS = 5000


def load_model(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    print(f"Loading weights from {CKPT_PATH}")
    model = CoTrackerPredictor(checkpoint=CKPT_PATH, offline=True).to(device)
    return model


def read_frames(path):
    frames = iio.imread(path, plugin="FFMPEG")
    frames = np.array(frames)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return frames, fps


def to_pixel_tracks(pred_tracks_np, frame_shape):
    arr = pred_tracks_np
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        T = arr.shape[0]
        if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
            tracks = arr
        else:
            tracks = np.transpose(arr, (1,0,2))
    else:
        raise RuntimeError("Unexpected pred_tracks shape: %s" % (arr.shape,))
    return tracks


def detect_corners_in_roi(frames, max_corners=100, quality=0.01, min_dist=7, block=7, valid_mask=None):
    """Detect corners only within the valid_mask (ROI)."""
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    
    # If mask is provided, we can either mask the image or filter points.
    # Masking the image is safer for goodFeaturesToTrack to find points ONLY in the region.
    mask = None
    if valid_mask is not None:
        mask = valid_mask.astype(np.uint8) * 255
        
    p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist, blockSize=block, mask=mask)
    
    if p0 is None:
        return np.empty((0,2), dtype=np.float32)
    pts = p0.reshape(-1,2).astype(np.float32)
    return pts


def load_surface_rois(base_name):
    os.makedirs(ROI_DIR, exist_ok=True)
    path = os.path.join(ROI_DIR, base_name + "_surface_rois.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                rois = json.load(f)
            rois = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
            return rois, path
        except Exception:
            return None, path
    return None, path


def save_surface_rois(path, rois):
    try:
        with open(path, 'w') as f:
            json.dump(rois, f)
        return True
    except Exception:
        return False


def select_surface_rois_interactive(img):
    """Let user draw rectangular ROIs for the OBJECT SURFACE."""
    win = "Select Object Surface - drag to draw, press r to remove last, q/Enter to finish"
    clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    rois = []
    current = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and current is not None:
            sx, sy, _, _ = current
            clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # Draw green box for object
            cv2.rectangle(clone, (sx, sy), (x, y), (0, 255, 0), 2)
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.imshow(win, clone)
        elif event == cv2.EVENT_LBUTTONUP and current is not None:
            sx, sy, _, _ = current
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            # Ensure some area
            if (x2-x1) > 5 and (y2-y1) > 5:
                rois.append([int(x1), int(y1), int(x2), int(y2)])
            current = None
            clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    print("Instructions: Drag mouse to draw the OBJECT SURFACE (Green Box).")
    print("Points will be detected INSIDE this box.")
    print("Press 'r' to remove last box, Enter or 'q' to finish.")
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(rois) > 0:
                rois.pop()
                clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for r in rois:
                    cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.imshow(win, clone)
        if key == 13 or key == ord('q'):
            break
    cv2.destroyWindow(win)
    return rois


def visualize_tracks(frames, tracks, out_path, fps, rois=None):
    N, T, _ = tracks.shape
    # Use a consistent color for all points on the surface to indicate they belong to the same object
    # Or keep random colors to distinguish them. Let's use random but similar hue? No, random is fine.
    rng = np.random.RandomState(42)
    colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]
    
    vis = []
    for t in range(T):
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        
        # Draw ROI if exists
        if rois:
            for r in rois:
                cv2.rectangle(img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                
        for i in range(N):
            pts = tracks[i]
            valid = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid) >= 2:
                vp = pts[:t+1][valid].astype(np.int32)
                # Draw trails
                # Limit trail length to keep it clean? Or full trail.
                # Let's draw full trail but thinner
                for k in range(len(vp)-1):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k+1]), colors[i], 1)
            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                cv2.circle(img, (x,y), 3, colors[i], -1)
        vis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    iio.imwrite(out_path, np.array(vis), plugin="FFMPEG", fps=fps, macro_block_size=1)
    return out_path


def save_tracks(out_dir, base, tracks, fps):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, base + "_surface_tracks.npz")
    np.savez_compressed(npz_path, tracks=tracks, fps=fps)
    csv_path = os.path.join(out_dir, base + "_surface_tracks.csv")
    import csv
    N, T, _ = tracks.shape
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(["id","frame","x","y"])
        for i in range(N):
            for t in range(T):
                x = tracks[i,t,0]
                y = tracks[i,t,1]
                if np.isnan(x) or np.isnan(y):
                    w.writerow([i, t, "", ""])
                else:
                    w.writerow([i, t, float(x), float(y)])
    return npz_path, csv_path


def save_averaged_waveform(out_dir, base, tracks, fps):
    """
    Calculate and save the averaged waveform of the object.
    """
    os.makedirs(out_dir, exist_ok=True)
    N, T, _ = tracks.shape
    times = np.arange(T) / float(fps)
    
    # Calculate relative displacements for all tracks
    # (T, N)
    displacements_y = np.full((T, N), np.nan, dtype=np.float32)
    displacements_x = np.full((T, N), np.nan, dtype=np.float32)
    
    for i in range(N):
        pts = tracks[i]
        valid = ~np.isnan(pts[:,1])
        if valid.any():
            first_idx = np.where(valid)[0][0]
            y0 = pts[first_idx, 1]
            x0 = pts[first_idx, 0]
            displacements_y[valid, i] = pts[valid, 1] - y0
            displacements_x[valid, i] = pts[valid, 0] - x0
            
    # Calculate Mean and Median across the N points for each time step
    # Use nanmean/nanmedian to ignore lost points
    mean_y = np.nanmean(displacements_y, axis=1)
    median_y = np.nanmedian(displacements_y, axis=1)
    
    mean_x = np.nanmean(displacements_x, axis=1)
    median_x = np.nanmedian(displacements_x, axis=1)
    
    # Save to CSV
    csv_path = os.path.join(out_dir, base + "_surface_waveform.csv")
    
    # Columns: time, mean_y, median_y, mean_x, median_x, [individual_y...]
    # We also save individual traces for reference
    
    header = ["time", "mean_y", "median_y", "mean_x", "median_x"] + [f"pt{i}_y" for i in range(N)]
    
    # Stack data
    # (T, 5 + N)
    data_list = [times, mean_y, median_y, mean_x, median_x]
    for i in range(N):
        data_list.append(displacements_y[:, i])
        
    data = np.column_stack(data_list)
    
    fmt = ["%.6f"] * data.shape[1]
    np.savetxt(csv_path, data, delimiter=",", header=",".join(header), comments='', fmt=fmt)
    
    return csv_path


def main():
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        if not os.path.exists(inp):
            candidate = os.path.join(INPUT_DIR, inp)
            if os.path.exists(candidate):
                inp = candidate
    else:
        inp = os.path.join(INPUT_DIR, DEFAULT_VIDEO)

    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model on", device)
    model = load_model(device)

    print("Reading frames from", inp)
    frames, fps = read_frames(inp)
    T = len(frames)
    H, W = frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")
    
    frames_input = frames

    # ROI handling: Select Object Surface
    base = os.path.splitext(os.path.basename(inp))[0]
    out_dir = os.path.join(out, f"{base}_output_surface")
    os.makedirs(out_dir, exist_ok=True)
    
    rois, roi_path = load_surface_rois(base)
    if rois is None:
        can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
        if can_interactive:
            try:
                print("No Surface ROI file found. Please select the OBJECT SURFACE (Green Box).")
                first = frames[0].copy()
                sel = select_surface_rois_interactive(first)
                if sel and len(sel) > 0:
                    save_surface_rois(roi_path, sel)
                    rois = sel
                else:
                    print("No ROIs selected. Will use whole image (not recommended for surface tracking).")
                    rois = []
            except Exception as e:
                print("Interactive selection failed:", e)
                rois = []
        else:
            # headless: save first frame for external annotation and abort
            save_path = os.path.join(out_dir, base + "_firstframe.png")
            # frames is RGB, convert to BGR for opencv save
            cv2.imwrite(save_path, cv2.cvtColor(frames[0], cv2.COLOR_RGB2BGR))
            print(f"当前环境无法交互选择 ROI（可能无显示）。已把第一帧保存到: {save_path}")
            print(f"请在本地打开该图片，使用图像工具标记要追踪的物体表面矩形并把结果保存为 JSON 至: {roi_path}")
            print("JSON 格式示例: [[x1,y1,x2,y2],...]。脚本退出以便你准备 ROI 文件（再次运行时会加载）。")
            sys.exit(1)
    else:
        print(f"Loaded Surface ROI file: {roi_path}, rectangles={len(rois)}")

    # Build mask for inclusion
    # Default: False (exclude everything)
    # If ROIs exist, set those regions to True
    if rois and len(rois) > 0:
        Hf, Wf = frames[0].shape[0], frames[0].shape[1]
        allowed_mask = np.zeros((Hf, Wf), dtype=bool)
        for r in rois:
            x1, y1, x2, y2 = map(int, r)
            x1 = max(0, min(Wf-1, x1)); x2 = max(0, min(Wf-1, x2))
            y1 = max(0, min(Hf-1, y1)); y2 = max(0, min(Hf-1, y2))
            allowed_mask[y1:y2+1, x1:x2+1] = True
        print(f"Surface mask created. Allowed pixel ratio: {allowed_mask.mean():.3f}")
    else:
        print("No Surface ROIs. Using full image.")
        allowed_mask = np.ones((frames[0].shape[0], frames[0].shape[1]), dtype=bool)

    # Generate a dense grid of points inside the ROI (every GRID_SPACING pixels)
    Hf, Wf = frames[0].shape[0], frames[0].shape[1]
    ys, xs = np.where(allowed_mask)
    if ys.size == 0:
        print("ROI mask contains no valid pixels. Exiting.")
        sys.exit(1)

    # Build full grid bounding box for efficiency
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())

    grid_y = np.arange(y_min, y_max + 1, GRID_SPACING)
    grid_x = np.arange(x_min, x_max + 1, GRID_SPACING)
    gx, gy = np.meshgrid(grid_x, grid_y)
    grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M,2) in (x,y)

    # Filter to keep only points inside allowed_mask
    keep = []
    Hmask, Wmask = allowed_mask.shape
    for x, y in grid_pts:
        xi, yi = int(x), int(y)
        if 0 <= yi < Hmask and 0 <= xi < Wmask and allowed_mask[yi, xi]:
            keep.append([float(x), float(y)])
    init_pts = np.array(keep, dtype=np.float32)

    if init_pts.shape[0] == 0:
        print("No grid points found inside ROI. Try enlarging ROI or increasing GRID_SPACING.")
        sys.exit(1)

    # If there are too many points, subsample deterministically to MAX_GRID_POINTS
    num_pts = init_pts.shape[0]
    if num_pts > MAX_GRID_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(num_pts, size=MAX_GRID_POINTS, replace=False)
        idx = np.sort(idx)
        init_pts = init_pts[idx]
        print(f"Grid produced {num_pts} points; subsampled to {init_pts.shape[0]} to avoid OOM.")
    else:
        print(f"Detected {init_pts.shape[0]} grid points on the object surface.")

    # Chunking logic
    # Keep chunk sizes smaller for high-res video to reduce memory usage
    CHUNK_LEN = 40
    OVERLAP = 10
    
    # Prepare initial queries
    q = np.zeros((1, init_pts.shape[0], 3), dtype=np.float32)
    q[0, :, 0] = 0
    q[0, :, 1:] = init_pts[:, :2]
    current_queries = torch.from_numpy(q).to(device)

    print(f"Processing video in chunks...")
    start_frame = 0
    stitched_tracks = None 

    while start_frame < T - 1:
        end_frame = min(start_frame + CHUNK_LEN, T)
        chunk_frames = frames_input[start_frame:end_frame]
        chunk_tensor = torch.tensor(chunk_frames).permute(0, 3, 1, 2)[None].float().to(device)

        print(f"  Chunk: {start_frame} to {end_frame}")

        with torch.no_grad():
            torch.cuda.empty_cache()
            if current_queries is not None:
                current_queries[:, :, 0] = 0
            pred_tracks_chunk, _ = model(chunk_tensor, queries=current_queries)

        chunk_tracks_np = pred_tracks_chunk.cpu().numpy()
        
        if stitched_tracks is None:
            stitched_tracks = chunk_tracks_np
        else:
            n_overlap = stitched_tracks.shape[1] - start_frame
            if n_overlap > 0:
                alpha = np.linspace(0, 1, n_overlap).reshape(1, -1, 1, 1)
                overlap_old = stitched_tracks[:, -n_overlap:, :, :]
                overlap_new = chunk_tracks_np[:, :n_overlap, :, :]
                blended = (1 - alpha) * overlap_old + alpha * overlap_new
                stitched_tracks[:, -n_overlap:, :, :] = blended
                stitched_tracks = np.concatenate([stitched_tracks, chunk_tracks_np[:, n_overlap:, :, :]], axis=1)
            else:
                stitched_tracks = np.concatenate([stitched_tracks, chunk_tracks_np], axis=1)

        if end_frame == T:
            break

        next_start_frame = end_frame - OVERLAP
        if next_start_frame <= start_frame: next_start_frame = start_frame + 1
        if next_start_frame >= T - 1: break
            
        query_idx = (next_start_frame - start_frame)
        last_pos = chunk_tracks_np[0, query_idx, :, :]
        new_q = torch.zeros((1, last_pos.shape[0], 3), device=device)
        new_q[0, :, 0] = 0
        new_q[0, :, 1:] = torch.from_numpy(last_pos).to(device)
        current_queries = new_q
        start_frame = next_start_frame

    pred_tracks_np = stitched_tracks
    raw_tracks = to_pixel_tracks(pred_tracks_np, (H, W))
    tracks = raw_tracks
    print("Final tracks shape:", tracks.shape)

    # Save raw tracks
    npz_path, csv_path = save_tracks(out_dir, base, tracks, fps)
    print("Saved tracks:", npz_path)

    # Visualize
    vis_path = os.path.join(out_dir, f"tracked_surface_{base}.mp4")
    visualize_tracks(frames, tracks, vis_path, fps, rois)
    print("Visualization saved:", vis_path)

    # Save Averaged Waveform
    wpath = save_averaged_waveform(out_dir, base, tracks, fps)
    print("Averaged Waveform saved:", wpath)
    print("Done. The 'mean_y' column in the waveform CSV represents the object's vibration.")

if __name__ == '__main__':
    main()
