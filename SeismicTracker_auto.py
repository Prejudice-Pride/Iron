"""Run cotracker-only seismic extraction pipeline.

Usage: run from repo root. Defaults:
  input: input_videos/sample.mp4
  output: output/

Outputs saved to `output/`:
  - tracked_<video>.mp4  (visualization)
  - <video>_cotracker_tracks.npz  (tracks array and fps)
  - <video>_cotracker_tracks.csv  (id,frame,x,y)
  - <video>_waveforms.csv  (time series per track, relative Y displacement)
"""
import os
import sys
import json
import numpy as np
import imageio.v3 as iio
import cv2
import torch
import importlib.util
import torch.nn.functional as F

# 添加 co-tracker 路径到 sys.path
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from cotracker.predictor import CoTrackerPredictor

INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"
ROI_DIR = "rois"
CORNER_DIR = "corner_points"
GRID_SIZE = 10
CKPT_PATH = os.path.join(REPO_DIR, "checkpoints", "scaled_online.pth")


def load_model(device):
    if not os.path.exists(CKPT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")
    print(f"Loading weights from {CKPT_PATH}")
    # 使用官方封装，自动处理权重加载和 resize
    # offline=False 启用 Online 模型 (Sliding Window)，支持长视频
    # window_len=16 是 scaled_online.pth 权重的训练参数
    model = CoTrackerPredictor(checkpoint=CKPT_PATH, offline=False, window_len=16).to(device)
    return model


def read_frames(path):
    # 移除手动 resize 和 padding，直接读取原始帧
    frames = iio.imread(path, plugin="FFMPEG")
    # imageio 读取的是 RGB
    frames = np.array(frames)
    # fps
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return frames, fps


def to_pixel_tracks(pred_tracks_np, frame_shape):
    # CoTrackerPredictor returns tracks in pixel coordinates (if input is original resolution)
    # So we just need to handle shape
    arr = pred_tracks_np
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    # prefer (N,T,2)
    if arr.ndim == 3:
        # try detect orientation
        T = arr.shape[0]
        if arr.shape[0] <= 4 and arr.shape[1] > arr.shape[0]:
            # likely (N,T,2)
            tracks = arr
        else:
            # assume (T,N,2)
            tracks = np.transpose(arr, (1,0,2))
    else:
        raise RuntimeError("Unexpected pred_tracks shape: %s" % (arr.shape,))
    
    # No normalization needed as CoTrackerPredictor handles it
    return tracks


def detect_corners_firstframe(frames, max_corners=10, quality=0.01, min_dist=7, block=7, valid_mask=None):
    # frames is RGB, convert to GRAY
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
    p0 = cv2.goodFeaturesToTrack(gray0, maxCorners=max_corners, qualityLevel=quality, minDistance=min_dist, blockSize=block)
    if p0 is None:
        return np.empty((0,2), dtype=np.float32)
    pts = p0.reshape(-1,2).astype(np.float32)
    if valid_mask is not None and pts.size>0:
        h,w = valid_mask.shape
        keep = []
        for x,y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0<=yi<h and 0<=xi<w and valid_mask[yi, xi]:
                keep.append([x,y])
        if len(keep)==0:
            return np.empty((0,2), dtype=np.float32)
        return np.array(keep, dtype=np.float32)
    return pts


def load_rois_for_video(base_name):
    os.makedirs(ROI_DIR, exist_ok=True)
    path = os.path.join(ROI_DIR, base_name + "_rois.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                rois = json.load(f)
            # ensure ints
            rois = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
            return rois, path
        except Exception:
            return None, path
    return None, path


def save_rois(path, rois):
    try:
        with open(path, 'w') as f:
            json.dump(rois, f)
        return True
    except Exception:
        return False


def select_rois_interactive(img):
    """Let user draw multiple rectangular ROIs to exclude. Returns list of [x1,y1,x2,y2]."""
    win = "Select exclusion ROIs - drag to draw, press r to remove last, q/Enter to finish"
    # img is RGB, convert to BGR for OpenCV
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
            cv2.rectangle(clone, (sx, sy), (x, y), (0, 0, 255), 2)
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
            cv2.imshow(win, clone)
        elif event == cv2.EVENT_LBUTTONUP and current is not None:
            sx, sy, _, _ = current
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            rois.append([int(x1), int(y1), int(x2), int(y2)])
            current = None
            clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    print("Instructions: drag left mouse to draw exclusion rectangles. Press 'r' to remove last, Enter or 'q' to finish.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(rois) > 0:
                rois.pop()
                clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                for r in rois:
                    cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
                cv2.imshow(win, clone)
        if key == 13 or key == ord('q'):
            break
    cv2.destroyWindow(win)
    return rois


def point_in_rois(pt, rois):
    x, y = float(pt[0]), float(pt[1])
    for r in rois:
        if x >= r[0] and x <= r[2] and y >= r[1] and y <= r[3]:
            return True
    return False

# --- NEW: corner-points load/save and interactive selection ---
def load_corner_points_for_video(base_name):
    os.makedirs(CORNER_DIR, exist_ok=True)
    path = os.path.join(CORNER_DIR, base_name + "_corners.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                pts = json.load(f)
            # ensure numeric list pairs
            pts = [[float(p[0]), float(p[1])] for p in pts]
            return pts, path
        except Exception:
            return None, path
    return None, path

def save_corner_points(path, pts):
    try:
        with open(path, 'w') as f:
            json.dump([[float(p[0]), float(p[1])] for p in pts], f)
        return True
    except Exception:
        return False

def select_points_interactive(img):
    """Let user click points on image. Left-click add, 'r' remove last, Enter/q finish. Returns [[x,y],...]."""
    win = "Select corner points - left click to add, r remove last, Enter/q finish"
    clone = img.copy()
    pts = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x), int(y)])
            clone = img.copy()
            for p in pts:
                cv2.circle(clone, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    print("Instructions: left-click to add points. Press 'r' to remove last, Enter or 'q' to finish.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(pts) > 0:
                pts.pop()
                clone = img.copy()
                for p in pts:
                    cv2.circle(clone, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
                cv2.imshow(win, clone)
        if key == 13 or key == ord('q'):
            break
    cv2.destroyWindow(win)
    return pts


def visualize_tracks(frames, tracks, out_path, fps):
    N, T, _ = tracks.shape
    rng = np.random.RandomState(42)
    colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]
    vis = []
    for t in range(T):
        # frames is RGB, convert to BGR for OpenCV drawing
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        for i in range(N):
            pts = tracks[i]
            valid = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid) >= 2:
                vp = pts[:t+1][valid].astype(np.int32)
                for k in range(len(vp)-1):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k+1]), colors[i], 2)
            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                cv2.circle(img, (x,y), 3, colors[i], -1)
        # Convert back to RGB for imageio saving
        vis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    iio.imwrite(out_path, np.array(vis), plugin="FFMPEG", fps=fps)
    return out_path


def save_tracks(out_dir, base, tracks, fps):
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, base + "_cotracker_tracks.npz")
    np.savez_compressed(npz_path, tracks=tracks, fps=fps)
    csv_path = os.path.join(out_dir, base + "_cotracker_tracks.csv")
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


def save_waveforms(out_dir, base, tracks, fps, min_valid_ratio=0.0):
    # export relative Y displacement time series for each track as columns
    os.makedirs(out_dir, exist_ok=True)
    N, T, _ = tracks.shape
    # time vector
    times = np.arange(T) / float(fps)
    # build matrix (T, N)
    mat = np.full((T, N), np.nan, dtype=np.float32)
    for i in range(N):
        pts = tracks[i]
        valid = ~np.isnan(pts[:,1])
        if valid.sum() / float(T) < min_valid_ratio:
            continue
        # subtract initial valid y to get relative displacement
        if valid.any():
            first_idx = np.where(valid)[0][0]
            y0 = pts[first_idx,1]
            mat[valid, i] = pts[valid,1] - y0
    # save csv with header pt{i}
    csv_path = os.path.join(out_dir, base + "_waveforms.csv")
    header = [f"pt{i}_y_rel" for i in range(N)]
    data = np.hstack([times.reshape(-1,1), mat])
    # save time + columns
    fmt = ["%.6f"] * (1 + N)
    np.savetxt(csv_path, data, delimiter=",", header=",".join(["time"]+header), comments='', fmt=fmt)
    return csv_path


def map_and_align_tracks(cot_tracks, init_pts, frame_shape, thresh=50.0):
    """
    Map cotracker pixel tracks (N_cot, T, 2) to init_pts (M,2).
    Greedy one-to-one match on first-frame positions (exclude cot rows with NaN at t=0).
    For each matched pair: aligned_tr[t] = init_pts[i] + (cot_tr[t] - cot_tr[0])
    Returns aligned_tracks shape (M, T, 2) and stats dict.
    """
    import numpy as np
    M = init_pts.shape[0]
    T = cot_tracks.shape[1]
    aligned = np.full((M, T, 2), np.nan, dtype=np.float32)

    cot_first = cot_tracks[:, 0, :]  # (N_cot,2)
    valid_mask = ~np.isnan(cot_first).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return aligned, {"num_cot_valid_first": 0, "num_matched": 0, "matched_pairs": []}

    valid_first = cot_first[valid_mask]
    # compute distance matrix (M x N_valid)
    dists = np.linalg.norm(init_pts[:, None, :] - valid_first[None, :, :], axis=2)
    pairs = []
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            pairs.append((i, j, float(dists[i, j])))
    pairs.sort(key=lambda x: x[2])
    assigned_init = np.zeros(M, dtype=bool)
    assigned_cot = np.zeros(dists.shape[1], dtype=bool)
    matched = []
    for i, j, dist in pairs:
        if dist > thresh:
            break
        if (not assigned_init[i]) and (not assigned_cot[j]):
            assigned_init[i] = True
            assigned_cot[j] = True
            cot_idx = valid_idx[j]
            cot_tr = cot_tracks[cot_idx]  # (T,2)
            # delta alignment
            delta = cot_tr - cot_tr[0]
            aligned[i] = init_pts[i][None, :] + delta
            matched.append((int(i), int(cot_idx), float(dist)))

    stats = {
        "num_cot_valid_first": int(len(valid_idx)),
        "num_matched": int(np.sum(assigned_init)),
        "matched_pairs": matched,
        "thresh": float(thresh),
    }
    return aligned, stats


def run_model_custom(model_predictor, video, queries):
    # Custom forward pass to handle memory issues by reducing fmaps_chunk_size
    B, T, C, H, W = video.shape
    interp_shape = model_predictor.interp_shape
    
    # 1. Resize video
    video = video.reshape(B * T, C, H, W)
    video = F.interpolate(
        video, tuple(interp_shape), mode="bilinear", align_corners=True
    )
    video = video.reshape(B, T, 3, interp_shape[0], interp_shape[1])
    
    # 2. Resize queries
    queries = queries.clone()
    queries[:, :, 1:] *= queries.new_tensor(
        [
            (interp_shape[1] - 1) / (W - 1),
            (interp_shape[0] - 1) / (H - 1),
        ]
    )
    
    # 3. Forward with chunking
    # Access the underlying CoTrackerThreeOffline model
    inner_model = model_predictor.model
    
    # Reduce fmaps_chunk_size to save memory (default is 200)
    # If T is large, this helps avoid OOM during feature extraction
    tracks, visibilities, *_ = inner_model(
        video=video, queries=queries, iters=6, fmaps_chunk_size=50
    )
    
    # 4. Restore tracks
    tracks *= tracks.new_tensor(
        [(W - 1) / (interp_shape[1] - 1), (H - 1) / (interp_shape[0] - 1)]
    )
    
    return tracks, visibilities


def main():
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        # If file doesn't exist, try looking in INPUT_DIR
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

    # Limit frames to avoid OOM with offline model
    # prepare tensor (1, T, C, H, W)
    # frames is RGB (from imageio), CoTracker expects RGB float [0, 255]
    # 优化：不要一次性把所有帧转为 Tensor 并放到 GPU 上，这会占用大量显存
    # CoTrackerPredictor 内部会处理，但如果我们直接传 Tensor，它就全在显存里了
    # 我们可以先传 numpy array 或者 cpu tensor，让模型内部按需处理（虽然 Predictor 接口可能还是会转）
    # 最直接的优化：手动 resize 视频到模型输入尺寸，减少显存占用
    # CoTracker 默认输入尺寸通常是 (384, 512) 或类似，原始 1080p 太大了
    
    # 策略：为了不损失精度，我们保持原始分辨率，但采用分段处理（Chunking）
    # 这样可以避免一次性将整个高分辨率视频加载到显存中导致的 OOM
    frames_input = frames


    # ROI handling: check for existing ROI file; if missing, ask user to draw exclusion rectangles
    base = os.path.splitext(os.path.basename(inp))[0]
    # per-video output directory under the main output folder
    out_dir = os.path.join(out, f"{base}_output_auto")
    os.makedirs(out_dir, exist_ok=True)
    rois, roi_path = load_rois_for_video(base)
    if rois is None:
        # interactive available?
        can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
        if can_interactive:
            try:
                print("No ROI file found. 请在弹出的窗口中框选要排除的区域（红色矩形），按 Enter 或 q 完成。")
                first = frames[0].copy()
                sel = select_rois_interactive(first)
                if sel and len(sel) > 0:
                    saved = save_rois(roi_path, sel)
                    if saved:
                        print(f"Saved ROI file: {roi_path}")
                        rois = sel
                    else:
                        print("Warning: unable to save ROI file, continuing with in-memory ROIs.")
                        rois = sel
                else:
                    print("No ROIs selected; continuing without exclusions.")
                    rois = []
            except Exception as e:
                print("交互式 ROI 选择失败：", e)
                rois = []
        else:
            # headless: save first frame for external annotation and abort
            save_path = os.path.join(out_dir, base + "_firstframe.png")
            cv2.imwrite(save_path, frames[0])
            print(f"当前环境无法交互选择 ROI（可能无显示）。已把第一帧保存到: {save_path}")
            print(f"请在本地打开该图片，使用图像工具标记要排除的矩形并把结果保存为 JSON 至: {roi_path}")
            print("JSON 格式示例: [[x1,y1,x2,y2],[x1,y1,x2,y2],...]。脚本退出以便你准备 ROI 文件（再次运行时会加载）。")
            raise SystemExit(1)
    else:
        print(f"Loaded ROI file: {roi_path}, rectangles={len(rois)}")

    # Use Shi-Tomasi to select corners, excluding any pixels inside ROI exclusion boxes.
    # Build allowed mask: True = allowed, False = excluded (inside any ROI)
    Hf, Wf = frames[0].shape[0], frames[0].shape[1]
    allowed_mask = np.ones((Hf, Wf), dtype=bool)
    if rois and len(rois) > 0:
        for r in rois:
            x1, y1, x2, y2 = map(int, r)
            x1 = max(0, min(Wf-1, x1)); x2 = max(0, min(Wf-1, x2))
            y1 = max(0, min(Hf-1, y1)); y2 = max(0, min(Hf-1, y2))
            allowed_mask[y1:y2+1, x1:x2+1] = False
        print(f"Loaded ROI exclusions: {len(rois)} rectangles. Allowed pixel ratio: {allowed_mask.mean():.3f}")
    else:
        print("No ROI exclusions provided; selecting corners over full frame.")

    # Auto mode: always detect Shi-Tomasi corners (do not prefer saved corner file)
    # 分段处理后显存压力减小，恢复较大的 max_corners 以确保能检测到点
    init_pts = detect_corners_firstframe(frames, max_corners=200, quality=0.001, min_dist=6, block=7, valid_mask=allowed_mask)
    if init_pts is None:
        init_pts = np.empty((0,2), dtype=np.float32)
    print(f"Detected Shi-Tomasi initial points (after ROI exclusion): {init_pts.shape[0]}")

    # Chunking logic to avoid OOM
    CHUNK_LEN = 100  # Process 100 frames at a time
    all_tracks_list = []

    # Prepare initial queries
    if init_pts.shape[0] > 0:
        q = np.zeros((1, init_pts.shape[0], 3), dtype=np.float32)
        q[0, :, 0] = 0
        q[0, :, 1:] = init_pts[:, :2]
        current_queries = torch.from_numpy(q).to(device)
        use_grid = False
    else:
        current_queries = None
        use_grid = True
        print("No corner points provided, using grid-based queries for first chunk")

    print(f"Processing video in chunks of {CHUNK_LEN} frames...")

    start_frame = 0
    while start_frame < T - 1:
        end_frame = min(start_frame + CHUNK_LEN, T)
        chunk_frames = frames_input[start_frame:end_frame]
        # Create tensor for just this chunk
        chunk_tensor = torch.tensor(chunk_frames).permute(0, 3, 1, 2)[None].float().to(device)

        print(f"  Chunk: {start_frame} to {end_frame} (len={len(chunk_frames)})")

        with torch.no_grad():
            torch.cuda.empty_cache()
            if start_frame == 0 and use_grid:
                # First chunk with grid
                pred_tracks_chunk, pred_vis_chunk = model(chunk_tensor, grid_size=5)
                use_grid = False  # Switch to queries for next chunks
            else:
                # Subsequent chunks or first chunk with queries
                # Ensure queries time is 0 relative to chunk
                if current_queries is not None:
                    current_queries[:, :, 0] = 0
                pred_tracks_chunk, pred_vis_chunk = model(chunk_tensor, queries=current_queries)

        # Store results
        # pred_tracks_chunk: (1, len, N, 2)
        chunk_tracks_np = pred_tracks_chunk.cpu().numpy()

        if start_frame == 0:
            all_tracks_list.append(chunk_tracks_np)
        else:
            # Skip the first frame as it overlaps with previous chunk's last frame
            all_tracks_list.append(chunk_tracks_np[:, 1:])

        # Prepare for next chunk
        # Get last positions from this chunk to use as queries for next
        last_pos = pred_tracks_chunk[0, -1, :, :]  # (N, 2)

        # Create new queries
        new_q = torch.zeros((1, last_pos.shape[0], 3), device=device)
        new_q[0, :, 0] = 0
        new_q[0, :, 1:] = last_pos
        current_queries = new_q

        start_frame = end_frame - 1  # Overlap by 1 frame

    # Concatenate all tracks
    # all_tracks_list elements are (1, T_part, N, 2)
    pred_tracks_np = np.concatenate(all_tracks_list, axis=1)

    print("Model output shape:", pred_tracks_np.shape)

    raw_tracks = to_pixel_tracks(pred_tracks_np, (frames_input.shape[1], frames_input.shape[2]))


    print("Raw cotracker tracks shape (N_cot,T,2):", raw_tracks.shape)
    # If we had user-specified init_pts, map & align cotracker tracks to those points
    if 'init_pts' in locals() and init_pts is not None and init_pts.shape[0] > 0:
        aligned_tracks, stats = map_and_align_tracks(raw_tracks, init_pts, (H, W), thresh=50.0)
        print(f"Mapping cotracker -> init_pts: matched {stats['num_matched']}/{init_pts.shape[0]} (thresh={stats['thresh']})")
        tracks = aligned_tracks
    else:
        tracks = raw_tracks
    print("Final tracks shape (N_query,T,2):", tracks.shape)

    base = os.path.splitext(os.path.basename(inp))[0]
    # Save raw npz + csv
    npz_path, csv_path = save_tracks(out_dir, base, tracks, fps)
    print("Saved tracks:", npz_path, csv_path)

    # visualize
    vis_path = os.path.join(out_dir, f"tracked_{base}.mp4")
    print("Rendering visualization to", vis_path)
    visualize_tracks(frames, tracks, vis_path, fps)
    print("Visualization saved.")

    # save waveforms (relative Y displacement)
    wpath = save_waveforms(out_dir, base, tracks, fps, min_valid_ratio=0.0)
    print("Waveforms saved:", wpath)


if __name__ == '__main__':
    main()
