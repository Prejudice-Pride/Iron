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


INPUT_DIR = "input_videos"
DEFAULT_VIDEO = "sample.mp4"
OUTPUT_DIR = "output"
ROI_DIR = "rois"
CORNER_DIR = "corner_points"
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
GRID_SIZE = 10
MODEL_TYPE = "cotracker3_offline"
CKPT_REL = os.path.join("checkpoints", "scaled_offline.pth")


def load_model(device):
    hubconf_path = os.path.join(REPO_DIR, "hubconf.py")
    spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
    hubconf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hubconf)
    builder = getattr(hubconf, MODEL_TYPE)
    model = builder(pretrained=False).to(device)
    ckpt_path = os.path.join(REPO_DIR, CKPT_REL)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"Loading weights from {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict):
        if "model" in state:
            state_dict = state["model"]
        elif "state_dict" in state:
            state_dict = state["state_dict"]
        else:
            state_dict = state
    else:
        state_dict = state
    # strip module.
    new_state = {}
    for k, v in state_dict.items():
        nk = k[len("module.") :] if k.startswith("module.") else k
        new_state[nk] = v
    model.load_state_dict(new_state, strict=False)
    model.eval()
    return model


def read_frames(path, target_res=None, mb=16, pad_value=120):
    frames = iio.imread(path, plugin="FFMPEG")
    frames = np.array(frames)
    if target_res is not None:
        frames = np.array([cv2.resize(f, target_res) for f in frames])
    H, W = frames.shape[1], frames.shape[2]
    pad_h = ((H + mb - 1) // mb) * mb
    pad_w = ((W + mb - 1) // mb) * mb
    if pad_h != H or pad_w != W:
        pad_bottom = pad_h - H
        pad_right = pad_w - W
        frames = np.array([cv2.copyMakeBorder(f, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_value) for f in frames])
        print(f"Padded frames from ({W},{H}) to ({pad_w},{pad_h})")
    # fps
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return frames, fps


def to_pixel_tracks(pred_tracks_np, frame_shape):
    # pred_tracks_np may be (1,T,N,2) or (T,N,2) or (N,T,2)
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
    # convert normalized to pixels
    H, W = frame_shape
    if np.nanmax(tracks) <= 1.01:
        tracks[..., 0] = tracks[..., 0] * W
        tracks[..., 1] = tracks[..., 1] * H
    return tracks


def detect_corners_firstframe(frames, max_corners=200, quality=0.01, min_dist=7, block=7, valid_mask=None):
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
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
    clone = img.copy()
    rois = []
    current = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and current is not None:
            sx, sy, _, _ = current
            clone = img.copy()
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
            clone = img.copy()
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
                clone = img.copy()
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
        img = frames[t].copy()
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
        vis.append(img)
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


def main():
    inp = sys.argv[1] if len(sys.argv) > 1 else os.path.join(INPUT_DIR, DEFAULT_VIDEO)
    out = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_DIR
    os.makedirs(out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading model:", MODEL_TYPE, "on", device)
    model = load_model(device)

    print("Reading frames from", inp)
    frames, fps = read_frames(inp)
    T = len(frames)
    H, W = frames.shape[1], frames.shape[2]
    print(f"Frames: T={T}, HxW={W}x{H}, fps={fps}")

    # prepare tensor (1, T, C, H, W)
    video_tensor = torch.tensor(frames).permute(0,3,1,2)[None].float().to(device)

    # ROI handling: check for existing ROI file; if missing, ask user to draw exclusion rectangles
    base = os.path.splitext(os.path.basename(inp))[0]
    # per-video output directory under the main output folder (match cotracker_seismic.py)
    out_dir = os.path.join(out, f"{base}_output_manual")
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

    # --- NEW: corner points handling (prefer loading from corner_points/, else interactive selection) ---
    corner_pts, corner_path = load_corner_points_for_video(base)
    if corner_pts is not None:
        print(f"Loaded corner points from: {corner_path}, count={len(corner_pts)}")
        init_pts = np.array(corner_pts, dtype=np.float32)
        # draw preview and save
        preview = frames[0].copy()
        for x,y in init_pts:
            cv2.circle(preview, (int(round(x)), int(round(y))), 4, (0,255,0), -1)
        preview_path = os.path.join(out_dir, base + "_corners_preview.png")
        cv2.imwrite(preview_path, preview)
        print(f"Saved corner preview: {preview_path}")
    else:
        can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
        if can_interactive:
            try:
                print("No corner file found. 请在弹出的窗口中用鼠标点击要追踪的角点：左键添加，按 r 删除最后一个，按 Enter 或 q 完成。")
                first = frames[0].copy()
                sel_pts = select_points_interactive(first)
                if sel_pts and len(sel_pts) > 0:
                    saved = save_corner_points(os.path.join(CORNER_DIR, base + "_corners.json"), sel_pts)
                    if saved:
                        print(f"Saved corner points: {os.path.join(CORNER_DIR, base + '_corners.json')}")
                    init_pts = np.array(sel_pts, dtype=np.float32)
                else:
                    print("No points selected; exiting.")
                    raise SystemExit(1)
            except Exception as e:
                print("交互式角点选择失败：", e)
                raise SystemExit(1)
        else:
            # headless: save first frame for external annotation and abort
            save_path = os.path.join(out_dir, base + "_firstframe_for_corners.png")
            cv2.imwrite(save_path, frames[0])
            print(f"当前环境无法交互选择角点（可能无显示）。已把第一帧保存到: {save_path}")
            print(f"请在本地打开该图片，使用图像工具标记要追踪的角点并把结果保存为 JSON 至: {os.path.join(CORNER_DIR, base + '_corners.json')}")
            print("JSON 格式示例: [[x1,y1],[x2,y2],...]。脚本退出以便你准备 corner file（再次运行时会加载）。")
            raise SystemExit(1)

    # filter out points inside exclusion ROIs (if any)
    if rois and len(rois) > 0 and init_pts.shape[0] > 0:
        keep = []
        for x,y in init_pts:
            if not point_in_rois((x,y), rois):
                keep.append([x,y])
        init_pts = np.array(keep, dtype=np.float32)
    print(f"Final corner points to track (after ROI exclusion): {init_pts.shape[0]}")

    with torch.no_grad():
        if init_pts.shape[0] > 0:
            # build queries tensor shape (B=1, N, 3) with (t, x, y)
            q = np.zeros((1, init_pts.shape[0], 3), dtype=np.float32)
            q[0, :, 0] = 0  # start frame index
            q[0, :, 1:] = init_pts[:, :2]
            queries = torch.from_numpy(q).to(device)
            pred_tracks, pred_vis = model(video_tensor, queries=queries)
        else:
            print("No corner points provided, falling back to grid-based queries")
            pred_tracks, pred_vis = model(video_tensor, grid_size=GRID_SIZE)

    pred_tracks_np = pred_tracks.cpu().numpy()
    print("Model output shape:", pred_tracks_np.shape)

    raw_tracks = to_pixel_tracks(pred_tracks_np, (H, W))
    print("Raw cotracker tracks shape (N_cot,T,2):", raw_tracks.shape)
    # If we had user-specified init_pts, map & align cotracker tracks to those points
    if 'init_pts' in locals() and init_pts is not None and init_pts.shape[0] > 0:
        aligned_tracks, stats = map_and_align_tracks(raw_tracks, init_pts, (H, W), thresh=50.0)
        print(f"Mapping cotracker -> init_pts: matched {stats['num_matched']}/{init_pts.shape[0]} (thresh={stats['thresh']})")
        tracks = aligned_tracks
    else:
        tracks = raw_tracks
    print("Final tracks shape (N_query,T,2):", tracks.shape)

    # Save raw npz + csv (use per-video out_dir)
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
