import os
import numpy as np
import cv2
import imageio.v3 as iio


def lk_track_from_points(frames, p0, lk_win=(21,21), lk_maxlevel=3):
    """Track given initial points p0 across frames using pyramidal LK.
    Returns trajectories (N, T, 2) and initial points p0.
    """
    if p0.size == 0:
        return np.empty((0, len(frames), 2)), p0
    num_pts = p0.shape[0]
    T = len(frames)
    trajectories = np.full((num_pts, T, 2), np.nan, dtype=np.float32)
    trajectories[:,0,:] = p0
    lk_params = dict(winSize=lk_win, maxLevel=lk_maxlevel,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    active_idxs = np.arange(num_pts)
    prev_pts = p0.reshape(-1,1,2).astype(np.float32)
    for t in range(1, T):
        curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_BGR2GRAY)
        if active_idxs.size == 0:
            break
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None, **lk_params)
        if next_pts is None:
            break
        for i_idx, ai in enumerate(active_idxs):
            if status[i_idx,0] == 1:
                trajectories[ai, t] = next_pts[i_idx,0]
        success_mask = (status.reshape(-1) == 1)
        active_idxs = active_idxs[success_mask]
        if active_idxs.size == 0:
            break
        prev_pts = next_pts[success_mask].reshape(-1,1,2).astype(np.float32)
        prev_gray = curr_gray.copy()
    return trajectories, p0


def visualize_and_save(frames, tracks, out_path, fps, colors=None):
    """Visualize tracks (N, T, 2) on frames and save video to out_path."""
    if tracks.size == 0:
        return None
    N, T, _ = tracks.shape
    if colors is None:
        rng = np.random.RandomState(42)
        colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]
    vis_frames = []
    for t in range(T):
        img = frames[t].copy()
        for i in range(N):
            pts = tracks[i]
            valid_idx = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid_idx) >= 2:
                valid_pts = pts[:t+1][valid_idx].astype(np.int32)
                for k in range(len(valid_pts)-1):
                    cv2.line(img, tuple(valid_pts[k]), tuple(valid_pts[k+1]), colors[i], 2)
            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                cv2.circle(img, (x,y), 3, colors[i], -1)
        vis_frames.append(img)
    iio.imwrite(out_path, np.array(vis_frames), plugin="FFMPEG", fps=fps)
    return out_path


def save_trajectories(out_dir, base_name, tracks, fps):
    """Save tracks to npz and CSV. CSV format: id,frame,x,y"""
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, base_name + ".npz")
    np.savez_compressed(npz_path, tracks=tracks, fps=fps)
    csv_path = os.path.join(out_dir, base_name + ".csv")
    import csv as _csv
    N, T, _ = tracks.shape
    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow(["id", "frame", "x", "y"])
        for i in range(N):
            for t in range(T):
                x = tracks[i, t, 0]
                y = tracks[i, t, 1]
                if np.isnan(x) or np.isnan(y):
                    writer.writerow([i, t, "", ""])
                else:
                    writer.writerow([i, t, float(x), float(y)])
    return npz_path, csv_path


def map_cotracker_to_init_points(pred_tracks_np, init_points, T, frame_shape, thresh=15.0):
    """Map cotracker tracks (various layouts) to given init_points with one-to-one greedy matching.
    Returns mapped array (N_init, T, 2) aligned to init_points; unmatched rows are NaN.
    """
    arr = pred_tracks_np
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    cot_tracks = None
    if arr.ndim == 3 and arr.shape[0] == T:
        cot_tracks = np.transpose(arr, (1, 0, 2))
    elif arr.ndim == 3 and arr.shape[1] == T:
        cot_tracks = arr
    if cot_tracks is None or init_points.size == 0:
        return None
    H, W = frame_shape
    cot_tracks = cot_tracks.astype(np.float32)
    if np.nanmax(cot_tracks) <= 1.01:
        cot_tracks[..., 0] = cot_tracks[..., 0] * W
        cot_tracks[..., 1] = cot_tracks[..., 1] * H
    cot_first = cot_tracks[:, 0, :]
    lk_init = init_points.astype(np.float32)
    num_lk = lk_init.shape[0]
    num_cot = cot_first.shape[0]
    best_mapped = None
    # New robust implementation: filter NaNs, compute distances, greedy one-to-one match,
    # return mapped array aligned to init_points and a stats dict.
    arr = pred_tracks_np
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # -> (T, N, 2)
    # normalize to (N_cot, T, 2)
    cot_tracks = None
    if arr.ndim == 3 and arr.shape[0] == T:
        cot_tracks = np.transpose(arr, (1, 0, 2))  # (N_cot, T, 2)
    elif arr.ndim == 3 and arr.shape[1] == T:
        cot_tracks = arr  # (N_cot, T, 2)
    else:
        try:
            cot_tracks = arr.reshape(-1, T, 2)
        except Exception:
            return None

    H, W = frame_shape
    cot_tracks = cot_tracks.astype(np.float32)

    # If normalized in [0,1], convert to pixel coords
    if np.nanmax(cot_tracks) <= 1.01:
        cot_tracks[..., 0] = cot_tracks[..., 0] * W
        cot_tracks[..., 1] = cot_tracks[..., 1] * H

    cot_first = cot_tracks[:, 0, :]
    lk_init = init_points.astype(np.float32)
    N_init = lk_init.shape[0]
    N_cot = cot_first.shape[0]

    mapped = np.full((N_init, T, 2), np.nan, dtype=np.float32)
    stats = {"num_cot_total": int(N_cot)}

    # filter out cot entries whose first-frame is NaN
    valid_mask = ~np.isnan(cot_first).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    valid_first = cot_first[valid_mask]
    stats["num_cot_valid_first"] = int(valid_first.shape[0])

    if valid_first.shape[0] == 0 or N_init == 0:
        stats.update({"num_matched": 0, "matched_pairs": [], "dist_summary": None})
        return mapped, stats

    # compute distance matrix between init and valid cot first-frame positions
    dists = np.linalg.norm(lk_init[:, None, :] - valid_first[None, :, :], axis=2)  # (N_init, N_valid)

    # prepare list of pairs sorted by distance
    pairs = []
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            pairs.append((i, j, float(dists[i, j])))
    pairs.sort(key=lambda x: x[2])

    assigned_init = np.zeros(N_init, dtype=bool)
    assigned_cot = np.zeros(dists.shape[1], dtype=bool)
    matched_pairs = []
    for (i, j, dist) in pairs:
        if dist > thresh:
            break
        if (not assigned_init[i]) and (not assigned_cot[j]):
            assigned_init[i] = True
            assigned_cot[j] = True
            cot_idx = int(valid_idx[j])
            mapped[i, :, :] = cot_tracks[cot_idx]
            matched_pairs.append({"init_idx": int(i), "cot_idx": int(cot_idx), "dist": float(dist)})

    stats["num_matched"] = int(np.sum(assigned_init))
    stats["matched_pairs"] = matched_pairs
    flat = dists.flatten()
    stats["dist_summary"] = {
        "min": float(np.nanmin(flat)),
        "25%": float(np.nanpercentile(flat, 25)),
        "median": float(np.nanpercentile(flat, 50)),
        "75%": float(np.nanpercentile(flat, 75)),
        "max": float(np.nanmax(flat)),
    }

    # delta-align matched rows so mapped[i,0] will equal init_points[i]
    for m in matched_pairs:
        i = m["init_idx"]
        if np.isnan(mapped[i, 0, 0]) or np.isnan(mapped[i, 0, 1]):
            continue
        delta = mapped[i] - mapped[i, 0:1, :]
        mapped[i] = lk_init[i : i + 1, :] + delta

    return mapped, stats
    valid_cot_mask = ~np.isnan(cot_first).any(axis=1)
    valid_cot_idx = np.where(valid_cot_mask)[0]
    valid_cot_first = cot_first[valid_cot_mask]

    lk_init = init_points.astype(np.float32)
    N_init = lk_init.shape[0]
    mapped = np.full((N_init, T, 2), np.nan, dtype=np.float32)

    stats = {}
    if valid_cot_first.shape[0] == 0:
        stats['num_cot_valid_first'] = 0
        stats['num_matched'] = 0
        stats['dists'] = None
        return mapped, stats

    # compute distance matrix between init and valid cot_first
    dists = np.linalg.norm(lk_init[:, None, :] - valid_cot_first[None, :, :], axis=2)  # (N_init, N_valid_cot)
    # flatten pairs and sort by distance
    pairs = []
    for i in range(dists.shape[0]):
        for j in range(dists.shape[1]):
            pairs.append((i, j, float(dists[i,j])))
    pairs.sort(key=lambda x: x[2])
    assigned_init = np.zeros(N_init, dtype=bool)
    assigned_cot = np.zeros(dists.shape[1], dtype=bool)
    matched_pairs = []
    for (i, j, dist) in pairs:
        if dist > thresh:
            break
        if not assigned_init[i] and not assigned_cot[j]:
            assigned_init[i] = True
            assigned_cot[j] = True
            cot_idx = valid_cot_idx[j]
            mapped[i, :, :] = cot_tracks[cot_idx]
            matched_pairs.append((i, cot_idx, dist))

    # stats
    stats['num_cot_valid_first'] = int(valid_cot_first.shape[0])
    stats['num_matched'] = int(np.sum(assigned_init))
    stats['matched_pairs'] = matched_pairs
    d_flat = dists.flatten()
    stats['dist_summary'] = {
        'min': float(np.nanmin(d_flat)),
        '25%': float(np.nanpercentile(d_flat, 25)),
        'median': float(np.nanpercentile(d_flat, 50)),
        '75%': float(np.nanpercentile(d_flat, 75)),
        'max': float(np.nanmax(d_flat))
    }

    # optionally delta-align matched rows so mapped[i,0] == init_points[i]
    if do_align and stats['num_matched'] > 0:
        for (i, cot_idx, dist) in matched_pairs:
            cot_tr = mapped[i]  # (T,2)
            if np.isnan(cot_tr[0,0]) or np.isnan(cot_tr[0,1]):
                continue
            delta = cot_tr - cot_tr[0]  # (T,2)
            mapped[i] = lk_init[i:i+1, :] + delta

    return mapped, stats
