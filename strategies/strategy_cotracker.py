from .common import map_cotracker_to_init_points, visualize_and_save, save_trajectories
import os
import numpy as np
import json

def run(frames, fps, init_pts, pred_tracks_np, frame_shape, out_dir, video_name, thresh=15.0):
    os.makedirs(out_dir, exist_ok=True)
    # returns mapped (N_init, T, 2) and stats dict
    cot_mapped, stats = map_cotracker_to_init_points(pred_tracks_np, init_pts, len(frames), frame_shape, thresh=thresh)
    # save stats for inspection
    stats_path = os.path.join(out_dir, f"{video_name}_cotracker_match_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
    except Exception:
        pass

    if cot_mapped is not None and stats.get("num_matched", 0) > 0:
        out_vis = os.path.join(out_dir, f"cotracker_tracked_{video_name}")
        if not out_vis.lower().endswith('.mp4'):
            out_vis = out_vis + '.mp4'
        # visualize the delta-aligned mapped tracks
        visualize_and_save(frames, cot_mapped, out_vis, fps)
        npz, csv = save_trajectories(out_dir, video_name + "_cotracker", cot_mapped, fps)
        return cot_mapped, out_vis, npz, csv

    # fallback: save raw pred_tracks for debugging
    raw_npz = os.path.join(out_dir, f"{video_name}_cotracker_output.npz")
    np.savez_compressed(raw_npz, pred_tracks=pred_tracks_np, fps=fps)
    return None, None, raw_npz, None
