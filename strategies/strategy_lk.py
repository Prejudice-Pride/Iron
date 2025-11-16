from .common import lk_track_from_points, visualize_and_save, save_trajectories
import os

def run(frames, fps, init_pts, out_dir, video_name):
    os.makedirs(out_dir, exist_ok=True)
    trajectories_lk, p0 = lk_track_from_points(frames, init_pts)
    if trajectories_lk.shape[0] > 0:
        out_vis = os.path.join(out_dir, f"lk_tracked_{video_name}")
        # ensure .mp4 extension
        if not out_vis.lower().endswith('.mp4'):
            out_vis = out_vis + '.mp4'
        visualize_and_save(frames, trajectories_lk, out_vis, fps)
        npz, csv = save_trajectories(out_dir, video_name + "_lk", trajectories_lk, fps)
        return trajectories_lk, out_vis, npz, csv
    return trajectories_lk, None, None, None
