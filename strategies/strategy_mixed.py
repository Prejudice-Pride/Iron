from .common import save_trajectories, visualize_and_save
import numpy as np
import os

def run(frames, fps, init_pts, trajectories_lk, cot_mapped, out_dir, video_name):
    os.makedirs(out_dir, exist_ok=True)
    T = len(frames)
    if trajectories_lk is None or trajectories_lk.size == 0:
        # fallback to cot_mapped if available
        if cot_mapped is None:
            return None, None, None, None
        fused = cot_mapped.copy()
    else:
        if cot_mapped is None:
            fused = trajectories_lk.copy()
        else:
            N_init = init_pts.shape[0]
            fused_list = []
            for i in range(N_init):
                lk_tr = trajectories_lk[i] if i < trajectories_lk.shape[0] else np.full((T,2), np.nan, dtype=np.float32)
                cot_tr = cot_mapped[i] if i < cot_mapped.shape[0] else np.full((T,2), np.nan, dtype=np.float32)
                fused = np.full((T,2), np.nan, dtype=np.float32)
                for t in range(T):
                    if not np.isnan(lk_tr[t,0]):
                        fused[t] = lk_tr[t]
                    elif not np.isnan(cot_tr[t,0]):
                        fused[t] = cot_tr[t]
                if np.any(~np.isnan(fused)):
                    fused_list.append(fused)
            if len(fused_list) == 0:
                fused = np.empty((0, T, 2))
            else:
                fused = np.stack(fused_list, axis=0)
    if fused.size == 0:
        return None, None, None, None
    out_vis = os.path.join(out_dir, f"mixed_tracked_{video_name}")
    if not out_vis.lower().endswith('.mp4'):
        out_vis = out_vis + '.mp4'
    visualize_and_save(frames, fused, out_vis, fps)
    npz, csv = save_trajectories(out_dir, video_name + "_mixed", fused, fps)
    return fused, out_vis, npz, csv
