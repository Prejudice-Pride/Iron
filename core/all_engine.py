import os
import sys
import torch
import numpy as np
import cv2

# Ensure AllTracker modules can be imported
# This file is in core/, project root is ../
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ALLTRACKER_DIR = os.path.join(PROJECT_ROOT, "alltracker")
if ALLTRACKER_DIR not in sys.path and os.path.exists(ALLTRACKER_DIR):
    sys.path.insert(0, ALLTRACKER_DIR)

try:
    from nets.alltracker import Net
    import utils.saveload
    import utils.basic
except ImportError:
    # If the above path hack didn't work, let the caller handle it
    pass

class AllTrackingEngine:
    def __init__(self, device=None, window_len=16, tiny=False, ckpt_path=None):
        """
        Initialize AllTracker Engine.
        注意：window_len 必须为 16 以匹配官方预训练权重。
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.window_len = window_len
        self.tiny = tiny
        
        print(f"Initializing AllTracker on {self.device} (tiny={tiny}, window_len={window_len})")
        
        # 1. Initialize architecture
        if tiny:
            self.model = Net(window_len, use_basicencoder=True, no_split=True)
        else:
            self.model = Net(window_len)

        # 2. Load weights
        if ckpt_path and os.path.exists(ckpt_path):
            _ = utils.saveload.load(None, ckpt_path, self.model, weights_only=False)
            print(f"Loaded weights from {ckpt_path}")
        else:
            # Fallback to downloading or using hub
            url_prefix = "https://hf-mirror.com/aharley/alltracker/resolve/main/"
            url = f"{url_prefix}alltracker_tiny.pth" if tiny else f"{url_prefix}alltracker.pth"
            print(f"Downloading/Loading weights from {url}")
            state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
            self.model.load_state_dict(state_dict['model'], strict=True)
            
        self.model.to(self.device).eval()

    def track_video(self, frames, points, inference_iters=4, image_size=512):
        """
        Track specified sparse points by sampling the dense flow field.
        frames: (T, H, W, 3) numpy array (RGB)
        points: (N, 2) numpy array of initial points (x, y)
        Returns: (N, T, 2) tracks, (N, T) visibilities
        """
        T_orig, H_orig, W_orig, _ = frames.shape
        N = points.shape[0]

        # 1. Preprocessing and Scaling
        torch.cuda.empty_cache() # Clear cache before processing
        # AllTracker works best at high resolution but needs divisible by 8 padding
        scale = min(image_size / H_orig, image_size / W_orig)
        H_new, W_new = int(H_orig * scale), int(W_orig * scale)
        H_new, W_new = (H_new // 8) * 8, (W_new // 8) * 8
        
        print(f"Processing video ({T_orig} frames) at {W_new}x{H_new} resolution...")
        
        resized_frames = [cv2.resize(f, (W_new, H_new), interpolation=cv2.INTER_LINEAR) for f in frames]
        rgbs = torch.from_numpy(np.stack(resized_frames)).permute(0, 3, 1, 2).float() # (T, 3, H, W)
        rgbs = rgbs.unsqueeze(0).to(self.device) # (1, T, 3, H, W)

        # Scale query points to the processing resolution
        points_scaled = points.copy().astype(np.float32)
        points_scaled[:, 0] *= (W_new / W_orig)
        points_scaled[:, 1] *= (H_new / H_orig)
        
        # 2. Inference (Compute Dense Flow)
        with torch.no_grad():
            # forward_sliding handles multi-frame stitching and high-resolution windows
            # Output flows: (1, T, 2, H, W) relative to anchor frame
            # Output visconfs: (1, T, 2, H, W) 
            full_flows, full_visconfs, _, _ = self.model.forward_sliding(
                rgbs, iters=inference_iters, is_training=False
            )
            
            # 3. Reconstruct Absolute Trajectories (Flow + Grid)
            # Create a coordinate grid for the processing resolution
            grid_xy = utils.basic.gridcloud2d(1, H_new, W_new, norm=False, device=self.device).float()
            grid_xy = grid_xy.permute(0, 2, 1).reshape(1, 1, 2, H_new, W_new)
            
            # Compute trajectory map: B, T, 2, H, W
            # Move flows to device if they were detached to CPU
            traj_maps = full_flows.to(self.device) + grid_xy 

            # 4. Dense-to-Sparse Sampling
            # Round and clip points to valid pixel indices
            px = points_scaled[:, 0].round().clip(0, W_new - 1).astype(np.int64)
            py = points_scaled[:, 1].round().clip(0, H_new - 1).astype(np.int64)
            
            # Sample from the T-dimensional flow map at the N specified coordinates
            # Result: (T, 2, N)
            sampled_trajs = traj_maps[0, :, :, py, px] 
            # Sample visibility/confidence: (T, 2, N), channel 0 is usually visibility
            sampled_vis = full_visconfs[0, :, 0, py, px] 
            
            # 5. Post-process and Map back to original resolution
            # Transpose to output shape: (N, T, 2)
            pred_tracks = sampled_trajs.permute(2, 0, 1).cpu().numpy()
            pred_tracks[:, :, 0] *= (W_orig / W_new)
            pred_tracks[:, :, 1] *= (H_orig / H_new)
            
            # Transpose to output shape: (N, T)
            pred_vis = sampled_vis.transpose(0, 1).cpu().numpy()

        return pred_tracks, pred_vis
