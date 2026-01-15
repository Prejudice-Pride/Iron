import os
import sys
import torch
import numpy as np

# Setup path for co-tracker
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COTRACKER_DIR = os.path.join(PROJECT_ROOT, "co-tracker")
if COTRACKER_DIR not in sys.path and os.path.exists(COTRACKER_DIR):
    sys.path.insert(0, COTRACKER_DIR)

try:
    from cotracker.predictor import CoTrackerOnlinePredictor
except ImportError:
    pass

class OnlineTrackingEngine:
    def __init__(self, checkpoint_path, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading online model on {self.device} from {checkpoint_path}")
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        self.model = CoTrackerOnlinePredictor(checkpoint=checkpoint_path).to(self.device)
        self.step = self.model.step # Usually window_len // 2

    def track_video_online(self, frames, points):
        """
        Simulate online tracking on a video, following the logic of cotracker online_demo.
        frames: (T, H, W, 3) numpy array
        points: (N, 2) initial points
        Returns: (N, T, 2) tracks
        """
        T_total = len(frames)
        N = points.shape[0]
        
        # CoTrackerOnlinePredictor expects (B, N, 3) for queries
        q = np.zeros((1, N, 3), dtype=np.float32)
        q[0, :, 0] = 0 # frame 0
        q[0, :, 1:] = points
        queries_tensor = torch.from_numpy(q).to(self.device)
        
        all_pred_tracks = []
        window_frames = []
        is_first_step = True
        
        print("Starting online tracking simulation...")
        
        for i in range(T_total):
            frame = frames[i]
            
            # The logic from online_demo.py:
            # Check for processing block BEFORE appending the frame
            if i % self.step == 0 and i != 0:
                # video_chunk should be at most 2*step = window_len
                # Note: window_frames currently contains frames 0 to i-1
                chunk = window_frames[-self.step * 2:]
                video_chunk = torch.tensor(np.stack(chunk)).float().permute(0, 3, 1, 2)[None].to(self.device)
                
                # If it's the first step, it just initializes and returns (None, None)
                res = self.model(video_chunk, is_first_step=is_first_step, queries=queries_tensor if is_first_step else None)
                if res[0] is not None:
                    all_pred_tracks.append(res[0].cpu().numpy())
                
                is_first_step = False
            
            window_frames.append(frame)

        # Processing the final frames
        last_chunk = window_frames[-(i % self.step) - self.step - 1 :]
        if len(last_chunk) > 0:
            video_chunk = torch.tensor(np.stack(last_chunk)).float().permute(0, 3, 1, 2)[None].to(self.device)
            res = self.model(video_chunk, is_first_step=is_first_step)
            if res[0] is not None:
                all_pred_tracks.append(res[0].cpu().numpy())

        if not all_pred_tracks:
            # This could happen for very short videos
            # Return initial points repeated
            out = np.tile(points[:, None, :], (1, T_total, 1))
            return out

        # Each chunk in all_pred_tracks is (1, step, N, 2)
        # We need to concatenate them
        stitched = np.concatenate(all_pred_tracks, axis=1) # (1, TotalTracked, N, 2)
        
        # Prepend the first frame's original points
        first_frame_pos = points[None, None, :, :] # (1, 1, N, 2)
        final_tracks_seq = np.concatenate([first_frame_pos, stitched], axis=1)[0] # (TotalT, N, 2)
        
        # Adjust length to match T_total
        if final_tracks_seq.shape[0] > T_total:
            final_tracks_seq = final_tracks_seq[:T_total]
        elif final_tracks_seq.shape[0] < T_total:
            diff = T_total - final_tracks_seq.shape[0]
            # Repeat last position
            padding = np.tile(final_tracks_seq[-1:], (diff, 1, 1))
            final_tracks_seq = np.concatenate([final_tracks_seq, padding], axis=0)
            
        # Transpose to (N, T, 2)
        return final_tracks_seq.transpose(1, 0, 2)
