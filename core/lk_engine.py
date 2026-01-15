import cv2
import numpy as np

class LKTrackingEngine:
    def __init__(self, win_size=(15, 15), max_level=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)):
        """
        Initialize Lucas-Kanade Optical Flow Engine.
        """
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=criteria
        )

    def track_video(self, frames, points):
        """
        Track points using Lucas-Kanade optical flow.
        frames: (T, H, W, 3) RGB numpy array
        points: (N, 2) initial points in (x, y)
        Returns: (N, T, 2) tracks
        """
        T = len(frames)
        N = points.shape[0]
        
        # Initialize tracks with NaNs
        tracks = np.full((N, T, 2), np.nan, dtype=np.float32)
        
        # Set initial positions
        tracks[:, 0, :] = points
        
        # Convert frames to grayscale for LK
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_RGB2GRAY)
        curr_pts = points.copy().reshape(-1, 1, 2)
        
        print(f"Starting LK tracking for {N} points over {T} frames...")
        
        for t in range(1, T):
            curr_gray = cv2.cvtColor(frames[t], cv2.COLOR_RGB2GRAY)
            
            # Identify points that were valid in the previous frame
            # (In LK, we usually track from the last known position)
            valid_mask = ~np.isnan(tracks[:, t-1, 0])
            if not np.any(valid_mask):
                break
                
            # pts_to_track: points found in the previous frame
            pts_prev = tracks[valid_mask, t-1, :].reshape(-1, 1, 2)
            
            # Calculate optical flow
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, pts_prev, None, **self.lk_params
            )
            
            # Update tracks
            # status == 1 means point was found
            found_mask = (status.flatten() == 1)
            
            # Map back to original indices
            original_indices = np.where(valid_mask)[0]
            successful_indices = original_indices[found_mask]
            
            tracks[successful_indices, t, :] = next_pts[found_mask].reshape(-1, 2)
            
            # Prepare for next frame
            prev_gray = curr_gray
            
            if t % 50 == 0:
                print(f"  Frame {t}/{T} processed...")

        return tracks
