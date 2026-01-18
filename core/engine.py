import os
import sys
import torch
import numpy as np

# Try to ensure cotracker can be imported
# Assumes 'co-tracker' folder is in the project root relative to this file?
# this file is in core/, so project root is ../
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COTRACKER_DIR = os.path.join(PROJECT_ROOT, "co-tracker")
if COTRACKER_DIR not in sys.path and os.path.exists(COTRACKER_DIR):
    sys.path.insert(0, COTRACKER_DIR)
print(COTRACKER_DIR)
CKPT_PATH = os.path.join(COTRACKER_DIR, "checkpoints", "scaled_offline.pth")

try:
    from cotracker.predictor import CoTrackerPredictor # type: ignore
except ImportError:
    # If the above path hack didn't work (e.g. structure is different), let the caller handle it or fail
    pass


class TrackingEngine:
    def __init__(self, device=None, offline=True):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Loading model on {self.device} from {CKPT_PATH}")
        if not os.path.exists(CKPT_PATH):
            raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

        self.model = CoTrackerPredictor(checkpoint=CKPT_PATH, offline=offline).to(
            self.device
        )

    def _to_pixel_tracks(self, pred_tracks_np):
        """Standardizes output shape to (N, T, 2)"""
        arr = pred_tracks_np
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[
                0
            ]  # remove batch dim -> (T, N, 2) or (N, T, 2) depending on model version

        # CoTracker usually returns (1, T, N, 2)
        # after arr[0] -> (T, N, 2)
        # We want (N, T, 2) for our visualizer/processing usually, or check consistency.
        # Original script:
        # if arr.ndim == 3:
        #   if arr.shape[0] <= 4 ... logic

        # Let's trust the logic from original script
        if arr.ndim == 3:
            # Check if Time is first dimension?
            # heuristic: T is usually larger than N? No, N can be 5000.
            # But the loop stitches along axis 1 (time?) in original script?
            # In original script: stitched_tracks = np.concatenate(..., axis=1)
            # This implies shape (1, T, N, 2) originally, so stiched is (1, TotalT, N, 2)
            # Then to_pixel_tracks takes arr[0] -> (TotalT, N, 2).
            # Then returns transpose(1, 0, 2) -> (N, TotalT, 2).

            # Let's replicate exact behavior
            tracks = np.transpose(arr, (1, 0, 2))  # (T, N, 2) -> (N, T, 2)
            return tracks
        else:
            # If it's something else
            return arr

    def _to_pixel_visibilities(self, pred_visibilities_np):
        """Standardizes visibilities shape to (N, T)"""
        arr = pred_visibilities_np
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]  # remove batch dim -> (T, N)
        
        if arr.ndim == 2:
            # Transpose to (N, T)
            vis = np.transpose(arr, (1, 0))  # (T, N) -> (N, T)
            return vis
        else:
            return arr

    def track_video(self, frames, points):
        """
        Run tracking on video in chunks.
        frames: (T, H, W, 3) numpy array
        points: (N, 2) initial points
        Returns: (N, T, 2) tracks
        """
        T = len(frames)
        N = points.shape[0]

        # Initial queries
        # CoTracker expects (1, N, 3) -> t, x, y
        q = np.zeros((1, N, 3), dtype=np.float32)
        q[0, :, 0] = 0
        q[0, :, 1:] = points

        queries = torch.from_numpy(q).to(self.device)

        frames_input = frames  # (T, H, W, 3)

        print("Processing video ...")
        queries = (
            torch.tensor(queries).permute(0, 3, 1, 2)[None].float().to(self.device)
        )
        pred_tracks, pred_visibilities = self.model(frames_input, queries=queries)

        return self._to_pixel_tracks(pred_tracks.cpu().numpy()), self._to_pixel_visibilities(pred_visibilities.cpu().numpy())

    def track_video_chunked(self, frames, points, chunk_len=40, overlap=10):
        """
        Run tracking on video in chunks.
        frames: (T, H, W, 3) numpy array
        points: (N, 2) initial points
        Returns: (N, T, 2) tracks
        """
        T = len(frames)
        N = points.shape[0]

        # Initial queries
        # CoTracker expects (1, N, 3) -> t, x, y
        q = np.zeros((1, N, 3), dtype=np.float32)
        q[0, :, 0] = 0
        q[0, :, 1:] = points

        current_queries = torch.from_numpy(q).to(self.device)

        frames_input = frames  # (T, H, W, 3)
        stitched_tracks = None
        stitched_visibilities = None

        print("Processing video in chunks...")
        start_frame = 0

        while start_frame < T - 1:
            end_frame = min(start_frame + chunk_len, T)
            chunk_frames = frames_input[start_frame:end_frame]

            # Prepare tensor: (1, T_chunk, 3, H, W)
            chunk_tensor = (
                torch.tensor(chunk_frames)
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(self.device)
            )

            print(f"  Chunk: {start_frame} to {end_frame}")

            with torch.no_grad():
                torch.cuda.empty_cache()
                # Update time in queries to be relative to the chunk start?
                # CoTracker offline usually takes queries relative to the video provided.
                # If we provide a chunk, time 0 is the start of the chunk.
                if current_queries is not None:
                    current_queries[:, :, 0] = 0  # Reset time to 0 for the new chunk

                pred_tracks_chunk, pred_visibilities_chunked = self.model(
                    chunk_tensor, queries=current_queries
                )

            chunk_tracks_np = pred_tracks_chunk.cpu().numpy()  # (1, T_chunk, N, 2)
            chunk_visibilities_np = pred_visibilities_chunked.cpu().numpy()  # (1, T_chunk, N)

            if stitched_tracks is None and stitched_visibilities is None:
                stitched_tracks = chunk_tracks_np  # (1, T, N, 2)
                stitched_visibilities = chunk_visibilities_np  # (1, T, N, 2)
            else:
                n_overlap = stitched_tracks.shape[1] - start_frame
                if n_overlap > 0:
                    # Blend overlap
                    # overlapping part in old: stitched_tracks[:, -n_overlap:, :, :]
                    # overlapping part in new: chunk_tracks_np[:, :n_overlap, :, :]
                    alpha = np.linspace(0, 1, n_overlap).reshape(1, -1, 1, 1)
                    overlap_old_tracks = stitched_tracks[:, -n_overlap:, :, :]
                    overlap_new_tracks = chunk_tracks_np[:, :n_overlap, :, :]

                    blended_tracks = (1 - alpha) * overlap_old_tracks + alpha * overlap_new_tracks
                    stitched_tracks[:, -n_overlap:, :, :] = blended_tracks
                    
                    overlap_old_visibilities = stitched_visibilities[:, -n_overlap:, :]
                    overlap_new_visibilities = chunk_visibilities_np[:, :n_overlap, :]
                    
                    blended_visibilities = overlap_old_visibilities & overlap_new_visibilities  # Both must be visible
                    stitched_visibilities[:, -n_overlap:, :] = blended_visibilities

                    # Append non-overlapping part
                    stitched_tracks = np.concatenate(
                        [stitched_tracks, chunk_tracks_np[:, n_overlap:, :, :]], axis=1
                    )
                    stitched_visibilities = np.concatenate(
                        [stitched_visibilities, chunk_visibilities_np[:, n_overlap:, :]], axis=1
                    )
                else:
                    stitched_tracks = np.concatenate(
                        [stitched_tracks, chunk_tracks_np], axis=1
                    )
                    stitched_visibilities = np.concatenate(
                        [stitched_visibilities, chunk_visibilities_np], axis=1
                    )

            if end_frame == T:
                break

            # Prepare next query
            next_start_frame = end_frame - overlap
            if next_start_frame <= start_frame:
                next_start_frame = start_frame + 1
            if next_start_frame >= T - 1:
                break

            query_idx = next_start_frame - start_frame
            # shape (1, T_chunk, N, 2) -> get [0, query_idx, :, :] -> (N, 2)
            last_pos = chunk_tracks_np[0, query_idx, :, :]

            new_q = torch.zeros((1, last_pos.shape[0], 3), device=self.device)
            new_q[0, :, 0] = 0
            new_q[0, :, 1:] = torch.from_numpy(last_pos).to(self.device)
            current_queries = new_q

            start_frame = next_start_frame

        # Final conversion
        return self._to_pixel_tracks(stitched_tracks), self._to_pixel_visibilities(stitched_visibilities)
