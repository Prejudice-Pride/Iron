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
        Run tracking on video in one go.
        frames: (T, H, W, 3) numpy array
        points: (N, 2) initial points
        Returns: (N, T, 2) tracks, (N, T) visibilities
        """
        T = len(frames)
        N = points.shape[0]

        # Initial queries: CoTracker expects (1, N, 3) -> (t, x, y)
        q = np.zeros((1, N, 3), dtype=np.float32)
        q[0, :, 0] = 0
        q[0, :, 1:] = points

        print("Processing video (Memory Optimized FP16) ...")
        # 优化：通过 half() 使用半精度 FP16，显存占用直接减半
        frames_tensor = (
            torch.from_numpy(frames).permute(0, 3, 1, 2)[None].half().to(self.device)
        )
        queries_tensor = torch.from_numpy(q).half().to(self.device)

        # 增加 no_grad 节省资源
        with torch.no_grad():
            # 开启 autocast 以确保模型内部计算的稳定性
            with torch.cuda.amp.autocast():
                pred_tracks, pred_visibilities = self.model(frames_tensor, queries=queries_tensor)

        return self._to_pixel_tracks(pred_tracks.cpu().float().numpy()), \
               self._to_pixel_visibilities(pred_visibilities.cpu().float().numpy())

    def track_video_chunked(self, frames, points, chunk_len=100, overlap=10):
        """
        Stable chunked tracking with drift-free stitching.
        frames: (T, H, W, 3)
        points: (N, 2)
        Returns: (N, T, 2)
        """

        T = len(frames)
        N = points.shape[0]

        # 初始化 query（第一帧）
        q = np.zeros((1, N, 3), dtype=np.float32)
        q[0, :, 0] = 0
        q[0, :, 1:] = points

        current_queries = torch.from_numpy(q).to(self.device)

        stitched_tracks = None
        stitched_visibilities = None

        print("Processing video in chunks (stable mode)...")

        start_frame = 0

        while start_frame < T - 1:

            end_frame = min(start_frame + chunk_len, T)
            chunk_frames = frames[start_frame:end_frame]

            chunk_tensor = (
                torch.from_numpy(chunk_frames)
                .permute(0, 3, 1, 2)[None]
                .float()
                .to(self.device)
            )

            print(f"  Chunk: {start_frame} → {end_frame}")

            with torch.no_grad():
                torch.cuda.empty_cache()

                # 每个 chunk 重新设时间起点
                current_queries[:, :, 0] = 0

                pred_tracks_chunk, pred_vis_chunk = self.model(
                    chunk_tensor, queries=current_queries
                )

            chunk_tracks_np = pred_tracks_chunk.cpu().numpy()      # (1, T_chunk, N, 2)
            chunk_vis_np = pred_vis_chunk.cpu().numpy()            # (1, T_chunk, N)

            # --------------------------------------------------
            # 第一段直接赋值
            # --------------------------------------------------
            if stitched_tracks is None:
                stitched_tracks = chunk_tracks_np
                stitched_visibilities = chunk_vis_np

            else:
                # overlap长度
                n_overlap = stitched_tracks.shape[1] - start_frame

                if n_overlap > 0:

                    old = stitched_tracks[:, -n_overlap:, :, :]     # (1, overlap, N, 2)
                    new = chunk_tracks_np[:, :n_overlap, :, :]

                    # --------------------------------------------------
                    # ✅ 核心修复：Robust Rigid Alignment（无漂移关键）
                    # --------------------------------------------------
                    diff = old - new   # (1, overlap, N, 2)

                    # reshape → (overlap*N, 2)
                    diff_2d = diff.reshape(-1, 2)

                    # 去 NaN
                    valid = ~np.isnan(diff_2d).any(axis=1)
                    diff_valid = diff_2d[valid]

                    if len(diff_valid) > 10:
                        offset = np.nanmean(diff_valid, axis=0)  # 使用均值对齐，减少累积偏差
                    else:
                        offset = np.zeros((2,))

                    offset = offset.reshape(1, 1, 1, 2)

                    # 对齐整个新chunk
                    chunk_tracks_np = chunk_tracks_np + offset

                    # --------------------------------------------------
                    # ✅ 修复：使用线性插值 Blending 处理衔接处的跳变
                    # --------------------------------------------------
                    # 这样做可以确保 start_frame 到之前拼接点之间的过渡更平滑
                    alpha = np.linspace(0, 1, n_overlap).reshape(1, n_overlap, 1, 1)
                    blended_overlap = (1 - alpha) * old + alpha * chunk_tracks_np[:, :n_overlap, :, :]
                    
                    # 更新旧的 overlap 部分
                    stitched_tracks[:, -n_overlap:, :, :] = blended_overlap

                    # 拼接剩余的部分
                    stitched_tracks = np.concatenate(
                        [stitched_tracks, chunk_tracks_np[:, n_overlap:, :, :]],
                        axis=1
                    )

                    stitched_visibilities = np.concatenate(
                        [stitched_visibilities, chunk_vis_np[:, n_overlap:, :]],
                        axis=1
                    )

                else:
                    stitched_tracks = np.concatenate(
                        [stitched_tracks, chunk_tracks_np],
                        axis=1
                    )
                    stitched_visibilities = np.concatenate(
                        [stitched_visibilities, chunk_vis_np],
                        axis=1
                    )

            # --------------------------------------------------
            # 下一段 query（使用已对齐的坐标作为起始点，减少累积偏差）
            # --------------------------------------------------
            if end_frame == T:
                break

            # 采用动态起始帧跳转
            next_start_frame = end_frame - overlap
            if next_start_frame <= start_frame:
                next_start_frame = start_frame + 1
            if next_start_frame >= T - 1:
                break

            # 计算下一段的起始 query
            query_idx = next_start_frame - start_frame
            # 这里的 last_pos 是在当前 chunk 坐标系下，经过偏移修正后的坐标
            last_pos = chunk_tracks_np[0, query_idx, :, :]

            new_q = torch.zeros((1, N, 3), device=self.device)
            new_q[0, :, 0] = 0
            new_q[0, :, 1:] = torch.from_numpy(last_pos).to(self.device).float()

            current_queries = new_q
            start_frame = next_start_frame

        return self._to_pixel_tracks(stitched_tracks), \
            self._to_pixel_visibilities(stitched_visibilities)