import sys
import os
import importlib.util
import torch
import imageio.v3 as iio
import cv2
import numpy as np

# 项目路径
REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "co-tracker"))
sys.path.insert(0, os.path.join(REPO_DIR, "cotracker"))

# 可视化器
from cotracker.utils.visualizer import Visualizer  # type: ignore

# 配置（保持原有常量）
INPUT_DIR = "input_videos"
OUTPUT_DIR = "output_videos"
GRID_SIZE = 10
MODEL_TYPE = "cotracker3_offline"  # 或 "cotracker3_online"
CKPT_REL = os.path.join("checkpoints", "scaled_offline.pth")
TARGET_RES = None

# 子目录（你之前要求的三种策略输出）
OUT_COTRACKER = os.path.join(OUTPUT_DIR, "cotracker_only")
OUT_LK = os.path.join(OUTPUT_DIR, "lk_only")
OUT_MIXED = os.path.join(OUTPUT_DIR, "mixed_lk_cotracker")
os.makedirs(OUT_COTRACKER, exist_ok=True)
os.makedirs(OUT_LK, exist_ok=True)
os.makedirs(OUT_MIXED, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model: {MODEL_TYPE} (device={device})")

# 加载 cotracker（保持原逻辑）
hubconf_path = os.path.join(REPO_DIR, "hubconf.py")
spec = importlib.util.spec_from_file_location("hubconf", hubconf_path)
hubconf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hubconf)
cotracker_builder = getattr(hubconf, MODEL_TYPE)
cotracker = cotracker_builder(pretrained=False).to(device)

ckpt_path = os.path.join(REPO_DIR, CKPT_REL)
if not os.path.isfile(ckpt_path):
    raise FileNotFoundError(f"本地权重未找到: {ckpt_path}")
print(f"Loading weights from local checkpoint: {ckpt_path}")
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

from collections import OrderedDict
new_state = OrderedDict()
for k, v in state_dict.items():
    new_key = k[len("module.") :] if k.startswith("module.") else k
    new_state[new_key] = v

load_result = cotracker.load_state_dict(new_state, strict=False)
try:
    missing = load_result.missing_keys
    unexpected = load_result.unexpected_keys
    print(f"Loaded checkpoint. missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
except Exception:
    print("Loaded checkpoint (unable to report missing/unexpected keys).")

cotracker.eval()

# --- 模块化函数（重构） ---
def read_and_prepare_frames(input_path, target_res=None, mb=16, pad_value=120):
    frames = iio.imread(input_path, plugin="FFMPEG")
    frames = np.array(frames)
    if target_res is not None:
        frames = np.array([cv2.resize(f, target_res) for f in frames])
    h, w = frames.shape[1], frames.shape[2]
    pad_h = ((h + mb - 1) // mb) * mb
    pad_w = ((w + mb - 1) // mb) * mb
    if pad_h != h or pad_w != w:
        pad_bottom = pad_h - h
        pad_right = pad_w - w
        frames = np.array([
            cv2.copyMakeBorder(f, 0, pad_bottom, 0, pad_right, cv2.BORDER_CONSTANT, value=pad_value)
            for f in frames
        ])
        print(f"Padded frames from ({w}, {h}) to ({pad_w}, {pad_h}) for macro_block_size={mb}")
    fps = cv2.VideoCapture(input_path).get(cv2.CAP_PROP_FPS) or 30.0
    return frames, fps

def detect_corners(frames, max_corners=200, quality=0.01, min_dist=7, block=7, valid_mask=None):
    """
    frames: ndarray[T,H,W,3]
    valid_mask: optional bool mask HxW, True means allowed region (keep points only where True)
    """
    gray0 = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(
        gray0,
        maxCorners=max_corners,
        qualityLevel=quality,
        minDistance=min_dist,
        blockSize=block,
    )
    if p0 is None:
        return np.empty((0, 2), dtype=np.float32)
    pts = p0.reshape(-1, 2).astype(np.float32)
    if valid_mask is not None:
        # filter points outside allowed region (valid_mask[y,x] expected)
        h, w = valid_mask.shape
        keep = []
        for x, y in pts:
            xi, yi = int(round(x)), int(round(y))
            if 0 <= yi < h and 0 <= xi < w and valid_mask[yi, xi]:
                keep.append([x, y])
        if len(keep) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(keep, dtype=np.float32)
    return pts

from strategies.strategy_lk import run as run_lk
from strategies.strategy_cotracker import run as run_cotracker_strategy
from strategies.strategy_mixed import run as run_mixed

def run_cotracker(video_tensor, grid_size=10):
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker(video_tensor, grid_size=grid_size)
    return pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()

def compute_static_text_mask(frames, corner_pct=((0.18,0.18),(0.18,0.18),(0.18,0.18)), 
                             canny_thresh=(50,150), dilate_iter=5):
    """
    Heuristic static-overlay (text) mask:
      - compute median frame (stable overlay like time/address remains)
      - detect edges on median -> dilate -> restrict to corners
    corner_pct: tuple of 3 (w_pct, h_pct) for top-left, bottom-left, bottom-right
    returns mask HxW (True = allowed area, False = overlay area to exclude)
    """
    median = np.median(frames.astype(np.uint8), axis=0).astype(np.uint8)
    gray = cv2.cvtColor(median, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_thresh[0], canny_thresh[1])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    H, W = gray.shape
    overlay_mask = np.zeros_like(edges, dtype=np.uint8)

    # restrict candidate overlay to three corners (user mentioned)
    # top-left
    w_pct, h_pct = corner_pct[0]
    overlay_mask[0:int(H*h_pct), 0:int(W*w_pct)] = edges[0:int(H*h_pct), 0:int(W*w_pct)]
    # bottom-left
    w_pct, h_pct = corner_pct[1]
    overlay_mask[int(H*(1-h_pct)):H, 0:int(W*w_pct)] |= edges[int(H*(1-h_pct)):H, 0:int(W*w_pct)]
    # bottom-right
    w_pct, h_pct = corner_pct[2]
    overlay_mask[int(H*(1-h_pct)):H, int(W*(1-w_pct)):W] |= edges[int(H*(1-h_pct)):H, int(W*(1-w_pct)):W]

    # connected components + area filtering (small components are likely text edges)
    contours, _ = cv2.findContours(overlay_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    final_mask = np.ones_like(gray, dtype=np.uint8)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        if area > 10:  # keep small threshold; tune if needed
            # mark region as excluded (set to 0)
            final_mask[y:y+h, x:x+w] = 0
    return final_mask.astype(bool)  # True allowed, False exclude

# 新增：基于时间序列方差的静态过滤（排除长期不变区域，如文字）
def compute_temporal_variance_mask(frames, std_thresh=2.0):
    """
    返回 True=允许 (变化足够)，False=排除 (长期静态)
    std_thresh: 灰度像素标准差阈值，小于该值视为静态（可能为叠加文字）
    """
    gray_seq = np.array([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames])
    per_pixel_std = np.std(gray_seq, axis=0)
    mask = per_pixel_std > std_thresh
    return mask

# 替换：更鲁棒且一对一的映射（处理归一化/像素坐标、坐标顺序、唯一匹配）
def map_cotracker_to_init_points(pred_tracks_np, init_points, T, frame_shape, thresh=15.0):
    """
    pred_tracks_np: raw cotracker output (various shapes)
    init_points: (N_init,2) in pixel coords (x,y)
    T: number of frames
    frame_shape: (H,W)
    thresh: matching threshold in pixels
    返回 (N_init, T, 2) 对齐数组，未匹配行为 NaN
    使用贪心一对一匹配（按距离排序）以避免重复映射
    """
    arr = pred_tracks_np
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]  # -> (T, N, 2)
    cot_tracks = None
    if arr.ndim == 3 and arr.shape[0] == T:
        cot_tracks = np.transpose(arr, (1, 0, 2))  # (N_cot, T, 2)
    elif arr.ndim == 3 and arr.shape[1] == T:
        cot_tracks = arr  # (N_cot, T, 2)
    if cot_tracks is None or init_points.size == 0:
        return None

    H, W = frame_shape
    cot_tracks = cot_tracks.astype(np.float32)

    # 如果 cotracker 输出看起来是归一化（<=1.01），转换为像素坐标 (x,y)
    if np.nanmax(cot_tracks) <= 1.01:
        cot_tracks[..., 0] = cot_tracks[..., 0] * W
        cot_tracks[..., 1] = cot_tracks[..., 1] * H

    cot_first = cot_tracks[:, 0, :]  # (N_cot,2)
    lk_init = init_points.astype(np.float32)  # (N_init,2)
    num_lk = lk_init.shape[0]
    num_cot = cot_first.shape[0]

    best_mapped = None
    best_count = -1

    # 尝试两种坐标顺序（x,y）和 (y,x) 的候选，并用贪心一对一分配
    for cand_first in [cot_first, cot_first[:, ::-1]]:
        # 计算距离矩阵 (num_lk, num_cot)
        dists = np.linalg.norm(lk_init[:, None, :] - cand_first[None, :, :], axis=2)
        # 构造所有 (i,j,dist) 对并按 dist 升序
        idxs = np.array([(i, j, dists[i, j]) for i in range(num_lk) for j in range(num_cot)],
                        dtype=[('i',int),('j',int),('d',float)])
        if idxs.size == 0:
            continue
        # sort by distance
        order = np.argsort(idxs['d'])
        assigned_init = np.zeros(num_lk, dtype=bool)
        assigned_cot = np.zeros(num_cot, dtype=bool)
        mapped = np.full((num_lk, T, 2), np.nan, dtype=np.float32)
        for idx in order:
            i = int(idxs['i'][idx])
            j = int(idxs['j'][idx])
            d = float(idxs['d'][idx])
            if d > thresh:
                break  # further ones will be larger
            if not assigned_init[i] and not assigned_cot[j]:
                # assign j -> i
                mapped[i, :, :] = cot_tracks[j]
                assigned_init[i] = True
                assigned_cot[j] = True
        cnt = int(np.sum(assigned_init))
        if cnt > best_count:
            best_count = cnt
            best_mapped = mapped

    return best_mapped

# visualization/tracking/saving helpers moved to strategies/common.py

# save_trajectories moved to strategies/common.py

# --- 主流程：使用同一组角点作为三种策略的起始点 ---
video_name = "sample.mp4"
input_path = os.path.join(INPUT_DIR, video_name)
if not os.path.isfile(input_path):
    print(f"⚠️ 未找到文件: {input_path}")
    raise SystemExit(1)
print(f"\n=== Processing: {input_path} ===")

frames, fps = read_and_prepare_frames(input_path, TARGET_RES, mb=16)

# 先计算静态文字/叠加区域和时间方差掩码并合并为 allowed_mask（True = 可检测区域）
text_mask = compute_static_text_mask(frames, corner_pct=((0.20,0.16),(0.20,0.16),(0.20,0.16)), canny_thresh=(40,150), dilate_iter=5)
temporal_mask = compute_temporal_variance_mask(frames, std_thresh=2.0)
allowed_mask = (text_mask & temporal_mask)

# 1) 用 Shi-Tomasi 在第一帧检测角点（作为三种策略的共同初始点集合），并过滤掉静态叠加区域
# Option: allow manual selection of points instead of automatic Shi-Tomasi
USE_MANUAL_POINTS = True
MANUAL_POINTS_PATH = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0] + "_manual_points.json")

def select_points_interactive(img):
    """Interactive point selection using mouse clicks. Left click to add, 'r' to remove last, Enter or 'q' to finish."""
    points = []
    clone = img.copy()
    win = "Select points - left click to add, r to remove, Enter/q to finish"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, param):
        nonlocal points, clone
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((float(x), float(y)))
            # redraw
            clone = img.copy()
            for (px, py) in points:
                cv2.circle(clone, (int(px), int(py)), 4, (0, 255, 0), -1)
            cv2.imshow(win, clone)

    cv2.setMouseCallback(win, on_mouse)
    cv2.imshow(win, clone)
    while True:
        key = cv2.waitKey(0) & 0xFF
        # Enter or q to finish
        if key == 13 or key == ord('q'):
            break
        if key == ord('r'):
            if len(points) > 0:
                points.pop()
                clone = img.copy()
                for (px, py) in points:
                    cv2.circle(clone, (int(px), int(py)), 4, (0, 255, 0), -1)
                cv2.imshow(win, clone)
    cv2.destroyWindow(win)
    return np.array(points, dtype=np.float32)


def try_load_manual_points(path):
    try:
        import json
        if os.path.exists(path):
            with open(path, 'r') as f:
                pts = json.load(f)
            return np.array(pts, dtype=np.float32)
    except Exception:
        return None
    return None


first = frames[0].copy()
manual_loaded = try_load_manual_points(MANUAL_POINTS_PATH)
if USE_MANUAL_POINTS and (manual_loaded is None):
    # decide if interactive selection is possible
    can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
    if can_interactive:
        try:
            print("请在弹出的窗口中用鼠标点击要追踪的角点：左键添加，按 r 删除最后一个，按 Enter 或 q 完成。")
            manual_pts = select_points_interactive(first)
        except Exception as e:
            print("交互式选择不可用（异常）：", e)
            manual_pts = None
    else:
        manual_pts = None

    if manual_pts is None or manual_pts.size == 0:
        # headless or user didn't pick; save first frame for external annotation and fall back to auto
        save_path = os.path.join(OUTPUT_DIR, os.path.splitext(video_name)[0] + "_firstframe.png")
        cv2.imwrite(save_path, first)
        print(f"无法进行交互式选择（可能是无显示环境）。已把第一帧保存到: {save_path}")
        print("你可以在本地打开该图片并手动创建一个 JSON 文件，路径：", MANUAL_POINTS_PATH)
        print("JSON 格式示例: [[x1,y1],[x2,y2],...]。脚本会优先加载该文件作为手动点列表。")
        MAX_CORNERS = 200
        init_pts = detect_corners(frames, max_corners=MAX_CORNERS, quality=0.01, min_dist=7, block=7, valid_mask=allowed_mask)
        print(f"采用自动检测的角点（after masking）: {init_pts.shape[0]}")
    else:
        init_pts = manual_pts
        try:
            import json
            with open(MANUAL_POINTS_PATH, 'w') as f:
                json.dump(init_pts.tolist(), f)
            print(f"Manual points saved to {MANUAL_POINTS_PATH}")
        except Exception:
            pass
        print(f"Using {init_pts.shape[0]} initial points")
elif manual_loaded is not None:
    init_pts = manual_loaded
    print(f"Loaded manual points from {MANUAL_POINTS_PATH}, count={init_pts.shape[0]}")
else:
    MAX_CORNERS = 200
    init_pts = detect_corners(frames, max_corners=MAX_CORNERS, quality=0.01, min_dist=7, block=7, valid_mask=allowed_mask)
    print(f"Detected initial corners (after masking): {init_pts.shape[0]}")

# Strategy A: LK only (直接用这些初始点)
print("Running strategy: LK only (from Shi-Tomasi points)")
# use strategy module
trajectories_lk, out_vis_lk, npz_lk, csv_lk = run_lk(frames, fps, init_pts, OUT_LK, os.path.splitext(video_name)[0])
if out_vis_lk:
    print(f"LK outputs: video={out_vis_lk}, npz={npz_lk}, csv={csv_lk}")
else:
    print("LK 未检测到可跟踪点。")

# Strategy B: cotracker only (把 cotracker 的网格轨迹映射到 init_pts)
print("Running strategy: cotracker only (map cotracker tracks to Shi-Tomasi points)")
video_tensor = torch.tensor(frames).permute(0,3,1,2)[None].float().to(device)
pred_tracks_np, pred_vis_np = run_cotracker(video_tensor, grid_size=GRID_SIZE)
frame_shape = (frames.shape[1], frames.shape[2])  # (H, W)
# delegate to strategy module
cot_mapped, out_vis_ct, npz_ct, csv_ct = run_cotracker_strategy(frames, fps, init_pts, pred_tracks_np, frame_shape, OUT_COTRACKER, os.path.splitext(video_name)[0], thresh=15.0)
if out_vis_ct:
    print(f"cotracker (mapped) outputs: video={out_vis_ct}, npz={npz_ct}, csv={csv_ct}")
else:
    print("无法将 cotracker 输出映射到初始角点（可能无匹配），已保存原始 cotracker tensors。")

# Strategy C: Mixed (prefer LK when available, else fallback to cotracker)
print("Running strategy: Mixed (LK preferred, cotracker fallback)")
# delegate to mixed strategy module
fused_arr, out_vis_mixed, npz_mixed, csv_mixed = run_mixed(frames, fps, init_pts, trajectories_lk, cot_mapped, OUT_MIXED, os.path.splitext(video_name)[0])
if out_vis_mixed:
    print(f"Mixed outputs: video={out_vis_mixed}, npz={npz_mixed}, csv={csv_mixed}")
else:
    print("混合输出未生成（参见上方原因）。")

print("\n处理完成。输出目录：")
print(f" - cotracker only: {OUT_COTRACKER}")
print(f" - lk only: {OUT_LK}")
print(f" - mixed lk + cotracker: {OUT_MIXED}")
