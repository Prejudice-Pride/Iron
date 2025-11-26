"""Interactive annotator to create ROI and corner JSON files for SeismicTracker_auto/manual.

Saves:
  - rois/<base>_rois.json       -> [[x1,y1,x2,y2],...]
  - corner_points/<base>_corners.json -> [[x,y],...]

Usage examples:
  python annotate_points.py input_videos/1.mp4 --rois
  python annotate_points.py input_videos/1.mp4 --corners
  python annotate_points.py input_videos/1.mp4 --both

If no interactive display available, the script will save the first frame to the working folder
and ask you to annotate externally and re-run.
"""
import os
import sys
import argparse
import json
import cv2
import imageio.v3 as iio

ROI_DIR = "rois"
CORNER_DIR = "corner_points"


def ensure_dirs():
    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(CORNER_DIR, exist_ok=True)


def read_first_frame(path):
    # Prefer using cv2.VideoCapture to obtain the original-resolution first
    # frame (more reliable across platforms/codecs). Use imageio as a fallback.
    try:
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            return frame
    except Exception:
        # fall through to imageio
        pass

    try:
        frames = iio.imread(path, plugin="FFMPEG")
        # imageio may return an ndarray or list-like; normalize to list
        if isinstance(frames, (list, tuple)) or (hasattr(frames, "shape") and getattr(frames, "ndim", 0) >= 3):
            frames = list(frames)
        if len(frames) == 0:
            raise RuntimeError("No frames found via imageio")
        return frames[0]
    except Exception:
        raise RuntimeError(f"Unable to read first frame from {path}")


# ---------------- ROIs interactive ----------------

def select_rois_interactive(img):
    win = "Select ROIs - drag to draw, r remove last, Enter/q finish"
    clone = img.copy()
    rois = []
    current = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and current is not None:
            sx, sy, _, _ = current
            clone = img.copy()
            cv2.rectangle(clone, (sx, sy), (x, y), (0, 0, 255), 2)
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
            cv2.imshow(win, clone)
        elif event == cv2.EVENT_LBUTTONUP and current is not None:
            sx, sy, _, _ = current
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            rois.append([int(x1), int(y1), int(x2), int(y2)])
            current = None
            clone = img.copy()
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Try to display at native resolution to avoid window scaling blur
    try:
        cv2.resizeWindow(win, img.shape[1], img.shape[0])
    except Exception:
        pass
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    print("Instructions: drag left mouse to draw exclusion rectangles. Press 'r' to remove last, Enter or 'q' to finish.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(rois) > 0:
                rois.pop()
                clone = img.copy()
                for r in rois:
                    cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)
                cv2.imshow(win, clone)
        if key == 13 or key == ord('q'):
            break
    cv2.destroyWindow(win)
    return rois


# ---------------- Corners interactive ----------------

def select_points_interactive(img):
    win = "Select corner points - left click to add, r remove last, Enter/q finish"
    clone = img.copy()
    pts = []

    def on_mouse(event, x, y, flags, param):
        nonlocal clone
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([int(x), int(y)])
            clone = img.copy()
            for p in pts:
                cv2.circle(clone, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Try to display at native resolution to avoid window scaling blur
    try:
        cv2.resizeWindow(win, img.shape[1], img.shape[0])
    except Exception:
        pass
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    print("Instructions: left-click to add points. Press 'r' to remove last, Enter or 'q' to finish.")
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(pts) > 0:
                pts.pop()
                clone = img.copy()
                for p in pts:
                    cv2.circle(clone, (int(p[0]), int(p[1])), 4, (0, 255, 0), -1)
                cv2.imshow(win, clone)
        if key == 13 or key == ord('q'):
            break
    cv2.destroyWindow(win)
    return pts


# ---------------- Save helpers ----------------

def save_rois_for_video(base_name, rois):
    ensure_dirs()
    path = os.path.join(ROI_DIR, base_name + "_rois.json")
    try:
        # ensure ints
        rois_i = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
        with open(path, 'w') as f:
            json.dump(rois_i, f)
        return path
    except Exception as e:
        raise


def save_corners_for_video(base_name, pts):
    ensure_dirs()
    path = os.path.join(CORNER_DIR, base_name + "_corners.json")
    try:
        pts_f = [[float(p[0]), float(p[1])] for p in pts]
        with open(path, 'w') as f:
            json.dump(pts_f, f)
        return path
    except Exception as e:
        raise


# ---------------- Main CLI ----------------

def has_display():
    # On Windows assume display available; on others check DISPLAY env
    if os.name == 'nt':
        return True
    return bool(os.environ.get('DISPLAY'))


def main():
    parser = argparse.ArgumentParser(description="Annotate ROIs and/or corner points for a video.")
    parser.add_argument('video', nargs='?', default=None, help='path to video file (default: input_videos/1.mp4)')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--rois', action='store_true', help='annotate ROIs only')
    group.add_argument('--corners', action='store_true', help='annotate corner points only')
    group.add_argument('--both', action='store_true', help='annotate both ROIs and corner points')
    args = parser.parse_args()

    video = args.video or os.path.join('input_videos', '1.mp4')
    if not os.path.exists(video):
        print(f"Video not found: {video}")
        sys.exit(1)

    print(f"Reading first frame from: {video}")
    try:
        first = read_first_frame(video)
    except Exception as e:
        print("Error reading first frame:", e)
        sys.exit(1)

    base = os.path.splitext(os.path.basename(video))[0]

    # If no display, save first frame and instruct
    if not has_display():
        out_img = base + "_firstframe.png"
        cv2.imwrite(out_img, first)
        print("No display available. Saved first frame as:", out_img)
        print("请在本地打开该图片，用任意图像工具标注 ROI 或角点，按需要保存 JSON 文件到:")
        print(f"  {os.path.abspath(os.path.join(ROI_DIR, base + '_rois.json'))}")
        print(f"  {os.path.abspath(os.path.join(CORNER_DIR, base + '_corners.json'))}")
        print("然后重新运行本脚本或直接运行 SeismicTracker_manual/auto。")
        sys.exit(0)

    # interactive
    # Save a full-resolution copy for debugging/inspection so user can check
    # whether the displayed window is being scaled by the OS/GUI.
    try:
        full_save = base + "_firstframe_fullres.png"
        cv2.imwrite(full_save, first)
        print(f"Saved full-resolution first frame for debug: {full_save}")
    except Exception:
        pass
    if args.rois or args.both:
        rois = select_rois_interactive(first)
        if rois is None:
            print("No ROIs selected")
        else:
            path = save_rois_for_video(base, rois)
            print(f"Saved ROIs to: {path}")

    if args.corners or args.both:
        pts = select_points_interactive(first)
        if pts is None or len(pts) == 0:
            print("No corner points selected")
        else:
            path = save_corners_for_video(base, pts)
            print(f"Saved corner points to: {path}")


if __name__ == '__main__':
    main()
