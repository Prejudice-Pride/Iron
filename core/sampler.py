import os
import json
import cv2
import numpy as np
import sys

class Sampler:
    def __init__(self, roi_dir="rois"):
        self.roi_dir = roi_dir
        os.makedirs(self.roi_dir, exist_ok=True)

    def load_rois(self, base_name, roi_suffix="_surface_rois.json"):
        path = os.path.join(self.roi_dir, base_name + roi_suffix)
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    rois = json.load(f)
                # Ensure integer coordinates
                rois = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
                return rois, path
            except Exception:
                return None, path
        return None, path

    def get_or_select_rois(self, base_name, frame_first):
        """
        Load ROIs from file, or interactively select if not found.
        Returns ROIs list or None if headless and no file.
        """
        rois, roi_path = self.load_rois(base_name)
        if rois is None:
            can_interactive = bool(os.environ.get('DISPLAY')) or (os.name == 'nt')
            if can_interactive:
                print("No Surface ROI file found. Please select object surface.")
                rois = self.select_rois_interactive(frame_first)
                if rois and len(rois) > 0:
                    self.save_rois(roi_path, rois)
                else:
                    print("No ROIs selected. Using full image.")
                    return None
            else:
                print("Headless mode: Cannot select ROI interactively.")
                return None
        else:
            print(f"Loaded ROIs from {roi_path}")
        return rois

    def save_rois(self, path, rois):
        try:
            with open(path, 'w') as f:
                json.dump(rois, f)
            return True
        except Exception:
            return False

    def select_rois_interactive(self, img, window_name="Select Object Surface"):
        """
        Interactively select ROIs on an image.
        Returns a list of [x1, y1, x2, y2].
        """
        win = f"{window_name} - drag to draw, r to remove last, q/Enter to finish"
        clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rois = []
        current = None

        def on_mouse(event, x, y, flags, param):
            nonlocal clone, current, rois
            temp_img = clone.copy()
            
            if event == cv2.EVENT_LBUTTONDOWN:
                current = (x, y)
            
            elif event == cv2.EVENT_MOUSEMOVE:
                if current is not None:
                    cv2.rectangle(temp_img, current, (x, y), (0, 255, 0), 2)
                for r in rois:
                    cv2.rectangle(temp_img, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.imshow(win, temp_img)
                
            elif event == cv2.EVENT_LBUTTONUP and current is not None:
                sx, sy = current
                x1, y1 = min(sx, x), min(sy, y)
                x2, y2 = max(sx, x), max(sy, y)
                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    rois.append([int(x1), int(y1), int(x2), int(y2)])
                current = None
                # Update base clone
                cv2.rectangle(clone, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow(win, clone)

        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.imshow(win, clone)
        cv2.setMouseCallback(win, on_mouse)
        
        print("Instructions: Drag mouse to draw the OBJECT SURFACE (Green Box).")
        print("Points will be detected INSIDE this box.")
        print("Press 'r' to remove last box, Enter or 'q' to finish.")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                # Reset logic needs original image
                if len(rois) > 0:
                    rois.pop()
                    clone = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    for r in rois:
                        cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                    cv2.imshow(win, clone)
            elif key == 13 or key == ord('q'): # Enter or q
                break
        
        cv2.destroyWindow(win)
        return rois

    def detect_roi_points(self, frame_rgb, rois, max_points=30, quality=0.01, min_dist=7):
        """
        使用 cv2.goodFeaturesToTrack 在 ROI 区域内检测角点。
        """
        H, W = frame_rgb.shape[:2]
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        
        # 创建掩码
        mask = np.zeros((H, W), dtype=np.uint8)
        if not rois or len(rois) == 0:
            mask[:] = 255
        else:
            for r in rois:
                x1, y1, x2, y2 = map(int, r)
                x1, x2 = max(0, min(W-1, x1)), max(0, min(W-1, x2))
                y1, y2 = max(0, min(H-1, y1)), max(0, min(H-1, y2))
                mask[y1:y2+1, x1:x2+1] = 255

        # 检测角点
        pts = cv2.goodFeaturesToTrack(
            gray, 
            maxCorners=max_points, 
            qualityLevel=quality, 
            minDistance=min_dist, 
            mask=mask
        )

        if pts is None:
            return np.empty((0, 2), dtype=np.float32)
        
        return pts.reshape(-1, 2).astype(np.float32)
