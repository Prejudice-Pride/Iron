"""
Interactive ROI Selector for Surface Tracking.

This script allows you to manually select the Region of Interest (ROI) for the "Surface Tracking" strategy.
It is designed to be run on a local machine (with a display) to generate the ROI JSON file, 
which can then be uploaded to a server where the main `SeismicTracker_surface.py` script runs.

Usage:
    python draw_surface_rois.py [video_file]

Output:
    Saves a JSON file to `rois/<video_name>_surface_rois.json`.
"""
import os
import sys
import json
import cv2
import imageio.v3 as iio
import numpy as np

# Configuration matching SeismicTracker_surface.py
INPUT_DIR = "input_videos"
ROI_DIR = "rois"
DEFAULT_VIDEO = "sample.mp4"

def load_surface_rois(base_name):
    """Load existing surface ROIs if they exist."""
    os.makedirs(ROI_DIR, exist_ok=True)
    path = os.path.join(ROI_DIR, base_name + "_surface_rois.json")
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                rois = json.load(f)
            # Ensure integers
            rois = [[int(r[0]), int(r[1]), int(r[2]), int(r[3])] for r in rois]
            return rois, path
        except Exception as e:
            print(f"Warning: Failed to load existing ROIs: {e}")
            return None, path
    return None, path

def save_surface_rois(path, rois):
    """Save ROIs to JSON file."""
    try:
        with open(path, 'w') as f:
            json.dump(rois, f)
        print(f"Successfully saved {len(rois)} ROIs to: {path}")
        return True
    except Exception as e:
        print(f"Error saving ROIs: {e}")
        return False

def select_surface_rois_interactive(img, existing_rois=None):
    """
    Interactive GUI to draw ROIs.
    Green boxes indicate the selected surface.
    """
    win = "Select Object Surface (Green) - Drag to draw, 'r' to remove last, Enter/q to finish"
    clone = img.copy() # img is expected to be BGR
    rois = list(existing_rois) if existing_rois else []
    current = None

    def on_mouse(event, x, y, flags, param):
        nonlocal clone, current
        if event == cv2.EVENT_LBUTTONDOWN:
            current = (x, y, x, y)
        elif event == cv2.EVENT_MOUSEMOVE and current is not None:
            sx, sy, _, _ = current
            clone = img.copy()
            # Draw current being dragged
            cv2.rectangle(clone, (sx, sy), (x, y), (0, 255, 0), 2)
            # Draw already selected
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.imshow(win, clone)
        elif event == cv2.EVENT_LBUTTONUP and current is not None:
            sx, sy, _, _ = current
            x1, y1 = min(sx, x), min(sy, y)
            x2, y2 = max(sx, x), max(sy, y)
            # Minimum size check
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                rois.append([int(x1), int(y1), int(x2), int(y2)])
            current = None
            clone = img.copy()
            for r in rois:
                cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
            cv2.imshow(win, clone)

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, clone)
    cv2.setMouseCallback(win, on_mouse)
    
    # Draw initial state
    for r in rois:
        cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
    cv2.imshow(win, clone)

    print("-" * 50)
    print("INTERACTIVE MODE INSTRUCTIONS:")
    print("1. Left-click and drag to draw a GREEN box around the object surface.")
    print("2. You can draw multiple boxes if the surface is disjoint.")
    print("3. Press 'r' to remove the last drawn box.")
    print("4. Press 'Enter' or 'q' to save and exit.")
    print("-" * 50)
    
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key == ord('r'):
            if len(rois) > 0:
                rois.pop()
                clone = img.copy()
                for r in rois:
                    cv2.rectangle(clone, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
                cv2.imshow(win, clone)
            else:
                print("No ROIs to remove.")
        if key == 13 or key == ord('q'): # Enter or q
            break
            
    cv2.destroyWindow(win)
    return rois

def main():
    # 1. Determine input video path
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        # Check if it's a direct path or in INPUT_DIR
        if not os.path.exists(inp):
            candidate = os.path.join(INPUT_DIR, inp)
            if os.path.exists(candidate):
                inp = candidate
    else:
        inp = os.path.join(INPUT_DIR, DEFAULT_VIDEO)
    
    if not os.path.exists(inp):
        print(f"Error: Video file not found: {inp}")
        print(f"Usage: python draw_surface_rois.py [video_filename]")
        return

    print(f"Processing video: {inp}")

    # 2. Read the first frame
    try:
        # imageio reads as RGB
        frame0 = iio.imread(inp, index=0, plugin="FFMPEG")
        # Convert to BGR for OpenCV
        frame0_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error reading video frame: {e}")
        return

    # 3. Load existing ROIs or start fresh
    base = os.path.splitext(os.path.basename(inp))[0]
    existing_rois, roi_path = load_surface_rois(base)
    
    print(f"ROI file target: {roi_path}")
    if existing_rois:
        print(f"Found {len(existing_rois)} existing ROIs. You can edit them.")
    else:
        print("No existing ROIs found. Starting fresh.")

    # 4. Open Interactive Selector
    final_rois = select_surface_rois_interactive(frame0_bgr, existing_rois)
    
    # 5. Save result
    if final_rois:
        save_surface_rois(roi_path, final_rois)
    else:
        print("No ROIs selected. Exiting without saving.")
        return

    print(f"Reading {inp}...")
    # Read first frame only
    try:
        # imageio reads RGB
        frame0 = iio.imread(inp, index=0, plugin="FFMPEG")
        # Convert to BGR for OpenCV
        frame0_bgr = cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Failed to read video: {e}")
        return

    base = os.path.splitext(os.path.basename(inp))[0]
    existing_rois, roi_path = load_surface_rois(base)
    
    print(f"ROI file path: {roi_path}")
    if existing_rois:
        print(f"Found {len(existing_rois)} existing ROIs.")
        # Draw existing
        for r in existing_rois:
            cv2.rectangle(frame0_bgr, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 2)
    
    print("Opening interactive window...")
    new_rois = select_surface_rois_interactive(frame0_bgr)
    
    if new_rois:
        if save_surface_rois(roi_path, new_rois):
            print(f"Saved {len(new_rois)} ROIs to {roi_path}")
        else:
            print("Failed to save ROIs.")
    else:
        print("No ROIs selected. Nothing saved.")

if __name__ == "__main__":
    main()
