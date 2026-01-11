"""
Crop the top-left region of a video using OpenCV and save to a new file.

Usage:
  python cut_top_left.py input.mp4 output.mp4 --w 640 --h 360 [--x 0 --y 0 --fourcc mp4v]

Defaults: crop from (x=0,y=0) with width/height provided. If w/h omitted, script
will exit with usage message.

python cut_top_left.py input_videos/sample.mp4 output/sample_top_left.mp4 --w 640 --h 360
ffmpeg -i input_videos/sample2.mp4 -filter:v "crop=640:360:0:0" -c:v ffv1 -level 3 -pix_fmt yuv444p -c:a copy output/sample2_top_left_lossless.mkv

This implementation uses cv2.VideoCapture and cv2.VideoWriter.
"""
import sys
import os
import argparse
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Crop top-left region of video using OpenCV")
    p.add_argument('input', help='input video path')
    p.add_argument('output', help='output video path')
    p.add_argument('--w', type=int, required=True, help='crop width in pixels')
    p.add_argument('--h', type=int, required=True, help='crop height in pixels')
    p.add_argument('--x', type=int, default=0, help='crop top-left x (default 0)')
    p.add_argument('--y', type=int, default=0, help='crop top-left y (default 0)')
    p.add_argument('--fourcc', type=str, default='mp4v', help='fourcc codec for output (default mp4v)')
    return p.parse_args()


def main():
    args = parse_args()
    inp = args.input
    out = args.output
    w = args.w
    h = args.h
    x = args.x
    y = args.y

    if not os.path.exists(inp):
        print(f"Input not found: {inp}")
        sys.exit(1)

    cap = cv2.VideoCapture(inp)
    if not cap.isOpened():
        print(f"Failed to open input: {inp}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    in_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    in_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Ensure crop inside bounds
    if x < 0 or y < 0:
        print("x and y must be >= 0")
        cap.release()
        sys.exit(1)
    if x >= in_w or y >= in_h:
        print("x/y outside frame")
        cap.release()
        sys.exit(1)

    w = min(w, in_w - x)
    h = min(h, in_h - y)

    print(f"Input: {inp}, frame size: {in_w}x{in_h}, fps: {fps}")
    print(f"Cropping region: x={x}, y={y}, w={w}, h={h}")

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    # Ensure output directory exists
    odir = os.path.dirname(out)
    if odir and not os.path.exists(odir):
        os.makedirs(odir, exist_ok=True)

    writer = cv2.VideoWriter(out, fourcc, float(fps), (w, h))
    if not writer.isOpened():
        print("Failed to open VideoWriter. Try different --fourcc (e.g., mp4v or XVID), or check ffmpeg installation.")
        cap.release()
        sys.exit(1)

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # frame is BGR (cv2 default)
        # Crop region
        crop = frame[y:y+h, x:x+w]
        # Ensure correct shape
        if crop.shape[0] != h or crop.shape[1] != w:
            # pad if necessary (shouldn't normally happen because we clamped w/h)
            crop = cv2.resize(crop, (w, h))
        writer.write(crop)
        frame_idx += 1
        if frame_idx % 200 == 0:
            print(f"  processed {frame_idx} frames...", end='\r')

    cap.release()
    writer.release()
    print(f"\nSaved cropped video to: {out}")


if __name__ == '__main__':
    main()
