import argparse
import cv2
import os

DEFAULT_VIDEO = r"output\qample2_output_cancel\tracking_vis_qample2.mp4"
DEFAULT_OUTPUT = r"output\qample2_output_cancel\first_frame.jpg"


def normalize_output_path(output_path: str) -> str:
    if not os.path.splitext(output_path)[1]:
        output_path = os.path.join(output_path, "first_frame.jpg") if os.path.isdir(output_path) or output_path.endswith(os.sep) else output_path + ".jpg"
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, "first_frame.jpg")
    return output_path


def extract_first_frame(video_path: str, output_path: str) -> bool:
    output_path = normalize_output_path(output_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return False

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("无法读取视频的第一帧。")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, frame)
    print(f"已保存第一帧到: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="提取视频的第一帧并保存为图像。")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="输入视频文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出图像文件路径，例如 first_frame.jpg")
    args = parser.parse_args()

    extract_first_frame(args.video, args.output)
