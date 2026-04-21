import argparse
import cv2
import os

# python extract_first_frame.py --video input_videos/sample_p1.mp4 --frame 2 --output output/frame_2.jpg

DEFAULT_VIDEO = r"input_videos\sample_p1.mp4"
DEFAULT_FRAME = 1
DEFAULT_OUTPUT = rf"output\frame_{DEFAULT_FRAME}.jpg"


def normalize_output_path(output_path: str, frame_number: int) -> str:
    if not os.path.splitext(output_path)[1]:
        if os.path.isdir(output_path) or output_path.endswith(os.sep):
            output_path = os.path.join(output_path, f"frame_{frame_number}.jpg")
        else:
            output_path = output_path + ".jpg"
    elif os.path.isdir(output_path):
        output_path = os.path.join(output_path, f"frame_{frame_number}.jpg")
    return output_path


def extract_frame(video_path: str, output_path: str, frame_number: int = 1) -> bool:
    if frame_number < 1:
        print("帧编号必须是大于等于1的整数。")
        return False

    output_path = normalize_output_path(output_path, frame_number)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return False

    frame_index = frame_number - 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"无法读取视频的第 {frame_number} 帧。")
        return False

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_path, frame)
    print(f"已保存第 {frame_number} 帧到: {output_path}")
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="提取视频的指定帧并保存为图像。")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="输入视频文件路径")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出图像文件路径，例如 frame_1.jpg")
    parser.add_argument("--frame", type=int, default=DEFAULT_FRAME, help="要提取的帧编号（从1开始）")
    args = parser.parse_args()

    extract_frame(args.video, args.output, args.frame)
