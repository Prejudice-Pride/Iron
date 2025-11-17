import ffmpeg

input_file = "input_videos\sample4.ts"
output_file = "sample4.mp4"

(
    ffmpeg
    .input(input_file)
    .output(output_file, vcodec='copy', acodec='copy')  # 尽量不重新编码，速度快
    .run()
)
