import ffmpeg

input_file = "input_videos\sample.ts"
output_file = "output.mp4"

(
    ffmpeg
    .input(input_file)
    .output(output_file, vcodec='copy', acodec='copy')  # 尽量不重新编码，速度快
    .run()
)
