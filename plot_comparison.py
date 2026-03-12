import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

# 文件路径
txt_path = r"c:\Users\Administrator\Desktop\Princess\TheForge\Iron\txt_home\ID210711_room_trimed206.txt"
csv_path = r"c:\Users\Administrator\Desktop\Princess\TheForge\Iron\output\sample_output_cancel\sample_corrected_surface_waveform.csv"

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    # 跳过头部信息，找到数据开始的行
    start_idx = 0
    date_part = ""
    for i, line in enumerate(lines):
        if "时间" in line and "-EW-" in line:
            start_idx = i + 1
            break
            
    # 从数据行中找第一行的时间，作为基准
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            # 格式例如 2025-3-26 1:17:0:0
            date_str = parts[0] # 2025-3-26
            time_str = parts[1] # 1:17:0:0
            time_parts = time_str.split(':')
            if len(time_parts) == 4:
                # 假设时间格式 H:M:S:MS
                # 这里的 MS 是十分之一秒或者是毫秒？ 
                # 从 [SPEED] 100 来看，采样频率是 100Hz，即 0.01 秒
                # 1:17:0:0 到 1:17:0:10 步长是 10，如果单位是 ms，10ms = 0.01s (1e-2 s)
                # 符合 100Hz 的频率
                try:
                    dt_base = datetime.strptime(f"{date_str} {time_parts[0]}:{time_parts[1]}:{time_parts[2]}", "%Y-%m-%d %H:%M:%S")
                    dt = dt_base + timedelta(milliseconds=int(time_parts[3]))
                    
                    ew = float(parts[2])
                    ns = float(parts[3])
                    data.append({"time": dt, "EW_acc": ew, "NS_acc": ns})
                except Exception as e:
                    continue
    return pd.DataFrame(data)

def parse_csv(file_path, start_time_str="2025-03-26 01:17:19"):
    df = pd.read_csv(file_path)
    # csv 中的 time 是从 0 开始的秒数
    base_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    df['real_time'] = df['time'].apply(lambda x: base_time + timedelta(seconds=x))
    return df

# 加载数据
print("Loading TXT data...")
df_txt = parse_txt(txt_path)
print("Loading CSV data...")
# 注意：TXT中的日期是2025-3-26，用户提到的视频开始时间是1:17:19
df_csv = parse_csv(csv_path, "2025-03-26 01:17:19")

# 绘图
fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 1. EW方向对比 (TXT 加速度 vs CSV y位移)
axes[0].plot(df_txt['time'], df_txt['EW_acc'], label='G1B EW Acceleration (g)', color='blue', alpha=0.7)
axes[0].set_ylabel('Acc (g)')
axes[0].set_title('EW Direction: Acceleration (Strong Motion Instrument)')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(df_csv['real_time'], df_csv['mean_y'], label='Video mean_y', color='orange')
axes[1].plot(df_csv['real_time'], df_csv['median_y'], label='Video median_y', color='red', linestyle='--')
axes[1].set_ylabel('Displacement (px)')
axes[1].set_title('EW Direction (Y): Displacement (Video Tracking)')
axes[1].legend()
axes[1].grid(True)

# 2. NS方向对比 (TXT 加速度 vs CSV x位移)
axes[2].plot(df_txt['time'], df_txt['NS_acc'], label='G1B NS Acceleration (g)', color='green', alpha=0.7)
axes[2].set_ylabel('Acc (g)')
axes[2].set_title('NS Direction: Acceleration (Strong Motion Instrument)')
axes[2].legend()
axes[2].grid(True)

axes[3].plot(df_csv['real_time'], df_csv['mean_x'], label='Video mean_x', color='purple')
axes[3].plot(df_csv['real_time'], df_csv['median_x'], label='Video median_x', color='brown', linestyle='--')
axes[3].set_ylabel('Displacement (px)')
axes[3].set_title('NS Direction (X): Displacement (Video Tracking)')
axes[3].legend()
axes[3].grid(True)

# 设置共享的X轴范围 (对齐时间)
start_plot = max(df_txt['time'].min(), df_csv['real_time'].min())
end_plot = min(df_txt['time'].max(), df_csv['real_time'].max())
if start_plot < end_plot:
    plt.xlim(start_plot, end_plot)
else:
    print(f"Warning: No overlapping time range found. TXT range: {df_txt['time'].min()} to {df_txt['time'].max()}, CSV range: {df_csv['real_time'].min()} to {df_csv['real_time'].max()}")
    # 如果没有交集，可以显示完整范围
    plt.xlim(df_csv['real_time'].min(), df_csv['real_time'].max())

plt.xlabel('Time (HH:MM:SS)')
plt.tight_layout()
plt.show()

# 如果需要保存图片
# plt.savefig('waveform_comparison.png', dpi=300)
