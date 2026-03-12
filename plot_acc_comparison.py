import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

# 文件路径
txt_path = r"c:\Users\Administrator\Desktop\Princess\TheForge\Iron\txt_home\ID210711_room_trimed206.txt"
csv_path = r"c:\Users\Administrator\Desktop\Princess\TheForge\Iron\output\sample_output_cancel\sample_corrected_surface_waveform.csv"

def parse_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    start_idx = 0
    for i, line in enumerate(lines):
        if "时间" in line and "-EW-" in line:
            start_idx = i + 1
            break
            
    for line in lines[start_idx:]:
        parts = line.split()
        if len(parts) >= 4:
            date_str = parts[0] # 2025-3-26
            time_str = parts[1] # 1:17:0:0
            time_parts = time_str.split(':')
            if len(time_parts) == 4:
                try:
                    # 注意：strptime 中 %M:%S 是对应的，但小时部分需要手动解析或使用 24H 格式
                    h, m, s, ms = map(int, time_parts)
                    year, month, day = map(int, date_str.split('-'))
                    dt = datetime(year, month, day, h, m, s) + timedelta(milliseconds=ms)
                    
                    ew = float(parts[2])
                    ns = float(parts[3])
                    data.append({"time": dt, "EW_acc": ew, "NS_acc": ns})
                except:
                    continue
    return pd.DataFrame(data)

def calculate_acceleration(df, freq=25):
    """
    通过位移的二阶导数计算加速度
    a = (x[i+1] - 2*x[i] + x[i-1]) / dt^2
    """
    dt = 1.0 / freq
    cols = ['mean_y', 'median_y', 'mean_x', 'median_x']
    results = {}
    
    for col in cols:
        disp = df[col].values
        # 计算加速度 (单位: px/s^2)
        acc = np.diff(disp, n=2) / (dt**2)
        # 用 0 补齐两端以保持长度一致
        acc_padded = np.zeros(len(disp))
        acc_padded[1:-1] = acc
        results[col + '_acc'] = acc_padded
        
    return pd.DataFrame(results)

def parse_csv(file_path, start_time_str="2025-03-26 01:17:09", freq=25):
    df = pd.read_csv(file_path)
    # csv 中的 time 是从 0 开始的秒数
    base_time = datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
    df['real_time'] = df['time'].apply(lambda x: base_time + timedelta(seconds=x))
    
    # 计算加速度
    acc_df = calculate_acceleration(df, freq)
    df = pd.concat([df, acc_df], axis=1)
    return df

# 主程序
print("Loading and processing data...")
df_txt = parse_txt(txt_path)
# 用户提到 csv 绘制从 1:17:9:0 开始
df_csv = parse_csv(csv_path, "2025-03-26 01:17:09", freq=25)

# 过滤到用户指定的时间范围
start_time = datetime(2025, 3, 26, 1, 17, 0)
end_time = datetime(2025, 3, 26, 1, 17, 20)

df_txt_filtered = df_txt[(df_txt['time'] >= start_time) & (df_txt['time'] <= end_time)]
df_csv_filtered = df_csv[(df_csv['real_time'] >= start_time) & (df_csv['real_time'] <= end_time)]

# 绘图对比
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# 1. EW方向对比
ax1.plot(df_txt_filtered['time'], df_txt_filtered['EW_acc'], label='G1B EW Acc (g)', color='blue', alpha=0.6)
# 右侧坐标轴用于视频派生的加速度 (px/s^2)，因为单位量级可能不同
ax1_twin = ax1.twinx()
ax1_twin.plot(df_csv_filtered['real_time'], df_csv_filtered['mean_y_acc'], label='Video Y Acc (px/s$^2$)', color='red', alpha=0.8)
ax1.set_title('EW Direction: G1B Acc (Left) vs Video-Derived Acc (Right)')
ax1.set_ylabel('G1B Acc (g)')
ax1_twin.set_ylabel('Video Acc (px/s$^2$)')
ax1.grid(True)
# 合并图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_twin.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

# 2. NS方向对比
ax2.plot(df_txt_filtered['time'], df_txt_filtered['NS_acc'], label='G1B NS Acc (g)', color='green', alpha=0.6)
ax2_twin = ax2.twinx()
ax2_twin.plot(df_csv_filtered['real_time'], df_csv_filtered['mean_x_acc'], label='Video X Acc (px/s$^2$)', color='orange', alpha=0.8)
ax2.set_title('NS Direction: G1B Acc (Left) vs Video-Derived Acc (Right)')
ax2.set_ylabel('G1B Acc (g)')
ax2_twin.set_ylabel('Video Acc (px/s$^2$)')
ax2.grid(True)
# 合并图例
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.xlim(start_time, end_time)
plt.xlabel('Time (2025-03-26 1:17:00 to 1:17:20)')
plt.tight_layout()
plt.show()
