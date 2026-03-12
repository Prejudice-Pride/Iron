import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import pearsonr

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
            date_str = parts[0]
            time_parts = parts[1].split(':')
            if len(time_parts) == 4:
                try:
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
    dt = 1.0 / freq
    cols = ['mean_y', 'mean_x']
    results = {}
    for col in cols:
        disp = df[col].values
        acc = np.diff(disp, n=2) / (dt**2)
        acc_padded = np.zeros(len(disp))
        acc_padded[1:-1] = acc
        results[col + '_acc'] = acc_padded
    return pd.DataFrame(results)

print("Loading and processing data...")
df_txt = parse_txt(txt_path).set_index('time')
df_csv = pd.read_csv(csv_path)

# 构建 CSV 时间轴 (1:17:09 开始)
base_time = datetime(2025, 3, 26, 1, 17, 9)
df_csv['real_time'] = df_csv['time'].apply(lambda x: base_time + timedelta(seconds=x))
df_csv_acc = calculate_acceleration(df_csv, freq=25)
df_csv = pd.concat([df_csv, df_csv_acc], axis=1).set_index('real_time')

# 1. 对齐时间范围 (1:17:09 to 1:17:20)
start_overlap = datetime(2025, 3, 26, 1, 17, 9)
end_overlap = datetime(2025, 3, 26, 1, 17, 20)

# 2. 重采样 TXT (100Hz -> 25Hz) 以便直接进行数学对比
# 我们将 TXT 重采样到与 CSV 相同的时间戳上
df_txt_resampled = df_txt.sort_index().reindex(df_csv.index, method='nearest')

# 选取重叠时间段的数据
mask = (df_txt_resampled.index >= start_overlap) & (df_txt_resampled.index <= end_overlap)
data_txt = df_txt_resampled[mask]
data_csv = df_csv[mask]

def get_similarity(s1, s2):
    # 使用皮尔逊相关系数衡量波形相似度
    corr, _ = pearsonr(s1, s2)
    # 归一化均方误差 (NRMSE) 的变体或简单的 R2 倾向于捕捉波形一致性
    # 注意：由于单位不同，我们先对数据进行标准化 (Scale/Normalize) 再计算
    s1_norm = (s1 - np.mean(s1)) / np.std(s1)
    s2_norm = (s2 - np.mean(s2)) / np.std(s2)
    
    # 计算标准化后的相关性
    return corr

# 计算指标
ew_corr = get_similarity(data_txt['EW_acc'].values, data_csv['mean_y_acc'].values)
ns_corr = get_similarity(data_txt['NS_acc'].values, data_csv['mean_x_acc'].values)

print("-" * 30)
print(f"相似度指标 (皮尔逊相关系数 Pearson Correlation):")
print(f"EW 方向 (Y轴位移转换): {ew_corr:.4f}")
print(f"NS 方向 (X轴位移转换): {ns_corr:.4f}")
print("-" * 30)
print("提示: 相关系数越接近 1，表示波形走势越一致。")

# 绘图展示
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
ax1.plot(data_txt.index, data_txt['EW_acc'], label=f'G1B EW (Corr: {ew_corr:.3f})', alpha=0.7)
ax1_t = ax1.twinx()
ax1_t.plot(data_csv.index, data_csv['mean_y_acc'], color='red', label='Video Y Acc', alpha=0.5)
ax1.set_title('EW Direction Comparison')
ax1.legend(loc='upper left')

ax2.plot(data_txt.index, data_txt['NS_acc'], label=f'G1B NS (Corr: {ns_corr:.3f})', color='green', alpha=0.7)
ax2_t = ax2.twinx()
ax2_t.plot(data_csv.index, data_csv['mean_x_acc'], color='orange', label='Video X Acc', alpha=0.5)
ax2.set_title('NS Direction Comparison')
ax2.legend(loc='upper left')

plt.tight_layout()
plt.show()
