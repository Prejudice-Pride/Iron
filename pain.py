import pandas as pd
import matplotlib.pyplot as plt

# ====== 1. 读取数据 ======
file_path = "output\\autodl_sample_output_cancel\sample_corrected_surface_waveform.csv"  # ← 改成你的CSV文件路径
df = pd.read_csv(file_path)

# ====== 2. 提取数据 ======
time = df["time"]

mean_x = df["mean_x"]
mean_y = df["mean_y"]

median_x = df["median_x"]
median_y = df["median_y"]

# ====== 3. 绘图 ======
plt.figure()

# 平均值曲线
plt.plot(time, mean_x, label="Mean X")
plt.plot(time, mean_y, label="Mean Y")

# 中位数曲线
plt.plot(time, median_x, linestyle='--', label="Median X")
plt.plot(time, median_y, linestyle='--', label="Median Y")

# ====== 4. 图像设置 ======
plt.xlabel("Time")
plt.ylabel("Displacement")
plt.title("Time-Displacement Curve (Mean & Median)")
plt.legend()
plt.grid()

# ====== 5. 显示 ======
plt.show()