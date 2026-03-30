import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件路径
file_path = r'c:\Users\Administrator\Desktop\Princess\TheForge\Iron\output\032902_sample_output_cancel\sample_corrected_surface_waveform.csv'

# 读取CSV文件
try:
    df = pd.read_csv(file_path)

    # 创建画布，包含两个子图（上下分布）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 绘制 median_y
    ax1.plot(df['time'], df['median_y'], color='blue', label='Y-Axis')
    ax1.set_ylabel('Median Y')
    ax1.set_title('Time Waveform: Y-Axis and X-Axis')
    ax1.grid(True)
    ax1.legend()

    # 绘制 median_x
    ax2.plot(df['time'], df['median_x'], color='red', label='X-Axis')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Median X')
    ax2.grid(True)
    ax2.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图片到同一目录下
    output_img = os.path.join(os.path.dirname(file_path), 'median_waveform_plot.png')
    plt.savefig(output_img)
    print(f"图表已保存至: {output_img}")

    # 显示图表
    plt.show()

except Exception as e:
    print(f"读取或绘制时出错: {e}")