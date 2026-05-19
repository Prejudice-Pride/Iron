import pandas as pd
import matplotlib.pyplot as plt
import os

# 定义文件路径
file_path = r'output\sample_output_cancel_full\sample_corrected_surface_waveform.csv'

# 读取CSV文件
try:
    df = pd.read_csv(file_path)


    # 创建画布，包含两个子图（上下分布）
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # 颜色区分
    eq1_color = '#90EE90'  # 浅绿色
    eq2_color = '#228B22'  # 深绿色

    # 绘制 median_x
    ax1.plot(df['time'], df['median_x'], color='blue', label='X-Axis')
    # 在 31s 和 71s 处做记号
    ax1.axvline(x=31, color=eq1_color, linestyle='--', label='Earthquake 1 (31s)')
    ax1.axvline(x=71, color=eq2_color, linestyle='--', label='Earthquake 2 (71s)')
    ax1.set_ylabel('Median X(cm)')
    ax1.set_title('Time Waveform: X-Axis and Y-Axis')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # 绘制 median_y
    ax2.plot(df['time'], df['median_y'], color='red', label='Y-Axis')
    # 在 31s 和 71s 处做记号
    ax2.axvline(x=31, color=eq1_color, linestyle='--', label='Earthquake 1 (31s)')
    ax2.axvline(x=71, color=eq2_color, linestyle='--', label='Earthquake 2 (71s)')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Median Y(cm)')
    ax2.grid(True)
    ax2.legend(loc='upper left')

    # 调整布局
    plt.tight_layout()

    # 保存图片到同一目录下
    output_img = os.path.join(os.path.dirname(file_path), 'median_waveform_plot.png')
    plt.savefig(output_img)
    print(f"图表已保存至: {output_img}")

    # 显示图表 (如果在支持GUI的环境下)
    plt.show()

except Exception as e:
    print(f"读取或绘制时出错: {e}")
