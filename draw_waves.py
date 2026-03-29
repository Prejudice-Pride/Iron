import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
df = pd.read_csv('output\points_sample\sample_raw_surface_tracks.csv')  # 请替换为实际文件路径
# df = pd.read_csv('output\point_sample\sample_raw_surface_tracks.csv') 
# 按id分组（如果有多个id，这里假设只有一个id=0）
for obj_id, group in df.groupby('id'):
    plt.figure(figsize=(12, 5))
    
    # 绘制x坐标随frame的变化
    plt.subplot(2, 1, 1)
    plt.plot(group['frame'], group['x'], 'b-', linewidth=1.5)
    plt.xlabel('Frame')
    plt.ylabel('X Coordinate')
    plt.title(f'Object {obj_id} - X vs Frame')
    plt.grid(True, alpha=0.3)
    
    # 绘制y坐标随frame的变化
    plt.subplot(2, 1, 2)
    plt.plot(group['frame'], group['y'], 'r-', linewidth=1.5)
    plt.xlabel('Frame')
    plt.ylabel('Y Coordinate')
    plt.title(f'Object {obj_id} - Y vs Frame')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()