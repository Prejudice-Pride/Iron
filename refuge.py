import pandas as pd
import numpy as np

def transform_tracks_to_waveform(input_csv, output_csv, fps=25.0):
    # 1. 读取原始数据
    df = pd.read_csv(input_csv)
    
    # 2. 获取所有点的 ID 和总帧数
    point_ids = sorted(df['id'].unique())
    frames = sorted(df['frame'].unique())
    
    # 3. 创建时间列
    time = [f / fps for f in frames]
    
    # 4. 初始化存储位移的字典
    # 逻辑：我们将存储相对于第一帧的相对位移 (relative displacement)
    result_data = {'time': time}
    
    all_dx = []
    all_dy = []
    
    # 5. 遍历每个点，计算其相对位移
    for i, pid in enumerate(point_ids):
        pt_df = df[df['id'] == pid].sort_values('frame')
        
        # 获取第一帧的坐标作为基准
        x0 = pt_df.iloc[0]['x']
        y0 = pt_df.iloc[0]['y']
        
        # 计算位移 (当前坐标 - 初始坐标)
        dx = pt_df['x'].values - x0
        dy = pt_df['y'].values - y0
        
        # 存入字典，列名为 pt{id}_x, pt{id}_y
        result_data[f'pt{i}_x'] = dx
        result_data[f'pt{i}_y'] = dy
        
        all_dx.append(dx)
        all_dy.append(dy)
    
    # 6. 计算统计指标 (均值和中位数)
    # 将列表转换为 numpy 矩阵以便计算: 形状为 (点数, 帧数)
    dx_matrix = np.array(all_dx)
    dy_matrix = np.array(all_dy)
    
    result_data['mean_x'] = np.mean(dx_matrix, axis=0)
    result_data['median_x'] = np.median(dx_matrix, axis=0)
    result_data['mean_y'] = np.mean(dy_matrix, axis=0)
    result_data['median_y'] = np.median(dy_matrix, axis=0)
    
    # 7. 整理列顺序 (按照要求的: time, mean_y, median_y, mean_x, median_x, pt0_x, pt0_y...)
    output_df = pd.DataFrame(result_data)
    
    cols = ['time', 'mean_y', 'median_y', 'mean_x', 'median_x']
    for i in range(len(point_ids)):
        cols.append(f'pt{i}_x')
        cols.append(f'pt{i}_y')
    
    output_df = output_df[cols]
    
    # 8. 保存文件
    output_df.to_csv(output_csv, index=False, float_format='%.6f')
    print(f"成功生成文件: {output_csv}")

# 使用示例
if __name__ == "__main__":
    # 修改以下路径为你实际的文件路径
    input_file = r"best\sample_p1_0.8848x_0.46y\sample_output_cancel\sample_reference_surface_tracks.csv"
    output_file = r"best\sample_p1_0.8848x_0.46y\sample_output_cancel\sample_raw_surface_waveform.csv"
    
    # 如果你的视频不是 25 帧，请修改 fps
    transform_tracks_to_waveform(input_file, output_file, fps=25.0)