import cv2
import numpy as np

def verify_homography():
    print("="*60)
    print(" 单应性变换 (Homography) 独立验证工具 ")
    print("="*60)

    # 1. 提取自 main_cancel.py 中的 qample2 标定数据
    src_pts = np.array([
        [1123, 953], [1133, 951], [988, 810], [998, 809], 
        [1768, 833], [1773, 830], [1569, 759], [1574, 757], 
        [876, 687], [880, 686], [833, 642], [840, 642], 
        [1324, 667], [1327, 665], [1220, 627], [1224, 626]
    ], dtype=np.float32)

    # 设定的物理坐标 (单位：毫米 mm)
    dst_pts_mm = np.array([
        [0,0],[15,0],[0,60],[15,60],
        [865,0],[880,0],[865,60],[880,60],
        [0,150],[15,150],[0,210],[15,210],
        [865,150],[880,150],[865,210],[880,210]
    ], dtype=np.float32)

    # 我们期望输出的单位是厘米 (cm)
    dst_pts_cm = dst_pts_mm / 10.0

    print(f"\n[1] 开始计算单应矩阵 (点对数量: {len(src_pts)})")
    # 使用 cv2.RANSAC 具有抗噪性
    H, mask = cv2.findHomography(src_pts, dst_pts_mm, cv2.RANSAC, 5.0)
    
    print("\n计算得到的 3x3 单应性矩阵 (映射到 mm):")
    print(np.round(H, 4))
    
    # 2. 核心比对：重投影测试 (Reprojection)
    print("\n[2] 重投影测试 (将源像素代入矩阵计算，并缩小10倍转为 cm，验证是否等于设定的 cm 值)")
    
    # 格式对齐 cv2 需求： (N, 1, 2)
    src_reshaped = src_pts.reshape(-1, 1, 2)
    projected_pts_mm = cv2.perspectiveTransform(src_reshaped, H).reshape(-1, 2)
    projected_pts_cm = projected_pts_mm / 10.0
    
    print(f"{'源像素点 (px)':<18} | {'数学映射出物理坐标 (cm)':<25} | {'我们预设期望的坐标 (cm)':<25} | {'误差 (cm)'}")
    print("-" * 85)
    
    errors = []
    for i in range(len(src_pts)):
        p_px = src_pts[i]
        p_cm = projected_pts_cm[i]
        target_cm = dst_pts_cm[i]
        
        # 计算欧氏距离误差
        diff = np.linalg.norm(p_cm - target_cm)
        errors.append(diff)
        
        print(f"({p_px[0]:4.0f}, {p_px[1]:4.0f})   |   ({p_cm[0]:7.2f}, {p_cm[1]:7.2f})        |   ({target_cm[0]:7.2f}, {target_cm[1]:7.2f})       |  {diff:.4f}")

    print(f"\n=> 最大重投影误差: {np.max(errors):.4f} cm")
    print(f"=> 平均重投影误差: {np.mean(errors):.4f} cm")
    
    if np.mean(errors) < 0.5:
        print("[结论] 标定点完美映射！单应性网格有效没有发生扭曲变形。")
    else:
        print("[警告] 误差过大，可能是你人工框选的源坐标(Pixels)不共面，导致发生了三维透视扭曲。")

    # 3. 手动推演矩阵乘法验证底层数学 (针对第一个标定点)
    print("\n[3] 手动数学推演验证 (底层验证)")
    x, y = src_pts[0]
    # 构建齐次座标 [x, y, 1]^T
    vec = np.array([x, y, 1.0])
    # 矩阵乘法
    res = np.dot(H, vec)
    # 齐次坐标除以最后一位 Z 归一化
    res_normalized = res / res[2]
    res_cm = res_normalized[:2] / 10.0
    print(f"手算 {x}, {y} 的物理坐标: X = {res_cm[0]:.4f} cm, Y = {res_cm[1]:.4f} cm")
    print(f"对比库函数算出的坐标: X = {projected_pts_cm[0][0]:.4f} cm, Y = {projected_pts_cm[0][1]:.4f} cm")
    print("=两相一致则证明代码逻辑(Pixel -> _cm)转换公式完全正确。=")

if __name__ == "__main__":
    verify_homography()
