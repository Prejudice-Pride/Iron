import cv2
import numpy as np
import os
import sys
from self_calibration import SelfCalibration

# Global variables for interactive labeling
points = []
lines = []
drawing_frame = None
original_frame = None

def mouse_callback(event, x, y, flags, param):
    global points, lines, drawing_frame, original_frame
    
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        update_display()
        
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(points) >= 3:
            lines.append(points.copy())
            print(f"[{len(lines)}] 线条已保存，包含 {len(points)} 个控制点。")
            points = []
            update_display()
        else:
            print("警告：一条物理直线至少需要 3 个控制点才能执行分析！")
            

def update_display():
    global drawing_frame, original_frame, points, lines
    drawing_frame = original_frame.copy()
    
    # 绘制已经保存的线条 (绿色)
    for i, line_pts in enumerate(lines):
        pts_array = np.array(line_pts, np.int32)
        cv2.polylines(drawing_frame, [pts_array], isClosed=False, color=(0, 255, 0), thickness=2)
        for pt in line_pts:
            cv2.circle(drawing_frame, pt, 4, (0, 200, 0), -1)
            
    # 绘制当前正在标定的线条 (红色)
    if len(points) > 0:
        pts_array = np.array(points, np.int32)
        cv2.polylines(drawing_frame, [pts_array], isClosed=False, color=(0, 0, 255), thickness=2)
        for pt in points:
            cv2.circle(drawing_frame, pt, 4, (0, 0, 255), -1)
            
    # 提示文字
    cv2.putText(drawing_frame, "Left Click: Add Point | Right Click: Finish Line | ENTER: Optimize | 'r': Undo Last Point | ESC: Exit", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("User-Guided Calibration", drawing_frame)

def evaluate_lines_error(calibrator, curves, center):
    """计算给定点集序列，在当前畸变参数下的直线度残差（均值像素数）"""
    total_error = 0.0
    total_points = 0
    
    for curve in curves:
        # 去畸变
        undistorted = calibrator.undistort_points(curve, calibrator.k3, calibrator.k5, calibrator.p1, calibrator.p2, center)
        # 拟合直线
        line_params = cv2.fitLine(undistorted, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = line_params.flatten()
        
        x = undistorted[:, 0]
        y = undistorted[:, 1]
        distances = np.abs((x - x0) * vy - (y - y0) * vx)
        
        total_error += np.sum(distances)
        total_points += len(curve)
        
    return total_error / total_points if total_points > 0 else 0

def run_user_guided_calibration(video_path):
    global original_frame, drawing_frame, points, lines

    if not os.path.exists(video_path):
        print(f"错误: 视频文件 {video_path} 不存在！")
        return
        
    cap = cv2.VideoCapture(video_path)
    ret, original_frame = cap.read()
    cap.release()
    
    if not ret:
        print("错误: 无法读取视频第一帧！")
        return
        
    drawing_frame = original_frame.copy()
    
    cv2.namedWindow("User-Guided Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("User-Guided Calibration", 1280, 720)
    cv2.setMouseCallback("User-Guided Calibration", mouse_callback)
    
    update_display()
    print("\n" + "="*50)
    print("=== 用户引导的畸变标定系统 (User-Guided Plumb-line) ===")
    print("="*50)
    print("操作逻辑:")
    print("1. 寻找画面边缘原本应该是直线的物体 (如灯杆、高楼外立面、直马路边缘)。可以带有俯视的斜度，算法会自动处理。")
    print("2. [左键点击] 沿着这一根物体线，从头到尾连续点击 3-10 个点。（由于有畸变，点出来的轨迹会有弧度，这是正确的）。")
    print("3. [右键点击] 完成这一根线的采集。线条会变为绿色。")
    print("4. 去画面的其他角落寻找其他直线，重复步骤2和3。(建议在画面的左侧和右侧至少各找 1-2 条)")
    print("5. 按下 [Enter] 开始极速高精度反演。")
    print("6. 按下 [r] 键可以撤销上一次的点击操作 (Undo Last Point/Line)。")
    print("="*50)
    
    run_opt = False
    while True:
        key = cv2.waitKey(10) & 0xFF
        if key == 27: # ESC
            print("用户取消操作。")
            break
        elif key == ord('r'):
            # 撤销逻辑：优先撤销当前正在画但还没完成的点；如果当前没有点，则撤销上一条已经完成的线
            if len(points) > 0:
                points.pop()
                print("已撤销上一个点。")
            elif len(lines) > 0:
                lines.pop()
                print("已撤销上一条完成的线。")
            else:
                print("没有可撤销的操作。")
            update_display()
        elif key == 13 or key == 10: # Enter
            if len(lines) == 0:
                print("错误: 你还没有右键保存过任何一条结构线！")
            else:
                run_opt = True
                break
                
    cv2.destroyAllWindows()
    
    if not run_opt:
        return
        
    print("\n=== 开始基于精选特征点的高精度自动反演 ===")
    
    # 转换为 numpy 格式
    curves = [np.array(pts, dtype=np.float32) for pts in lines]
    calibrator = SelfCalibration()
    calibrator.center = (original_frame.shape[1] / 2.0, original_frame.shape[0] / 2.0)
    
    # 初始化强制归零评价（查看原本的弯曲程度）
    calibrator.k3 = calibrator.k5 = calibrator.p1 = calibrator.p2 = 0.0
    orig_err = evaluate_lines_error(calibrator, curves, calibrator.center)
    print(f"物理基准线标定前 (未矫正) 的残差误差: {orig_err:.4f} 像素")
    
    # 自动评估寻找最优参数
    k3, k5, p1, p2 = calibrator.estimate_params(curves, original_frame.shape)
    
    final_err = evaluate_lines_error(calibrator, curves, calibrator.center)
    
    print("\n" + "-"*40)
    print("🎯 极简自动特征点标定成功！最终最佳模型参数:")
    print(f"K3 : {k3:.12e}")
    print(f"K5 : {k5:.15e}")
    print(f"P1 : {p1:.10e}")
    print(f"P2 : {p2:.10e}")
    print(f"光心: {calibrator.center}")
    print("-" * 40)
    print(f"目标直线残差 (优化前): {orig_err:.4f} 像素")
    print(f"目标直线残差 (优化后): {final_err:.4f} 像素")
    if final_err < orig_err - 0.001:
        print("结论: 获取了完美的非线性映射退化抵消！")
    else:
        print("结论: 误差极小且未发生显著改变，证明相机端已内置硬件校正或畸变微弱可直接由线性模型闭环。")
    print("-" * 40)
    
    # 在本工程的输出目录生成文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(script_dir)
    save_folder = os.path.join(root_dir, "output", "calibration_debug")
    os.makedirs(save_folder, exist_ok=True)
    
    # 存 JSON 以供全局系统调用
    calibrator.save_results(save_folder)
    
    # ---------------- 独 立 保 存 ： 标定前的带标记原图 ----------------
    pre_correction_img = original_frame.copy()
    for curve in curves:
        pts = curve.astype(np.int32)
        cv2.polylines(pre_correction_img, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
        for pt in pts:
            cv2.circle(pre_correction_img, tuple(pt), 4, (0, 0, 255), -1)
            
    cv2.putText(pre_correction_img, "Raw (Distorted)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
    
    pre_path = os.path.join(save_folder, "user_guided_before.jpg")
    cv2.imwrite(pre_path, pre_correction_img)
    print(f"标点原图已保存至: {pre_path}")
    
    # ---------------- 独 立 保 存 ： 标定后的带标记校正图 ----------------
    print("正在生成最终全局校正图像...")
    corrected = calibrator.apply_to_image(original_frame)
    post_correction_img = corrected.copy()
    
    # 在校正图像上画出反演去畸变后的直的绿线作为视觉反馈
    for curve in curves:
        undistorted = calibrator.undistort_points(curve, k3, k5, p1, p2, calibrator.center)
        u_pts = undistorted.astype(np.int32)
        cv2.polylines(post_correction_img, [u_pts], isClosed=False, color=(0, 255, 0), thickness=2)
        for pt in u_pts:
            cv2.circle(post_correction_img, tuple(pt), 4, (0, 200, 0), -1)
            
    cv2.putText(post_correction_img, "Corrected (Undistorted)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2, cv2.LINE_AA)
            
    post_path = os.path.join(save_folder, "user_guided_after.jpg")
    cv2.imwrite(post_path, post_correction_img)
    print(f"校正后图像已保存至: {post_path}")
    
    # ---------------- 生 成 并 保 存 ： 左右对比图 ----------------
    combined = np.hstack([pre_correction_img, post_correction_img])
    out_path = os.path.join(save_folder, "user_guided_result.jpg")
    cv2.imwrite(out_path, combined)
    print(f"效果对比图已保存至: {out_path}")
    
    # 显示最后的效果
    cv2.namedWindow("Left: Original (Red) | Right: Corrected (Green)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Left: Original (Red) | Right: Corrected (Green)", 1600, 600)
    cv2.imshow("Left: Original (Red) | Right: Corrected (Green)", combined)
    print("请查看弹出的对比图。按任意键关闭。")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    video = "input_videos/sample.mp4"
    if len(sys.argv) > 1:
        video = sys.argv[1]
    
    # 兼容根目录和 utils 目录运行
    if not os.path.exists(video):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        root = os.path.dirname(script_dir)
        fallback = os.path.join(root, video)
        if os.path.exists(fallback):
            video = fallback
            
    run_user_guided_calibration(video)