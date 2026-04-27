import cv2
import numpy as np
from scipy.optimize import least_squares
import os
import json

class SelfCalibration:
    def __init__(self, k3=0.0, k5=0.0, p1=0.0, p2=0.0):
        self.k3 = k3
        self.k5 = k5
        self.p1 = p1
        self.p2 = p2
        self.center = None

    def extract_curves(self, image, min_length=150, debug_folder=None, frame_idx=0):
        """
        提取图像中的直线/曲线
        1. Canny边缘检测
        2. 连通分量提取轮廓
        3. 长度过滤获得长直线/曲线
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # [DEBUG] 保存 Canny 边缘检测结果
        if debug_folder:
            if not os.path.exists(debug_folder): os.makedirs(debug_folder)
            cv2.imwrite(os.path.join(debug_folder, f"step1_canny_frame_{frame_idx}.jpg"), edges)

        # 寻找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        
        curves = []
        debug_vis = image.copy()
        for cnt in contours:
            if len(cnt) >= min_length:
                # 转换坐标为 (N, 2)
                curve = cnt.reshape(-1, 2).astype(np.float32)
                curves.append(curve)
                # [DEBUG] 在原图上画出保留的长曲线
                cv2.polylines(debug_vis, [cnt], False, (0, 255, 0), 2)
        
        # [DEBUG] 保存曲线提取可视化结果 (绿色为保留的有效曲线)
        if debug_folder:
            cv2.imwrite(os.path.join(debug_folder, f"step2_extracted_curves_{frame_idx}.jpg"), debug_vis)
        
        return curves

    def undistort_points(self, points, k3, k5, p1, p2, center):
        """
        应用径向和切向畸变模型进行校正
        """
        xd = points[:, 0] - center[0]
        yd = points[:, 1] - center[1]
        rd2 = xd**2 + yd**2
        rd4 = rd2**2
        
        # 径向畸变
        distortion = 1.0 - (k3 * rd2 + k5 * rd4)
        
        # 切向畸变
        # 减去切向畸变项，保持与径向畸变负号一致的逆向映射逻辑
        dx_tangential = -(2 * p1 * xd * yd + p2 * (rd2 + 2 * xd**2))
        dy_tangential = -(p1 * (rd2 + 2 * yd**2) + 2 * p2 * xd * yd)
        
        xu = xd * distortion + dx_tangential + center[0]
        yu = yd * distortion + dy_tangential + center[1]
        
        return np.stack([xu, yu], axis=1)

    def objective_function(self, params, curves, center):
        """
        目标函数：最小化校正后点到拟合直线的残差平方和
        """
        k3, k5, p1, p2 = params
        all_residuals = []
        
        for curve in curves:
            # 校正曲线
            undistorted_curve = self.undistort_points(curve, k3, k5, p1, p2, center)
            
            # 使用最小二乘法拟合直线
            line_params = cv2.fitLine(undistorted_curve, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line_params.flatten()
            
            # 计算点到直线的残差
            x = undistorted_curve[:, 0]
            y = undistorted_curve[:, 1]
            residuals = (x - x0) * vy - (y - y0) * vx
            all_residuals.extend(residuals.flatten())
            
        return np.array(all_residuals)

    def estimate_params(self, curves, img_shape):
        """
        基于提取的长直线集合估计 k3, k5, p1, p2
        """
        self.center = (img_shape[1] / 2.0, img_shape[0] / 2.0)
        initial_guess = [0.0, 0.0, 0.0, 0.0]
        
        # 限制搜索范围，防止出现极端扭曲
        res = least_squares(
            self.objective_function, 
            initial_guess, 
            args=(curves, self.center),
            bounds=([-1e-6, -1e-12, -1e-5, -1e-5], [1e-6, 1e-12, 1e-5, 1e-5]),
            ftol=1e-8
        )
        
        self.k3, self.k5, self.p1, self.p2 = res.x
        return self.k3, self.k5, self.p1, self.p2

    def evaluate_straightness(self, image, min_length=150):
        """
        评估图像的直线度（Straightness Error）
        返回：所有提取长直线中，像素点到各自拟合理想直线的平均距离（单位：像素）
        """
        curves = self.extract_curves(image, min_length=min_length)
        if not curves:
            return 0.0

        total_error = 0.0
        total_points = 0

        for curve in curves:
            # line_params = [vx, vy, x0, y0]
            line_params = cv2.fitLine(curve, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy, x0, y0 = line_params.flatten()
            
            x = curve[:, 0]
            y = curve[:, 1]
            
            # 使用向量叉乘的面积形式计算点到直线距离: 
            # 距离 = |(x-x0)*vy - (y-y0)*vx| / sqrt(vx^2 + vy^2)
            # 因为 fitLine 返回的是单位方向向量，所以 sqrt(vx^2 + vy^2) 约等于 1
            distances = np.abs((x - x0) * vy - (y - y0) * vx)
            
            total_error += np.sum(distances)
            total_points += len(curve)
            
        if total_points == 0:
            return 0.0
            
        mean_error = total_error / total_points
        return mean_error

    def apply_to_image(self, image):
        """
        全图像校正映射
        """
        h, w = image.shape[:2]
        
        # 优化：计算一次映射和填补掩码进行缓存，加速视频处理
        if not hasattr(self, '_map_initialized') or self._map_h != h or self._map_w != w:
            y, x = np.indices((h, w), dtype=np.float32)
            coords = np.stack([x.ravel(), y.ravel()], axis=1)
            
            if self.center is None:
                self.center = (w / 2.0, h / 2.0)
                
            undistorted = self.undistort_points(coords, self.k3, self.k5, self.p1, self.p2, self.center)
            
            ux = undistorted[:, 0]
            uy = undistorted[:, 1]
            
            u_int_x = np.round(ux).astype(int)
            u_int_y = np.round(uy).astype(int)
            
            valid = (u_int_x >= 0) & (u_int_x < w) & (u_int_y >= 0) & (u_int_y < h)
            grid_y, grid_x = y.ravel().astype(int), x.ravel().astype(int)
            
            self._dst_y = u_int_y[valid]
            self._dst_x = u_int_x[valid]
            self._src_y = grid_y[valid]
            self._src_x = grid_x[valid]
            self._map_h = h
            self._map_w = w
            self._map_initialized = True
            
            # 预热形态学核
            self._kernel = np.ones((3,3), np.uint8)

        new_img = np.zeros_like(image)
        new_img[self._dst_y, self._dst_x] = image[self._src_y, self._src_x]
        
        # 形态学填补空洞
        new_img = cv2.morphologyEx(new_img, cv2.MORPH_CLOSE, self._kernel)
        
        return new_img

    def save_results(self, folder="output/calibration_debug"):
        if not os.path.exists(folder):
            os.makedirs(folder)
        params_path = os.path.join(folder, "distortion_params.json")
        data = {
            "k3": float(self.k3),
            "k5": float(self.k5),
            "p1": float(self.p1),
            "p2": float(self.p2),
            "cx": float(self.center[0]),
            "cy": float(self.center[1])
        }
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
        print(f"标定参数已成功保存: {params_path}")

def run_self_calibration(video_path, output_dir="output/calibration_debug"):
    from utils.video_io import read_frames
    
    print(f"正在分析视频: {video_path}")
    frames, fps = read_frames(video_path)
    if not frames:
        print("错误: 无法加载视频帧")
        return None
    
    sample_frames = min(200, len(frames))
    indices = np.random.choice(sample_frames, size=min(10, sample_frames), replace=False)
    selected_frames = [frames[i] for i in indices]
    
    calibrator = SelfCalibration()
    
    # 2. 曲线提取与 Debug 图片输出
    print("正在提取几何特征曲线 (Curves Extraction)...")
    all_curves = []
    for i, frame in enumerate(selected_frames):
        curves = calibrator.extract_curves(frame, min_length=150, debug_folder=output_dir, frame_idx=i)
        all_curves.extend(curves)
    
    if len(all_curves) < 3:
        print("错误: 提取到的曲线特征不足")
        return None
    
    # 3. 参数分析
    print(f"正在反演畸变参数...")
    k3, k5, p1, p2 = calibrator.estimate_params(all_curves, frames[0].shape)
    print(f"\nK3: {k3:.8e}, K5: {k5:.8e}, P1: {p1:.8e}, P2: {p2:.8e}, Center: {calibrator.center}")
    
    # 4. 预览
    calibrator.save_results(output_dir)
    res = calibrator.apply_to_image(frames[0])
    combined = np.hstack([frames[0], res])
    cv2.imwrite(os.path.join(output_dir, "distort_calibration_preview.jpg"), combined)
    
    return calibrator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=str, help="输入视频路径")
    parser.add_argument("--out", type=str, default="output/calibration_debug", help="输出目录")
    args = parser.parse_args()
    run_self_calibration(args.video, args.out)
