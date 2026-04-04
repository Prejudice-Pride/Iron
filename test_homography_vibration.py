import cv2
import numpy as np

def test_vibration():
    # qample2 src_pts (px) and dst_pts (mm)
    src_pts = np.array([
        [1123, 953], [1133, 951], [988, 810], [998, 809], 
        [1768, 833], [1773, 830], [1569, 759], [1574, 757], 
        [876, 687], [880, 686], [833, 642], [840, 642], 
        [1324, 667], [1327, 665], [1220, 627], [1224, 626]
    ], dtype=np.float32)

    dst_pts_mm = np.array([
        [0,0],[15,0],[0,60],[15,60],
        [865,0],[880,0],[865,60],[880,60],
        [0,150],[15,150],[0,210],[15,210],
        [865,150],[880,150],[865,210],[880,210]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts_mm, cv2.RANSAC, 5.0)

    # test 1 pixel shift at different base points
    test_points = np.array([
        [1123, 953], [1768, 833], [1324, 667], [500, 500]
    ], dtype=np.float32)

    # shifts to test
    shifts = [
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.1, 0.0]),
    ]

    for pt in test_points:
        print(f"\n--- Base point {pt} ---")
        pt_3d = np.array([pt[0], pt[1], 1.0])
        mapped = H @ pt_3d
        mapped = mapped[:2] / mapped[2] / 10.0 # to cm
        
        for shift in shifts:
            shifted_pt = pt + shift
            shifted_3d = np.array([shifted_pt[0], shifted_pt[1], 1.0])
            mapped_shifted = H @ shifted_3d
            mapped_shifted = mapped_shifted[:2] / mapped_shifted[2] / 10.0 # to cm
            
            diff_cm = mapped_shifted - mapped
            mag_cm = np.linalg.norm(diff_cm)
            print(f"Shift px {shift} -> diff cm {diff_cm}, magnitude {mag_cm:.6f} cm")

test_vibration()
