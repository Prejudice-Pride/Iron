import cv2
import numpy as np
import imageio.v3 as iio

def read_frames(path):
    """
    Reads all frames from a video file into a numpy array (T, H, W, 3) in RGB format.
    Also returns the FPS.
    """
    # iio.imread with plugin="FFMPEG" reads as RGB
    frames = iio.imread(path, plugin="FFMPEG")
    frames = np.array(frames)
    
    # Use OpenCV to get FPS and other metadata if needed
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    
    return frames, fps
