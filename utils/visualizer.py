import cv2
import numpy as np
import imageio.v3 as iio

def visualize_tracks(frames, tracks, visibilities, out_path, fps, rois=None):
    """
    Visualizes tracking results by drawing points and trails on video frames.
    
    Args:
        frames: (T, H, W, 3) RGB video frames
        tracks: (N, T, 2) Coordinates of tracked points
        visibilities: (N, T) Vsibilities of tracked points
        out_path: Path to save the output video
        fps: Frames per second
        rois: Optional list of ROIs to draw [x1, y1, x2, y2]
    """
    N, T, _ = tracks.shape
    # N points
    rng = np.random.RandomState(42)
    # Different colors for points, or uniform? 
    # Original script used random colors.
    colors = [tuple(int(c) for c in (rng.randint(50,255), rng.randint(50,255), rng.randint(50,255))) for _ in range(N)]
    
    vis = []
    print(f"Generating visualization for {T} frames...")
    
    for t in range(T):
        # Convert RGB to BGR for OpenCV drawing
        img = cv2.cvtColor(frames[t], cv2.COLOR_RGB2BGR)
        
        # Draw ROIs
        if rois:
            for r in rois:
                cv2.rectangle(img, (int(r[0]), int(r[1])), (int(r[2]), int(r[3])), (0, 255, 0), 2)
                
        for i in range(N):
            pts = tracks[i]
            # Draw trails
            valid = ~np.isnan(pts[:t+1,0])
            if np.count_nonzero(valid) >= 2:
                vp = pts[:t+1][valid].astype(np.int32)
                for k in range(len(vp)-1):
                    cv2.line(img, tuple(vp[k]), tuple(vp[k+1]), colors[i], 1)
            
            # Draw current point
            if not np.isnan(pts[t,0]):
                x,y = int(pts[t,0]), int(pts[t,1])
                cv2.circle(img, (x,y), 3, colors[i], -1)
        
        # Convert back to RGB for imageio
        vis.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Save video
    iio.imwrite(out_path, np.array(vis), plugin="FFMPEG", fps=fps, macro_block_size=1)
    return out_path
