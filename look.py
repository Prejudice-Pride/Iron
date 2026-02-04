import cv2
import numpy as np

def _interactive_select_coords(img):
    points = []
    clone = img.copy()

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Clicked at: {x}, {y}")
            # Draw a circle on the image
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Select Coordinates", clone)

    cv2.namedWindow("Select Coordinates")
    cv2.setMouseCallback("Select Coordinates", mouse_callback)
    
    print("Instruction:")
    print("  - Click on the image to select a point.")
    print("  - Press 'r' to undo the last point.")
    print("  - Press 'q' or 'Esc' to finish selecting and return the coordinates.")
    
    cv2.imshow("Select Coordinates", img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if points:
                points.pop()
                print(f"Removed last point. Remaining: {len(points)}")
                clone = img.copy()
                for p in points:
                    cv2.circle(clone, tuple(p), 5, (0, 0, 255), -1)
                cv2.imshow("Select Coordinates", clone)
        elif key == ord('q') or key == 27:  # q or Esc
            break
            
    cv2.destroyAllWindows()
    return points

def select_coords(image_path):
    """
    Open an image and allow the user to select points by clicking.
    Args:
        image_path (str): Path to the image file.
    Returns:
        list: A list of selected coordinates [(x, y), ...].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")
    
    return _interactive_select_coords(img)


def _interactive_select_boxes(img):
    boxes = []
    current_img = img.copy()  # Accumulates drawn boxes
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, current_img

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_img = current_img.copy()
                cv2.rectangle(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Boxes", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x_min, y_min = min(ix, x), min(iy, y)
            x_max, y_max = max(ix, x), max(iy, y)
            
            # Ignore zero-size boxes
            if x_max > x_min and y_max > y_min:
                box = [x_min, y_min, x_max, y_max]
                boxes.append(box)
                print(f"Box added: {box}")
                
                # Draw confirmed box
                cv2.rectangle(current_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.imshow("Select Boxes", current_img)

    cv2.namedWindow("Select Boxes")
    cv2.setMouseCallback("Select Boxes", mouse_callback)
    
    print("\nInstruction:")
    print("  - Drag mouse to draw a box.")
    print("  - Press 'r' to undo the last box.")
    print("  - Press 'q' or 'Esc' to finish selecting.")
    
    cv2.imshow("Select Boxes", img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if boxes:
                boxes.pop()
                print(f"Removed last box. Remaining: {len(boxes)}")
                current_img = img.copy()
                for box in boxes:
                    cv2.rectangle(current_img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                cv2.imshow("Select Boxes", current_img)
        elif key == ord('q') or key == 27:  # q or Esc
            break
            
    cv2.destroyAllWindows()
    return boxes

def select_boxes(image_path):
    """
    Open an image and allow the user to select bounding boxes by dragging.
    Args:
        image_path (str): Path to the image file.
    Returns:
        list: A list of selected boxes [[x_min, y_min, x_max, y_max], ...].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")
    return _interactive_select_boxes(img)


def _interactive_select_lines(img):
    lines = []
    current_img = img.copy()  # Accumulates drawn lines
    drawing = False
    ix, iy = -1, -1

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, current_img

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_img = current_img.copy()
                cv2.line(temp_img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Lines", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            # Ignore zero-length lines
            if abs(x - ix) > 0 or abs(y - iy) > 0:
                line = [ix, iy, x, y]
                lines.append(line)
                print(f"Line added: {line}")
                
                # Draw confirmed line
                cv2.line(current_img, (ix, iy), (x, y), (0, 255, 0), 2)
                cv2.imshow("Select Lines", current_img)

    cv2.namedWindow("Select Lines")
    cv2.setMouseCallback("Select Lines", mouse_callback)
    
    print("\nInstruction:")
    print("  - Drag mouse to draw a line.")
    print("  - Press 'r' to undo the last line.")
    print("  - Press 'q' or 'Esc' to finish selecting.")
    
    cv2.imshow("Select Lines", img)
    
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('r'):
            if lines:
                lines.pop()
                print(f"Removed last line. Remaining: {len(lines)}")
                current_img = img.copy()
                for line in lines:
                    cv2.line(current_img, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 2)
                cv2.imshow("Select Lines", current_img)
        elif key == ord('q') or key == 27:  # q or Esc
            break
            
    cv2.destroyAllWindows()
    return lines

def select_lines(image_path):
    """
    Open an image and allow the user to select lines by dragging.
    Args:
        image_path (str): Path to the image file.
    Returns:
        list: A list of selected lines [[x1, y1, x2, y2], ...].
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not open image: {image_path}")
    return _interactive_select_lines(img)


def _load_media_first_frame(media_path):
    # 1. Try to read as image
    img = cv2.imread(media_path)
    if img is not None:
        return img
    
    # 2. Try to read as video
    cap = cv2.VideoCapture(media_path)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret:
            # frame is the first frame
            return frame
            
    # If both fail
    raise ValueError(f"Could not open media (image or video): {media_path}")

def select_coords_from_media(media_path):
    """
    Open an image or video (first frame) and allow the user to select points.
    Args:
        media_path (str): Path to the image or video file.
    Returns:
        list: A list of selected coordinates [(x, y), ...].
    """
    img = _load_media_first_frame(media_path)
    return _interactive_select_coords(img)

def select_boxes_from_media(media_path):
    """
    Open an image or video (first frame) and allow the user to select boxes.
    Args:
        media_path (str): Path to the image or video file.
    Returns:
        list: A list of selected boxes [[x_min, y_min, x_max, y_max], ...].
    """
    img = _load_media_first_frame(media_path)
    return _interactive_select_boxes(img)

def select_lines_from_media(media_path):
    """
    Open an image or video (first frame) and allow the user to select lines.
    Args:
        media_path (str): Path to the image or video file.
    Returns:
        list: A list of selected lines [[x1, y1, x2, y2], ...].
    """
    img = _load_media_first_frame(media_path)
    return _interactive_select_lines(img)


def main():
    image_path = "input_videos\sample.mp4"
    # points = select_coords_from_media(image_path)
    # boxes = select_boxes_from_media(image_path)
    lines = select_lines_from_media(image_path)
    print("All selected boxes:", lines)

if __name__ == '__main__':
    main()