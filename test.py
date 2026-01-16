import cv2
import numpy as np
# 回调函数：当鼠标事件发生时调用
def mouse_callback(event, x, y, flags, userdata):
   # 如果检测到左键点击
   if event == cv2.EVENT_LBUTTONDOWN:
       print(f'点击坐标: ({x}, {y})')
       # 在图像上绘制一个点
       cv2.circle(userdata, (x, y), 3, (0, 0, 255), -1)
       # 显示坐标
       cv2.putText(userdata, f'({x},{y})', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
# 读取图像
img = cv2.imread('horse_riding.jpg')
# 创建窗口
cv2.namedWindow('Image')
# 设置鼠标回调函数
cv2.setMouseCallback('Image', mouse_callback, img)
# 显示图像
while True:
   cv2.imshow('Image', img)
   # 按ESC键退出
   if cv2.waitKey(1) & 0xFF == 27:
       break
# 关闭所有窗口
cv2.destroyAllWindows()