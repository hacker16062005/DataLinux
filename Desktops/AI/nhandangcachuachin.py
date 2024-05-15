'''Cài đặt môi trường nếu chưa có thư viện Opencv để làm bài toán
pip/pip3 install numpy   dùng pip hoặc pip3 cài numpy
pip search "opencv"
pip install opencv-python==3.4.2.16
'''
import cv2
import numpy as np

# Đọc hình ảnh
image = cv2.imread('cachua11.jpg')

# Chuyển đổi hình ảnh sang không gian màu HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Xác định ngưỡng màu của cà chua chín trong không gian màu HSV
lower_red = np.array([0, 100, 100])
upper_red = np.array([30, 255, 255])

# Tạo mặt nạ từng mảnh vùng màu của cà chua chín
mask = cv2.inRange(hsv_image, lower_red, upper_red)

# Áp dụng mặt nạ để lấy ra các vùng tương ứng
result = cv2.bitwise_and(image, image, mask=mask)

# Hiển thị hình ảnh kết quả
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
