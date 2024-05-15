import cv2

# Đọc ảnh từ file
image = cv2.imread('image_chumauxanh.png')

# Chuyển ảnh sang định dạng HSV để dễ dàng phát hiện màu xanh
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Thiết lập phạm vi của màu xanh trong không gian màu HSV
lower_green = (60, 50, 50)
upper_green = (80, 255, 255)

# Tìm kiếm các vùng có màu nằm trong phạm vi đã thiết lập
mask = cv2.inRange(hsv, lower_green, upper_green)

# Áp dụng phép toán đóng và mở để loại bỏ nhiễu và kết nối các vùng có màu xanh gần nhau
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel)

# Tìm kiếm các contour trong ảnh đã xử lý
contours, hierarchy = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Duyệt qua các contour tìm được và vẽ hình chữ nhật xung quanh các contour có diện tích lớn hơn một ngưỡng nào đó (ở đây là 500)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Hiển thị ảnh kết quả
cv2.imshow('Result', image)
cv2.waitKey(0)
