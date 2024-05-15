import numpy as np
from sklearn.linear_model import LinearRegression

# Dữ liệu mẫu
# X là ma trận đặc trưng, mỗi hàng là một sinh viên và mỗi cột là một yếu tố ảnh hưởng đến kết quả học (ví dụ: số giờ ôn tập, điểm trung bình các kỳ trước, v.v.)
X = np.array([[5, 7.5], [6, 8], [7, 7.8], [4, 6.5], [5.5, 7]])
# y là vector kết quả, mỗi phần tử là kết quả học của một sinh viên tính điểm 100
y = np.array([75, 80, 85, 70, 78])

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# Dự đoán kết quả học của sinh viên mới
new_student_features = np.array([[6.5, 8.2]])  # Giả sử sinh viên mới có 6.5 giờ ôn tập và điểm trung bình các kỳ trước là 8.2
predicted_score = model.predict(new_student_features)

print("Dự đoán kết quả học của sinh viên mới là:", predicted_score[0])
