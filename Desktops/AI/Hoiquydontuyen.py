"""Có năm bước cơ bản khi bạn triển khai hồi quy tuyến tính:
1.	Import các gói và lớp bạn cần.
2.	Cung cấp dữ liệu để làm việc và cuối cùng thực hiện các chuyển đổi thích hợp.
3.	Tạo mô hình hồi quy và phù hợp với dữ liệu hiện có.
4.	Kiểm tra kết quả mô hình để biết mô hình có đạt yêu cầu hay không.
5.	Áp dụng mô hình cho các dự đoán.
"""
import numpy as np
from sklearn.linear_model import LinearRegression
#dữ liệu X bộ dl hồi quy, đầu ra dữ liệu dự đoán
#reshape() trên x bởi vì mảng này bắt buộc phải là hai chiều,
x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([5, 20, 14, 32, 22, 38])
print(x)
print(y)
#tạo mô hình huấn luyện với scikit-learning.
#tạo mô hình hồi quy tuyến tính và điều chỉnh nó bằng cách sử dụng dữ liệu hiện có.
model = LinearRegression()
#tính toán giá trị toi ưu của trọng số b0, b1
model.fit(x, y)
model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print('coefficient of determination:', r_sq)
#coefficient of determination: 0.715875613747954
#lấy b0, b1
print('intercept:', model.intercept_)
print('slope:', model.coef_)
new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print('intercept:', new_model.intercept_)
print('slope:', new_model.coef_)
#dự đoán hồi quy
y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')
#dự đoán phản dứng
y_pred = model.intercept_ + model.coef_ * x
print('predicted response:', y_pred, sep='\n')
