import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Tạo dữ liệu mẫu
np.random.seed(0)
X = 2 * np.random.rand(100, 1)  # Tạo 100 mẫu từ phân phối đều trong khoảng [0, 2]
y = 3 * X + 1 + np.random.randn(100, 1)  # Tạo các giá trị y tương ứng với mỗi mẫu, thêm nhiễu Gauss

# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# In ra các tham số hồi quy
print("Hệ số hồi quy (slope):", model.coef_)
print("Hệ số chặn (intercept):", model.intercept_)

# Vẽ đồ thị dữ liệu và đường hồi quy
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
