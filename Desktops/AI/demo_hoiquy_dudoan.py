import numpy as np
from sklearn.linear_model import LinearRegression


# Dữ liệu mẫu
X = np.array([-1, 0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([-2, 1, 4, 7, 10, 13])
''' thử chạy với bộ dữ liệu sau đây không chính xác tại x,y= (0,0) để rút ra suy luận
X = np.array([-1, 0, 1, 2, 3, 4]).reshape(-1, 1)
y = np.array([-2, 0, 4, 7, 10, 13])
'''
# Huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X, y)

# Dự đoán giá trị x = 10
x_predict = np.array([[10]])
y_predict = model.predict(x_predict)

print("Dự đoán giá trị của x = 10 là:", y_predict[0])
print("Hệ số hồi quy (slope):", model.coef_)
print("Hệ số chặn (intercept):", model.intercept_)
# Vẽ đồ thị dữ liệu và đường hồi quy
import matplotlib.pyplot as plt
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.show()