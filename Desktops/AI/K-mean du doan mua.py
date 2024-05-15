import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Đọc dữ liệu từ tệp CSV (ví dụ)
data = pd.read_csv('rain_data.csv')

# Chọn các biến đầu vào (nhiệt độ, độ ẩm, vv.) và biến mục tiêu (lượng mưa)
X = data[['Temperature', 'Humidity', 'Pressure']]
y = data['Rainfall']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Xây dựng mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán lượng mưa trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
