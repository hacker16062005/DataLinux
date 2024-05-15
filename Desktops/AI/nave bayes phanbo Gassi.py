#Mô hình thuật toán với phân bổ xác suất với các biến dữ liệu liên tục
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Dữ liệu mẫu: tín dụng được chấp hoặc không được chấp
# Dữ liệu mẫu gồm có thu nhập hàng tháng và điểm số tín dụng
# 1 là được chấp, 0 là không được chấp
X = np.array([[2000, 8],
              [3000, 7],
              [4000, 6],
              [1000, 3],
              [1500, 5],
              [5000, 9]])
y = np.array([1, 1, 1, 0, 0, 1])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Naive Bayes và huấn luyện trên tập huấn luyện
model = GaussianNB()
model.fit(X_train, y_train)

# Dự đoán xác suất của các lớp trên tập kiểm tra
probabilities = model.predict_proba(X_test)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# In ra xác suất dự đoán và nhãn dự đoán
for i, (prob, pred) in enumerate(zip(probabilities, y_pred)):
    print(f"Sample {i+1}:")
    print(f"   - Probability of class 0 (Not approved): {prob[0]:.4f}")
    print(f"   - Probability of class 1 (Approved): {prob[1]:.4f}")
    print(f"   - Predicted class: {'Approved' if pred == 1 else 'Not approved'}")
    print()

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
