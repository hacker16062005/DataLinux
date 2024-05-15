#Mô hình này được áp dụng cho các loại dữ liệu mà mỗi thành phần là một giá trị binary - bẳng 0 hoặc 1
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, classification_report

# Dữ liệu mẫu: tín dụng được chấp hoặc không được chấp
# Dữ liệu mẫu gồm có thu nhập hàng tháng và điểm số tín dụng
# 1 là được chấp, 0 là không được chấp
X = [[2000, 8],
     [3000, 7],
     [4000, 6],
     [1000, 3],
     [1500, 5],
     [5000, 9]]
y = [1, 1, 1, 0, 0, 1]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình Naive Bayes Bernoulli và huấn luyện trên tập huấn luyện
model = BernoulliNB()
model.fit(X_train, y_train)

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
