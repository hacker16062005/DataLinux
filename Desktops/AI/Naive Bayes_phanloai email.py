from sklearn.naive_bayes import MultinomialNB
import numpy as np
#dữ liệu traning và nhan
"""Ta có bộ training data gồm E1, E2, E3. Cần phân loại E4
    Nhãn N là thư not spam, còn S là thư spam
    Bảng từ vựng: [w1,w2,w3,w4,w5,w6,w7].
    Số lần xuất hiện của từng từ trong từng email tương ứng
"""
e1 = [1, 2, 1, 0, 1, 0, 0]
e2 = [0, 2, 0, 0, 1, 1, 1]
e3 = [1, 0, 1, 1, 0, 2, 0]
train_data = np.array([e1, e2, e3])
label = np.array(['N', 'N', 'S'])
e4 = np.array([[1, 0, 0, 0, 0, 0, 1]])
#dữ liệu dataset
clf1 = MultinomialNB(alpha=1)
clf1.fit(train_data, label)
print(clf1.predict_proba(e4)) #Xác xuất của e4 trên mỗi lớp dữ liệu
print(str(clf1.predict(e4)[0])) #Dự đoán e4 thuộc lớp
