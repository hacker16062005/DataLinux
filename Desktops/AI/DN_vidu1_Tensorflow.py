import tensorflow as tf
import numpy as np
from tensorflow import keras
#xác định Mô h́nh mạng nơ-ron dưới dạng tập hợp các lớp tuần tự được gọi là keras
#mô h́inh này units=1 có 1 no-ron, 1 tham so đầu vào x dự đoán y theo x input_shape
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
#Hàm loss đo lường các câu trả lời đã dự đoán so với các câu trả lời đúng
#đã biết và đo lường mức độ hiệu quả hoặc kém của câu trả lời đó
#Hàm optimizer để thực hiện một việc đoán khác.Dựa trên kết quả của hàm
#bị sai khác từ dữ liệu, hàm này cố gắng giảm thiểu sự sai lệch dự đoán. 
model.compile(optimizer='sgd', loss='mean_squared_error')
#Sử dụng mean_squared_error để giảm sự sai lệch xuống(sgd) đối với trình tối ưu hóa.
#Y=3X+1
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
#Tạo ra 500 vong lặp dự đoán Y tương ứng X
model.fit(xs, ys, epochs=1000)
#Sử dụng mô h́nh mạng nơ-ron để dự đoán giá trị Y khi X=10
print(model.predict([10.0]))
