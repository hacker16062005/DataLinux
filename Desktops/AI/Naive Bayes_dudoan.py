import numpy as np
X = np.array([
    [1, 0],
    [0, 0],
    [2, 1],
    [1, 2],
    [0, 1],
])

y = np.array(["C1", "C1", "C2", "C2", "C1"])
def get_class_prob(y: np.array):
  prob = {}
  n = len(y)
  for c in np.unique(y):
    prob[c] = np.count_nonzero(y == c) / n
  return prob

get_class_prob(y)

# output {'C1': 0.6, 'C2': 0.4} 
def get_condition_prob(X: np.ndarray, y: np.array, record: np.array):
  prob = {}
  for c in np.unique(y):
    # Lay tat ca record co class Ci
    class_records = X[y == c]
    n = class_records.shape[0]
    # Tinh xac xuat cua diem du lieu trong lop Ci theo cong thuc nhan
    prob[c] = np.prod(np.count_nonzero(class_records == record, axis=0)/n)
  return prob

input = np.array([1, 1])
get_condition_prob(X, y, input)

# output: {'C1': 0.1111111111111111, 'C2': 0.25}
#https://viblo.asia/p/ml-from-scratch-thuat-toan-phan-loai-naive-bayes-viblo-aNj4vXOqL6r
