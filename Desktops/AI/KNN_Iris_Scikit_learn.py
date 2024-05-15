import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets
#Load data
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))
X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])
X1 = iris_X[iris_y == 1,:]
print ('\nSamples from class 1:\n', X1[:5,:])
X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])
#Tách training và test sets (Tính khoảng cách)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)
print ("Training size: %d" %len(y_train))
print ("Test size    : %d" %len(y_test))
#Xét k=1
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print ("Print results for 20 test data points:")
print ("Predicted labels: ", y_pred[20:40])
print ("Ground truth    : ", y_test[20:40])
#đánh giá độ chính xác
from sklearn.metrics import accuracy_score
print ("Do chinh xac voi k=1 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
#Xét với k=10
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Do chinh xac voi k=10 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
#Đánh giá khoảng cách Unform
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print( "Do chinh xac voi k=0 10NN (1/khoang cach): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
#Đánh trọng số
def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print( "Do chinh xac voi k=10 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))
