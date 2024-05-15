#Tạo hàm khởi tạo K cụm ban đầu

# X là ma trận dữ liệu, n_cluster = K là số cụm 
def kmeans_init_centers(X, n_cluster):
  return X[np.random.choice(X.shape[0], n_cluster, replace=False)]
#Tạo hàm phân chia dữ liệu vào các cụm gần nó nhất

def kmeans_predict_labels(X, centers):
  D = cdist(X, centers) #cdist là hàm tính khoảng cách
  return np.argmin(D, axis = 1) #argmin là trả về giá trị nhỏ nhất
#Tạo hàm cập nhập lại tâm cụm

def kmeans_update_centers(X, labels, n_cluster):
  centers = np.zeros((n_cluster, X.shape[1]))
  for k in range(n_cluster):
    Xk = X[labels == k, :]
    centers[k,:] = np.mean(Xk, axis = 0) #mean là trả về giá trị trung bình cộng
  return centers
#Khảo sát hội tụ (xem thử tâm cụm lúc này bằng tâm cụm trước đó hay chưa)

def kmeans_has_converged(centers, new_centers):
  return (set([tuple(a) for a in centers]) ==
      set([tuple(a) for a in new_centers]))
#Giờ thì là hàm build K-means nhé

# Hàm xây dựng thuật toán K-means
def kmeans(init_centes, init_labels, X, n_cluster):
  centers = init_centes
  labels = init_labels
  times = 0
  while True:
    labels = kmeans_predict_labels(X, centers)
    kmeans_visualize(X, centers, labels, n_cluster, 
                     'Phân nhãn cho dữ liệu lần  ' + str(times + 1))
    new_centers = kmeans_update_centers(X, labels, n_cluster)
    if kmeans_has_converged(centers, new_centers):
      break
    centers = new_centers
    kmeans_visualize(X, centers, labels, n_cluster, 
                     'Update center lần ' + str(times + 1))
    times += 1 # Biến times số lần update center xảy ra
  return (centers, labels, times)
