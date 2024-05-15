import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

# Khởi tạo một biến X ngẫu nhiên
X = np.random.randn(1000, 1)+5
# minmax scaler của X
X_minmax = MinMaxScaler().fit_transform(X.reshape(-1, 1))

def _plot_dist(x, bins=10, xlim=(-1, 1), varname='x'):
  sns.histplot(x, bins = bins, kde = True)
  plt.title('histogram of {}'.format(varname))
  plt.xlim(xlim)
  plt.legend([varname])


# Visualization
fig = plt.figure(figsize=(20, 6))

ax_1 = fig.add_subplot(1, 2, 1)
ax_1 = _plot_dist(X, bins=10, xlim=(0, 10), varname='original data')

ax_2 = fig.add_subplot(1, 2, 2)
ax_2 = _plot_dist(X_minmax, bins=10, xlim=(0, 10), varname='minmax scaling data')
