import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Khởi tạo một biến X ngẫu nhiên
X = np.random.randn(1000, 1)*6 + 5
# Standardization
X_std = StandardScaler().fit_transform(X.reshape(-1, 1))

def _plot_dist(x, bins=10, xlim=(-1, 1), varname='x'):
  sns.histplot(x, bins = bins, kde = True)
  plt.title('histogram of {}'.format(varname))
  plt.xlim(xlim)
  plt.legend([varname])


# Visualization
fig = plt.figure(figsize=(20, 6))

ax_1 = fig.add_subplot(1, 2, 1)
ax_1 = _plot_dist(X, bins=10, xlim=(-15, 25), varname='original data')


ax_2 = fig.add_subplot(1, 2, 2)
ax_2 = _plot_dist(X_std, bins=10, xlim=(-15, 25), varname='standardized data')
