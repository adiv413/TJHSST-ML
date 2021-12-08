from sklearn.datasets import make_blobs
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

x, y = make_blobs(n_samples=100, n_features=2, centers=[(1.0, 5.0), (2.25, 1.0)], cluster_std=[0.4, 0.4])

# plt.scatter(x, y)

# plt.scatter(x[:, 0], x[:, 1], s=50, cmap='viridis')

model = SVC(kernel='linear')
clf = model.fit(x, y)

# code to plot the line adapted from https://stackoverflow.com/questions/51495819/how-to-plot-svm-decision-boundary-in-sklearn-python

def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy

def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out

fig, ax = plt.subplots()
# Set-up grid for plotting.
X0, X1 = x[:, 0], x[:, 1]
xx, yy = make_meshgrid(X0, X1)

plot_contours(ax, clf, xx, yy, alpha=0.8)
ax.scatter(X0, X1, c=y, s=20, edgecolors='k')
ax.set_xticks(())
ax.set_yticks(())
ax.legend()
plt.show()