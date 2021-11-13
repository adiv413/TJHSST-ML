from sklearn.datasets import make_circles, make_swiss_roll
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# from https://scikit-learn.org/stable/modules/classes.html#module-sklearn.datasets

x, y = make_swiss_roll(n_samples=1000, noise=0, random_state=10)

labels = KMeans(2, random_state=0).fit_predict(x)
ax = plt.axes(projection='3d')

# plt.scatter(x[:, 0], x[:, 1], x[:, 2], c=labels, s=50, cmap='viridis')
ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=labels, cmap='viridis');

plt.show()