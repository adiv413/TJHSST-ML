from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import math
x, y = make_blobs(n_samples=400, n_features=2, centers=[(0, 0), (6, 1.5), (6, -1.5)], cluster_std=[2, 0.5, 0.5], random_state=0)

distances = []
max_distance = 0
for i in range(len(x)):
    inner_distances = []
    for j in range(i, len(x)):
        inner_distances.append(math.sqrt((x[i][0] - x[j][0])**2 + (x[i][1] - x[j][1])**2))
    distances.append(inner_distances)
    if max_distance < max(inner_distances):
        max_distance = max(inner_distances)

total_distances = len(distances) * len(distances[0])
close_pairs = 0

for i in range(len(distances)):
    for j in range(len(distances[i])):
        if distances[i][j] < 0.3 * max_distance:
            close_pairs += 1

k_start = total_distances / close_pairs
p = math.sqrt(k_start)

for k in range(int(k_start - p), int(k_start + p)):
    model = KMeans(k, random_state=0)
    labels = model.fit_predict(x)
    plt.scatter(x[:, 0], x[:, 1], c=labels, s=50, cmap='viridis')
    plt.show()