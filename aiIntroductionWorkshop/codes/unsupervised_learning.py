from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


iris = load_iris()

kmeans = KMeans(n_clusters=3)

kmeans.fit(iris.data)

centers = kmeans.cluster_centers_
labels = kmeans.labels_

plt.scatter(iris.data[:, 0], iris.data[:, 1], c=labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='#050505')
plt.show()