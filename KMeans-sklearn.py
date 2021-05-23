import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs
from sklearn.cluster import KMeans

X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)
# print(X)
# print(y)

color_map = {
    0: '#aaaaaa',
    1: '#ff0000',
    2: '#00ff00',
    3: '#0000ff'
}

fig = plt.figure(figsize=(9, 9))
plt.scatter(X[:, 0], X[:, 1], alpha=0.5, marker='.')
plt.show()

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# print(wcss)
plt.clf()
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('# of Clusters')
plt.ylabel('WCSS')
plt.show()

kmeans = KMeans(n_clusters=4, random_state=0)
pred_y = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_
# print(pred_y)
# print(centroids)


colors = list(map(lambda x: color_map[x], pred_y))
centroid_colors = list(color_map.values())
# print(centroid_colors)

plt.clf()
plt.scatter(X[:, 0], X[:, 1], color=colors, edgecolors=colors, alpha=0.5, marker='.')
plt.scatter(centroids[:, 0], centroids[:, 1], color=centroid_colors, edgecolors=centroid_colors, alpha=1, marker='s',
            s=100)
plt.title('KMeans Clustering')
plt.show()
