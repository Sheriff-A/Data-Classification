import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets._samples_generator import make_blobs

color_map = {
    0: 'r',
    1: 'g',
    2: 'b',
    3: '#d0d0d0'
}
seed = 49
np.random.seed(seed)
data, c = make_blobs(n_samples=600, centers=10, cluster_std=0.6)
points = pd.DataFrame.from_dict(data={
    'x': data[:, 0],
    'y': data[:, 1],
    'center': [0] * len(data),
})
print('--- Data Points ---')
print(points.head(5))

print('--- Initializing the Centroids ---')
k = 4  # number of centroids
centroid_colors = list(map(lambda x: color_map[x], range(k)))
show_centroids = True
x_upper, x_lower = np.max(points['x']), np.min(points['x'])
y_upper, y_lower = np.max(points['y']), np.min(points['y'])
upper, lower = np.min([x_upper, y_upper]), np.min([x_lower, y_lower])
init_centroids = np.random.uniform(lower, upper, [k, 2])
centroids = pd.DataFrame.from_dict(data={
    'x': init_centroids[:, 0],
    'y': init_centroids[:, 1],
    'points': [[]] * k
})
print(centroids.head(5))

print('--- Displaying the Data ---')
# fig = plt.figure(figsize=(16, 9))
plt.scatter(points['x'], points['y'], alpha=0.5, marker='.')
if show_centroids:
    plt.scatter(centroids['x'], centroids['y'], alpha=1, marker='^',
                color=centroid_colors, edgecolors=centroid_colors)
plt.show()
plt.clf()


# Euclidean Dist
def get_dist(x1, y1, x2, y2):
    return np.sqrt(np.sum([(x1 - x2) ** 2, (y1 - y2) ** 2]))


epochs = 100
for e in range(epochs):
    # print(f'--- Epoch {e} ---')
    centroids['points'] = [[]] * k
    # print(centroids.head(5))
    for pi, point in enumerate(points.values):
        px, py = point[:2]
        closest_centroid, min_dist = -1, 10000000
        for ci, centroid in enumerate(centroids.values):
            dist_from_point = get_dist(px, py, *centroid[:2])
            if min_dist > dist_from_point:
                closest_centroid, min_dist = ci, dist_from_point
        points.loc[pi, ['center']] = closest_centroid  # Update the centroid associated with this point
        # print(points)

        # Update the points associated with the found closest centroid
        centroid_points = centroids.loc[:, ['points']]['points'].tolist()
        centroid_points[closest_centroid] = [*centroid_points[closest_centroid], pi]
        centroids['points'] = centroid_points

    # Update the Centroid Positions
    centroid_info_update = []
    no_change = True
    # print(centroids)
    for _, centroid in enumerate(centroids.values):
        list_of_points = centroid[2]
        if len(list_of_points) == 0:
            centroid_info_update.append(centroid)
            continue
        pos = points.loc[list_of_points, ['x', 'y']].to_numpy()
        # print(pos)
        x_ave = np.mean(pos[:, 0])
        y_ave = np.mean(pos[:, 1])
        if no_change:
            cx, cy = centroid[:2]
            if cx != x_ave or cy != y_ave:
                no_change = False
        centroid_info_update.append([x_ave, y_ave, list_of_points])

    # Update the centroid positions
    centroids.loc[:, :] = centroid_info_update

    # Plot
    colors = list(map(lambda x: color_map[x], points['center']))
    # fig = plt.figure(figsize=(16, 9))
    plt.scatter(points['x'], points['y'], alpha=0.5, marker='.', color=colors, edgecolors=colors)
    if show_centroids:
        plt.scatter(centroids['x'], centroids['y'], alpha=1, marker='^',
                    color=centroid_colors, edgecolors=centroid_colors)
    plt.show()
    plt.clf()

    if no_change:
        break

print(f'--- {k} Centroids Decided After {e} Epochs ---')
