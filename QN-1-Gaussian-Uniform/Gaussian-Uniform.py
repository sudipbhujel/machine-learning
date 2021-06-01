# Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define uniform and gaussian
np.random.seed(10)
uniform_set = np.random.uniform(size=(300, 2))
gaussian_set = np.random.normal(size=(300, 2))

# Scatter plot
plt.scatter(uniform_set[:, 0], uniform_set[:, 1], c='red')
plt.scatter(gaussian_set[:, 0], gaussian_set[:, 1], c='blue')
plt.legend(["Uniform", "Gaussian"])
plt.xlabel("X")
plt.ylabel("Y")
print("Saving Fig...")
plt.savefig("op-scatter-plot.jpg", dpi=500)
plt.show()

# K-Means cluster 1
cluster_size = 2
cluster1 = KMeans(cluster_size)
cluster1.fit(uniform_set)

# Centroids and Labels
centroids = cluster1.cluster_centers_
labels = cluster1.labels_

# Plot Uniform Set and Cluster
colors = ["g.", "r.", "c.", "y."]
for i in range(len(uniform_set)):
    plt.plot(uniform_set[i][0], uniform_set[i][1],
             colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="x", s=150, linewidths=5, zorder=10)
plt.xlabel("X")
plt.ylabel("Y")
print("Saving Fig...")
plt.savefig("op-uniform-scatter-plot.jpg", dpi=500)
plt.show()

# Fitting model for gaussian set (Cluster 2)
cluster2 = KMeans(cluster_size)
cluster2.fit(gaussian_set)

# Centroid and labels
centroids = cluster2.cluster_centers_
labels = cluster2.labels_

# Plot Gaussian Set and Cluster
colors = ["g.", "r.", "c.", "y."]
for i in range(len(gaussian_set)):
    plt.plot(gaussian_set[i][0], gaussian_set[i]
             [1], colors[labels[i]], markersize=10)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker="x", s=150, linewidths=5, zorder=10)
plt.xlabel("X")
plt.ylabel("Y")
print("Saving Fig...")
plt.savefig("op-gaussian-scatter-plot.jpg", dpi=500)
plt.show()

# Mean calculation
uniform_mean = np.mean(uniform_set)
gaussian_mean = np.mean(gaussian_set)

print(f"Uniform Mean {uniform_mean} \n Gaussian Mean {gaussian_mean}")

# Calculate Variance


def variance(data):
    n = len(data)
    mean = sum(data)/n
    deviations = [(x-mean)**2 for x in data]
    vairance = sum(deviations)/n
    return vairance


print(
    f"Variance of Uniform Set {variance(uniform_set)} \n Variance of Gaussian Set {variance(gaussian_set)}")
