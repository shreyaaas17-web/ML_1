# hierarchical_dbscan_clustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score

# ============================================================
# PART 1: HIERARCHICAL CLUSTERING
# ============================================================
X_blobs, _ = make_blobs(n_samples=150, centers=4, cluster_std=0.8, random_state=42)
X_blobs_scaled = StandardScaler().fit_transform(X_blobs)

# 1. Plot dendrogram to visually decide number of clusters
linked = linkage(X_blobs_scaled, method='ward')

plt.figure(figsize=(10, 5))
dendrogram(linked, truncate_mode='lastp', p=15)
plt.title("Dendrogram - Hierarchical Clustering")
plt.xlabel("Sample clusters")
plt.ylabel("Distance")
plt.axhline(y=8, color='red', linestyle='--', label='Cut here -> 4 clusters')
plt.legend()
plt.show()

# 2. Fit Agglomerative Clustering with chosen number of clusters
hier_model = AgglomerativeClustering(n_clusters=4, linkage='ward')
hier_labels = hier_model.fit_predict(X_blobs_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], c=hier_labels, cmap='viridis')
plt.title("Hierarchical Clustering Result")
plt.show()

print("Hierarchical Silhouette Score:", silhouette_score(X_blobs_scaled, hier_labels))

# ============================================================
# PART 2: DBSCAN - handles non-spherical shapes + outliers
# ============================================================
# Dataset where K-Means/Hierarchical struggle (moon shapes + some noise points)
X_moons, _ = make_moons(n_samples=300, noise=0.08, random_state=42)
X_moons_scaled = StandardScaler().fit_transform(X_moons)

# add some random noise points (outliers)
outliers = np.random.uniform(low=-3, high=3, size=(15, 2))
X_with_noise = np.vstack([X_moons_scaled, outliers])

# 3. Compare KMeans vs DBSCAN on this tricky shape
kmeans_labels = KMeans(n_clusters=2, random_state=42, n_init=10).fit_predict(X_with_noise)
dbscan_labels = DBSCAN(eps=0.2, min_samples=5).fit_predict(X_with_noise)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].scatter(X_with_noise[:, 0], X_with_noise[:, 1], c=kmeans_labels, cmap='coolwarm')
axes[0].set_title("K-Means (fails on curved shapes)")

axes[1].scatter(X_with_noise[:, 0], X_with_noise[:, 1], c=dbscan_labels, cmap='coolwarm')
axes[1].set_title("DBSCAN (handles curves + flags outliers as -1)")
plt.show()

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
print(f"\nDBSCAN found {n_clusters_dbscan} clusters and {n_noise} noise points")

# 4. Tune DBSCAN's eps parameter (most important hyperparameter)
from sklearn.neighbors import NearestNeighbors

neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(X_with_noise)
distances, _ = neighbors_fit.kneighbors(X_with_noise)
distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(distances)
plt.xlabel("Points sorted by distance")
plt.ylabel("5th nearest neighbor distance")
plt.title("K-distance Graph - look for the 'elbow' to pick eps")
plt.show()