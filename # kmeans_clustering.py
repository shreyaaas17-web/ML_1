# kmeans_clustering.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 1. Generate dataset with natural clusters (no labels used)
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.9, random_state=42)

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find best k using Elbow Method + Silhouette Score
inertias = []
silhouettes = []
k_range = range(2, 10)

for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X_scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, model.labels_))

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(k_range, inertias, marker='o')
axes[0].set_xlabel("k")
axes[0].set_ylabel("Inertia")
axes[0].set_title("Elbow Method")

axes[1].plot(k_range, silhouettes, marker='o', color='green')
axes[1].set_xlabel("k")
axes[1].set_ylabel("Silhouette Score")
axes[1].set_title("Silhouette Method")
plt.show()

best_k = k_range[np.argmax(silhouettes)]
print(f"Best k (by silhouette score): {best_k}")

# 4. Final clustering
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
clusters = model.fit_predict(X_scaled)

# 5. Visualize clusters
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            c='red', marker='X', s=200, label='Centroids')
plt.legend()
plt.title(f"K-Means Clustering (k={best_k})")
plt.show()