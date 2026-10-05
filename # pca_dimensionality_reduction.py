# pca_dimensionality_reduction.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_wine

# 1. Load dataset (13 features - hard to visualize directly)
data = load_wine()
X = data.data
y = data.target

# 2. Scale (CRITICAL for PCA - it's variance-based)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Fit PCA with all components first - see how much variance each explains
pca_full = PCA()
pca_full.fit(X_scaled)

explained_var = pca_full.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_var)+1), cumulative_var, marker='o')
plt.axhline(0.95, color='red', linestyle='--', label='95% variance')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA - Choosing Number of Components")
plt.legend()
plt.show()

n_components_95 = np.argmax(cumulative_var >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95} (out of {X.shape[1]} original features)")

# 4. Reduce to 2D for visualization
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
for label in np.unique(y):
    plt.scatter(X_pca_2d[y == label, 0], X_pca_2d[y == label, 1], label=data.target_names[label], alpha=0.7)
plt.xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}% variance)")
plt.ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}% variance)")
plt.legend()
plt.title("Wine Dataset - Reduced to 2D via PCA")
plt.show()

# 5. Compare model accuracy: original features vs PCA-reduced features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

model_full = LogisticRegression(max_iter=1000)
model_full.fit(X_train, y_train)
acc_full = accuracy_score(y_test, model_full.predict(X_test))

pca_reduced = PCA(n_components=n_components_95)
X_train_pca = pca_reduced.fit_transform(X_train)
X_test_pca = pca_reduced.transform(X_test)

model_pca = LogisticRegression(max_iter=1000)
model_pca.fit(X_train_pca, y_train)
acc_pca = accuracy_score(y_test, model_pca.predict(X_test_pca))

print(f"\nAccuracy with all {X.shape[1]} features: {acc_full:.3f}")
print(f"Accuracy with {n_components_95} PCA components: {acc_pca:.3f}")