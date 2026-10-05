# customer_segmentation_kmeans.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Generate realistic customer dataset
np.random.seed(42)
n = 400

# Simulate 4 natural customer types with some noise
segment_sizes = [100, 100, 100, 100]
data = []

# High income, high spending (premium customers)
data.append(np.column_stack([
    np.random.normal(85, 10, segment_sizes[0]),   # annual income (k$)
    np.random.normal(80, 10, segment_sizes[0])    # spending score (0-100)
]))
# High income, low spending (savers)
data.append(np.column_stack([
    np.random.normal(85, 10, segment_sizes[1]),
    np.random.normal(20, 10, segment_sizes[1])
]))
# Low income, high spending (impulsive)
data.append(np.column_stack([
    np.random.normal(25, 10, segment_sizes[2]),
    np.random.normal(75, 10, segment_sizes[2])
]))
# Low income, low spending (budget-conscious)
data.append(np.column_stack([
    np.random.normal(25, 10, segment_sizes[3]),
    np.random.normal(25, 10, segment_sizes[3])
]))

X = np.vstack(data)
df = pd.DataFrame(X, columns=["annual_income_k", "spending_score"])

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# 3. Find best k
k_range = range(2, 8)
silhouettes = []
for k in k_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    silhouettes.append(silhouette_score(X_scaled, labels))

best_k = k_range[np.argmax(silhouettes)]
print(f"Best k: {best_k}")

plt.plot(k_range, silhouettes, marker='o')
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.title("Choosing k for Customer Segments")
plt.show()

# 4. Final clustering
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["segment"] = model.fit_predict(X_scaled)

# 5. Visualize segments
plt.figure(figsize=(8, 6))
for seg in sorted(df["segment"].unique()):
    subset = df[df["segment"] == seg]
    plt.scatter(subset["annual_income_k"], subset["spending_score"], label=f"Segment {seg}", alpha=0.6)
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score")
plt.legend()
plt.title("Customer Segments")
plt.show()

# 6. Business interpretation - profile each segment
print("\nSegment profiles (avg income & spending):")
print(df.groupby("segment")[["annual_income_k", "spending_score"]].mean())

# 7. Assign business labels based on profile
profile = df.groupby("segment")[["annual_income_k", "spending_score"]].mean()
labels_map = {}
for seg, row in profile.iterrows():
    income_level = "High" if row["annual_income_k"] > 50 else "Low"
    spend_level = "High" if row["spending_score"] > 50 else "Low"
    labels_map[seg] = f"{income_level} Income / {spend_level} Spending"

df["segment_label"] = df["segment"].map(labels_map)
print("\nSegment business labels:\n", df.groupby("segment_label").size())