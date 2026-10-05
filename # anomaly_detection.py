# anomaly_detection.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# 1. GENERATE REALISTIC FRAUD-LIKE DATASET (mostly normal + few anomalies)
# ============================================================
np.random.seed(42)
n_normal = 950
n_anomalies = 50

# Normal transactions (clustered)
normal_data = np.random.normal(loc=[50, 5], scale=[15, 2], size=(n_normal, 2))
# Anomalous transactions (unusual amount/frequency combos)
anomaly_data = np.random.uniform(low=[150, 15], high=[300, 30], size=(n_anomalies, 2))

X = np.vstack([normal_data, anomaly_data])
y_true = np.array([0]*n_normal + [1]*n_anomalies)   # 0=normal, 1=anomaly (only for evaluation)

df = pd.DataFrame(X, columns=["transaction_amount", "transactions_per_day"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', alpha=0.6)
plt.xlabel("Transaction Amount")
plt.ylabel("Transactions per Day")
plt.title("Ground Truth: Normal (blue) vs Anomaly (red)")
plt.show()

# ============================================================
# 2. ISOLATION FOREST - isolates anomalies faster than normal points
# ============================================================
iso_forest = IsolationForest(contamination=0.05, random_state=42)   # contamination = expected % anomalies
iso_pred = iso_forest.fit_predict(X_scaled)   # returns 1=normal, -1=anomaly
iso_pred_binary = np.where(iso_pred == -1, 1, 0)   # convert to 0/1 to match y_true

print("--- Isolation Forest ---")
print(confusion_matrix(y_true, iso_pred_binary))
print(classification_report(y_true, iso_pred_binary))

# ============================================================
# 3. LOCAL OUTLIER FACTOR - density-based (compares local density to neighbors)
# ============================================================
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
lof_pred = lof.fit_predict(X_scaled)
lof_pred_binary = np.where(lof_pred == -1, 1, 0)

print("\n--- Local Outlier Factor ---")
print(confusion_matrix(y_true, lof_pred_binary))
print(classification_report(y_true, lof_pred_binary))

# ============================================================
# 4. ONE-CLASS SVM - learns boundary around normal data
# ============================================================
oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto')
svm_pred = oc_svm.fit_predict(X_scaled)
svm_pred_binary = np.where(svm_pred == -1, 1, 0)

print("\n--- One-Class SVM ---")
print(confusion_matrix(y_true, svm_pred_binary))
print(classification_report(y_true, svm_pred_binary))

# ============================================================
# 5. VISUALIZE ALL THREE METHODS SIDE BY SIDE
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, pred, title in zip(
    axes,
    [iso_pred_binary, lof_pred_binary, svm_pred_binary],
    ["Isolation Forest", "Local Outlier Factor", "One-Class SVM"]
):
    ax.scatter(X[:, 0], X[:, 1], c=pred, cmap='coolwarm', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Transaction Amount")
    ax.set_ylabel("Transactions per Day")

plt.tight_layout()
plt.show()

# ============================================================
# 6. STATISTICAL METHOD (simple baseline - Z-score)
# ============================================================
z_scores = np.abs((df - df.mean()) / df.std())
z_anomalies = (z_scores > 3).any(axis=1).astype(int)   # flag if any feature is 3+ std devs away

print("\n--- Z-Score Method (baseline) ---")
print(confusion_matrix(y_true, z_anomalies))
print(classification_report(y_true, z_anomalies))

# ============================================================
# 7. SCORE NEW TRANSACTIONS (real-world usage)
# ============================================================
new_transactions = pd.DataFrame({
    "transaction_amount": [55, 250],
    "transactions_per_day": [6, 22]
})
new_scaled = scaler.transform(new_transactions)
new_pred = iso_forest.predict(new_scaled)
anomaly_scores = iso_forest.score_samples(new_scaled)   # lower score = more anomalous

for i in range(len(new_transactions)):
    status = "ANOMALY" if new_pred[i] == -1 else "Normal"
    print(f"Transaction {i+1}: {status} (anomaly score: {anomaly_scores[i]:.3f})")