# wine_classification_knn.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine

# 1. Load real-world dataset (built into sklearn - actual wine chemical data)
data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 3 wine classes based on cultivar

print("Features:", list(X.columns))
print("Classes:", data.target_names)
print("Dataset shape:", X.shape)

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale (CRITICAL here - features like 'proline' are on totally different scale than 'ash')
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Find best k using cross-validation
k_values = range(1, 21)
cv_scores = []
for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_scores.append(scores.mean())

best_k = k_values[np.argmax(cv_scores)]
print(f"\nBest k found: {best_k} (CV accuracy: {max(cv_scores):.3f})")

plt.plot(k_values, cv_scores, marker='o')
plt.xlabel("k")
plt.ylabel("Cross-validated Accuracy")
plt.title("Choosing best k")
plt.show()

# 5. Train final model with best k
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# 6. Evaluate
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Predict a new wine sample
new_wine = X_test.iloc[[0]]
predicted_class = model.predict(scaler.transform(new_wine))
print(f"\nSample prediction: {data.target_names[predicted_class[0]]} (actual: {data.target_names[y_test[0]]})")