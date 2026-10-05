# digit_recognition_svm.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits

# 1. Load real-world dataset (1797 actual 8x8 handwritten digit images)
data = load_digits()
X = data.data      # each row = 64 pixel values (flattened 8x8 image)
y = data.target    # digit label (0-9)

print("Dataset shape:", X.shape)
print("Classes:", np.unique(y))

# 2. Visualize a few sample digits
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i, ax in enumerate(axes):
    ax.imshow(data.images[i], cmap='gray')
    ax.set_title(f"Label: {data.target[i]}")
    ax.axis('off')
plt.show()

# 3. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Tune hyperparameters
param_grid = {"C": [1, 10, 100], "gamma": [0.001, 0.01, "scale"]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("\nBest params:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

# 6. Final model
model = grid.best_estimator_
y_pred = model.predict(X_test_scaled)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Show some predictions (correct and wrong)
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.ravel()):
    idx = i
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    color = 'green' if y_pred[idx] == y_test[idx] else 'red'
    ax.set_title(f"Pred: {y_pred[idx]}, True: {y_test[idx]}", color=color)
    ax.axis('off')
plt.tight_layout()
plt.show()