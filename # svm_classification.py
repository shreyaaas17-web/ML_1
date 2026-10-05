# svm_classification.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_moons

# 1. Non-linear dataset (same moons shape as KNN - compare boundaries)
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale (CRITICAL for SVM - it's distance/margin based)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Compare kernels
for kernel in ["linear", "rbf", "poly"]:
    model = SVC(kernel=kernel, random_state=42)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    print(f"kernel={kernel} -> Accuracy: {acc:.3f}")

# 4. Tune C and gamma for rbf kernel (GridSearch - real-world practice)
param_grid = {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, "scale"]}
grid = GridSearchCV(SVC(kernel="rbf", random_state=42), param_grid, cv=5)
grid.fit(X_train_scaled, y_train)
print("\nBest params:", grid.best_params_)

# 5. Final model
model = grid.best_estimator_
y_pred = model.predict(X_test_scaled)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Visualize decision boundary
h = 0.02
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s=100, facecolors='none', edgecolors='black', label='Support Vectors')
plt.legend()
plt.title(f"SVM Decision Boundary (kernel={model.kernel})")
plt.show()