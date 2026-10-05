# breast_cancer_random_forest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.datasets import load_breast_cancer

# 1. Load real-world medical dataset (built into sklearn)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target   # 0 = malignant, 1 = benign

print("Dataset shape:", X.shape)
print("Class balance:", np.bincount(y), "(malignant, benign)")

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Hyperparameter tuning with GridSearchCV (real-world practice)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5]
}
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring="roc_auc")
grid.fit(X_train, y_train)

print("\nBest params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# 4. Final model with best params
model = grid.best_estimator_
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Evaluate (in medicine, RECALL for malignant class matters most - missing cancer is worse than false alarm)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Feature importance - which measurements matter most for diagnosis
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 diagnostic features:\n", importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).plot(kind='barh')
plt.title("Top Features for Cancer Diagnosis")
plt.gca().invert_yaxis()
plt.show()

# 7. ROC curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve - Breast Cancer Diagnosis")
plt.show()