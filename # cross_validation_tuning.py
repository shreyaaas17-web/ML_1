# cross_validation_tuning.py
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split, cross_val_score, KFold, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_wine
from scipy.stats import randint, uniform

# 1. Load data
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Basic cross-validation (5-fold, stratified = keeps class balance in each fold)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(random_state=42)
scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
print("CV scores per fold:", np.round(scores, 3))
print(f"Mean CV accuracy: {scores.mean():.3f} (+/- {scores.std():.3f})")

# 3. Pipeline: chain scaling + model (prevents data leakage - scaler only fits on train folds)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC())
])

# 4. GridSearchCV - exhaustive search over specified parameter grid
param_grid = {
    "svm__C": [0.1, 1, 10, 100],
    "svm__gamma": [0.001, 0.01, 0.1, 1],
    "svm__kernel": ["rbf", "linear"]
}
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n--- GridSearchCV ---")
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# 5. RandomizedSearchCV - faster, samples random combinations (better for large search spaces)
param_dist = {
    "svm__C": uniform(0.1, 100),
    "svm__gamma": uniform(0.001, 1),
    "svm__kernel": ["rbf", "linear"]
}
random_search = RandomizedSearchCV(pipeline, param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

print("\n--- RandomizedSearchCV ---")
print("Best params:", random_search.best_params_)
print("Best CV score:", random_search.best_score_)

# 6. Final evaluation on untouched test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nFinal Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))