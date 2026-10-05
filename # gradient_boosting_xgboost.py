# gradient_boosting_xgboost.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# XGBoost/LightGBM need separate install: pip install xgboost lightgbm
import xgboost as xgb
import lightgbm as lgb

# 1. Load dataset (same breast cancer data - compare against Random Forest results)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. sklearn's built-in Gradient Boosting (baseline, no extra install needed)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)
print("sklearn GradientBoosting -> Accuracy:", accuracy_score(y_test, gb_pred))

# 3. XGBoost (industry standard, faster, more features)
xgb_model = xgb.XGBClassifier(
    n_estimators=100, learning_rate=0.1, max_depth=3,
    use_label_encoder=False, eval_metric='logloss', random_state=42
)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)
print("XGBoost -> Accuracy:", accuracy_score(y_test, xgb_pred))

# 4. LightGBM (faster on large datasets, leaf-wise growth)
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
lgb_model.fit(X_train, y_train)
lgb_pred = lgb_model.predict(X_test)
print("LightGBM -> Accuracy:", accuracy_score(y_test, lgb_pred))

# 5. Compare against Random Forest (what you already know)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("Random Forest -> Accuracy:", accuracy_score(y_test, rf_pred))

# 6. Tune XGBoost (most important hyperparameters)
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.01, 0.1, 0.3]
}
grid = GridSearchCV(
    xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    param_grid, cv=5, scoring="roc_auc", n_jobs=-1
)
grid.fit(X_train, y_train)
print("\nBest XGBoost params:", grid.best_params_)
print("Best CV ROC-AUC:", grid.best_score_)

# 7. Final evaluation
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("Test ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 8. Feature importance
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 important features:\n", importances.head(10))

plt.figure(figsize=(8, 6))
importances.head(10).plot(kind='barh')
plt.title("XGBoost Feature Importance")
plt.gca().invert_yaxis()
plt.show()

# 9. Early stopping (prevents overfitting - stops training when validation score stops improving)
X_train2, X_val, y_train2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

xgb_early = xgb.XGBClassifier(
    n_estimators=1000, learning_rate=0.1, max_depth=3,
    eval_metric='logloss', random_state=42, early_stopping_rounds=10
)
xgb_early.fit(X_train2, y_train2, eval_set=[(X_val, y_val)], verbose=False)
print(f"\nEarly stopping triggered at iteration: {xgb_early.best_iteration}")
print("Test accuracy with early stopping:", accuracy_score(y_test, xgb_early.predict(X_test)))