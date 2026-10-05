# random_forest_classification.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine

# 1. Load dataset (same wine data - compare against single tree)
data = load_wine()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 2. Compare single tree vs random forest (overfitting fix demo)
tree = DecisionTreeClassifier(random_state=42)  # unlimited depth - overfits
tree.fit(X_train, y_train)
print("Single Tree -> Train:", accuracy_score(y_train, tree.predict(X_train)),
      "Test:", accuracy_score(y_test, tree.predict(X_test)))

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train, y_train)
print("Random Forest -> Train:", accuracy_score(y_train, forest.predict(X_train)),
      "Test:", accuracy_score(y_test, forest.predict(X_test)))

# 3. Tune number of trees
for n in [10, 50, 100, 200]:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    print(f"n_estimators={n} -> CV Accuracy: {scores.mean():.3f}")

# 4. Final model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 5. Feature importance (averaged across all trees - more reliable than single tree)
importances = pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False)
print("\nTop 5 important features:\n", importances.head())

plt.figure(figsize=(8, 5))
importances.head(8).plot(kind='barh')
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()