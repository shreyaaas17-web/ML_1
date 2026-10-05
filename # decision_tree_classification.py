# decision_tree_classification.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine

# 1. Load dataset (same wine data - compare against KNN results)
data = load_wine()
X = data.data
y = data.target

# 2. Split (NOTE: no scaling needed for trees!)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Compare shallow vs deep tree (overfitting demo)
for depth in [2, 4, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"max_depth={depth} -> Train acc: {train_acc:.3f}, Test acc: {test_acc:.3f}")

# 4. Find best depth using cross-validation
depths = range(1, 15)
cv_scores = [cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X_train, y_train, cv=5).mean() for d in depths]
best_depth = depths[np.argmax(cv_scores)]
print(f"\nBest max_depth: {best_depth} (CV accuracy: {max(cv_scores):.3f})")

# 5. Train final model
model = DecisionTreeClassifier(max_depth=best_depth, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 6. Feature importance (unique to trees - shows which features mattered most)
importances = pd.Series(model.feature_importances_, index=data.feature_names).sort_values(ascending=False)
print("\nTop 5 important features:\n", importances.head())

# 7. Visualize the tree
plt.figure(figsize=(16, 8))
plot_tree(model, feature_names=data.feature_names, class_names=data.target_names, filled=True, fontsize=8)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.show()