# naive_bayes_classification.py
import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_wine

# 1. Load dataset (same wine data - compare against previous models)
data = load_wine()
X = data.data
y = data.target

# 2. Split (no scaling needed for Naive Bayes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Cross-validation check
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print("CV Accuracy:", cv_scores.mean())

# 5. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# 6. Evaluate
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=data.target_names))

# 7. Show probability breakdown for a few samples
print("\nSample predictions with probabilities:")
for i in range(3):
    print(f"True: {data.target_names[y_test[i]]}, Predicted: {data.target_names[y_pred[i]]}, "
          f"Probabilities: {np.round(y_prob[i], 3)}")