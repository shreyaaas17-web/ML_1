# logistic_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Generate dataset: predict pass/fail based on hours studied
np.random.seed(42)
n = 200
hours_studied = np.random.uniform(0, 10, n)

# probability of passing increases with hours studied
prob_pass = 1 / (1 + np.exp(-(hours_studied - 5)))
passed = (np.random.rand(n) < prob_pass).astype(int)   # 0 = fail, 1 = pass

X = hours_studied.reshape(-1, 1)
y = passed

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale (good practice, though not critical for 1 feature)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]   # probability of class 1

# 6. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 7. Visualize sigmoid curve
X_range = np.linspace(0, 10, 200).reshape(-1, 1)
X_range_scaled = scaler.transform(X_range)
y_range_prob = model.predict_proba(X_range_scaled)[:, 1]

plt.scatter(X, y, color='blue', alpha=0.5, label='Actual (0=fail, 1=pass)')
plt.plot(X_range, y_range_prob, color='red', label='Predicted probability')
plt.axhline(0.5, color='gray', linestyle='--', label='Decision boundary')
plt.xlabel("Hours Studied")
plt.ylabel("Probability of Passing")
plt.legend()
plt.title("Logistic Regression")
plt.show()