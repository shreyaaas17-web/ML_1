# customer_churn_classification.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# 1. Generate realistic customer dataset
np.random.seed(42)
n = 500

tenure_months = np.random.uniform(0, 60, n)
monthly_charges = np.random.uniform(20, 120, n)
support_calls = np.random.poisson(2, n)
contract_type = np.random.choice([0, 1, 2], n)  # 0=monthly, 1=yearly, 2=2yr
satisfaction = np.random.uniform(1, 10, n)

# churn probability logic: short tenure, high charges, many support calls,
# monthly contract, low satisfaction -> higher churn risk
logit = (
    -0.05 * tenure_months
    + 0.02 * monthly_charges
    + 0.4 * support_calls
    - 0.6 * contract_type
    - 0.5 * satisfaction
    + 3
)
prob_churn = 1 / (1 + np.exp(-logit))
churned = (np.random.rand(n) < prob_churn).astype(int)

df = pd.DataFrame({
    "tenure_months": tenure_months,
    "monthly_charges": monthly_charges,
    "support_calls": support_calls,
    "contract_type": contract_type,
    "satisfaction": satisfaction,
    "churned": churned
})

X = df.drop("churned", axis=1)
y = df["churned"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# 5. Predict
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# 6. Evaluate
print("Class balance (churn rate):", y.mean())
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nCoefficients:", dict(zip(X.columns, np.round(model.coef_[0], 3))))

# 7. Predict a new customer
new_customer = pd.DataFrame({
    "tenure_months": [3], "monthly_charges": [95],
    "support_calls": [5], "contract_type": [0], "satisfaction": [3]
})
new_scaled = scaler.transform(new_customer)
print("\nNew customer churn probability:", model.predict_proba(new_scaled)[0][1])