# student_score_regularization.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate realistic dataset with correlated features
np.random.seed(42)
n = 300

study_hours = np.random.uniform(0, 10, n)
sleep_hours = np.random.uniform(4, 9, n)
attendance = np.random.uniform(50, 100, n)
phone_hours = np.random.uniform(0, 6, n)

# correlated/noisy extra features (redundant info -> good for testing Lasso)
revision_hours = study_hours * 0.8 + np.random.normal(0, 1, n)      # correlated with study_hours
tuition_hours = study_hours * 0.5 + np.random.normal(0, 1, n)       # correlated with study_hours

noise = np.random.normal(0, 5, n)

score = (
    study_hours * 5
    + sleep_hours * 2
    + attendance * 0.3
    - phone_hours * 3
    + noise
)
score = np.clip(score, 0, 100)

df = pd.DataFrame({
    "study_hours": study_hours,
    "sleep_hours": sleep_hours,
    "attendance": attendance,
    "phone_hours": phone_hours,
    "revision_hours": revision_hours,   # redundant feature
    "tuition_hours": tuition_hours,     # redundant feature
    "score": score
})

X = df.drop("score", axis=1)
y = df["score"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Scale features (IMPORTANT for Ridge/Lasso - puts all features on same scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.5)": Lasso(alpha=0.5)
}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    print(f"\n{name}")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    print("Coefficients:", dict(zip(X.columns, np.round(model.coef_, 2))))