# ridge_lasso_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate curved dataset (same as before)
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
y = 0.5 * X[:, 0]**3 - X[:, 0]**2 + 2 * X[:, 0] + np.random.normal(0, 3, 100)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Create HIGH degree polynomial features (prone to overfitting)
poly = PolynomialFeatures(degree=15)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 3. Train three models: plain, Ridge, Lasso
models = {
    "Linear (no regularization)": LinearRegression(),
    "Ridge (L2)": Ridge(alpha=1.0),
    "Lasso (L1)": Lasso(alpha=0.1)
}

X_range = np.linspace(-3, 3, 200).reshape(-1, 1)
X_range_poly = poly.transform(X_range)

plt.scatter(X, y, color='gray', alpha=0.5, label='Actual data')

for name, model in models.items():
    model.fit(X_train_poly, y_train)
    y_pred = model.predict(X_test_poly)