# polynomial_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate curved (non-linear) dataset
np.random.seed(42)
X = np.sort(np.random.uniform(-3, 3, 100)).reshape(-1, 1)
y = 0.5 * X[:, 0]**3 - X[:, 0]**2 + 2 * X[:, 0] + np.random.normal(0, 3, 100)

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Transform features into polynomial terms (degree 3)
poly = PolynomialFeatures(degree=3)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 4. Train model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 5. Predict
y_pred = model.predict(X_test_poly)

# 6. Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Visualize (smooth curve)
X_range = np.linspace(-3, 3, 200).reshape(-1, 1)
X_range_poly = poly.transform(X_range)
y_range_pred = model.predict(X_range_poly)

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X_range, y_range_pred, color='red', label='Polynomial fit (degree 3)')
plt.legend()
plt.title("Polynomial Regression")
plt.show()