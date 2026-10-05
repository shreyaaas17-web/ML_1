# house_price_regression.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate realistic noisy dataset (100 samples)
np.random.seed(42)
area = np.random.randint(500, 4000, 100)                     # sq ft
noise = np.random.normal(0, 15000, 100)                      # random noise
price = area * 150 + 50000 + noise                           # price with some randomness

X = area.reshape(-1, 1)
y = price

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate
print("Slope (price per sq ft):", model.coef_[0])
print("Intercept (base price):", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 6. Predict a new value
new_area = np.array([[2500]])
predicted_price = model.predict(new_area)
print(f"Predicted price for 2500 sq ft: {predicted_price[0]:.2f}")

# 7. Visualize
plt.scatter(X_test, y_test, color='blue', label='Actual price')
plt.plot(X_test, y_pred, color='red', label='Predicted line')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.legend()
plt.title("House Price Prediction (Simple Linear Regression)")
plt.show()