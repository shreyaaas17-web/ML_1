# multiple_linear_regression.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 1. Generate realistic dataset with multiple features
np.random.seed(42)
n = 200
area = np.random.randint(500, 4000, n)
bedrooms = np.random.randint(1, 6, n)
age = np.random.randint(0, 30, n)
noise = np.random.normal(0, 20000, n)

price = (area * 150) + (bedrooms * 20000) - (age * 1000) + 50000 + noise

df = pd.DataFrame({
    "area": area,
    "bedrooms": bedrooms,
    "age": age,
    "price": price
})

# 2. Features (X) and target (y)
X = df[["area", "bedrooms", "age"]]
y = df["price"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict
y_pred = model.predict(X_test)

# 6. Evaluate
print("Coefficients (area, bedrooms, age):", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# 7. Predict a new house
new_house = pd.DataFrame({"area": [2500], "bedrooms": [3], "age": [5]})
predicted_price = model.predict(new_house)
print(f"Predicted price: {predicted_price[0]:.2f}")