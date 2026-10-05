# time_series_forecasting.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error, mean_squared_error
from prophet import Prophet

# Install first: pip install statsmodels prophet

# ============================================================
# 1. GENERATE REALISTIC TIME SERIES (sales with trend + seasonality)
# ============================================================
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", periods=365*3, freq="D")
trend = np.linspace(100, 300, len(dates))
seasonality = 30 * np.sin(2 * np.pi * dates.dayofyear / 365)
weekly_pattern = 10 * np.sin(2 * np.pi * dates.dayofweek / 7)
noise = np.random.normal(0, 10, len(dates))

sales = trend + seasonality + weekly_pattern + noise
df = pd.DataFrame({"date": dates, "sales": sales})

plt.figure(figsize=(14, 4))
plt.plot(df["date"], df["sales"])
plt.title("Daily Sales - Time Series")
plt.show()

# ============================================================
# 2. DECOMPOSE - see trend, seasonality, residuals separately
# ============================================================
ts = df.set_index("date")["sales"]
decomposition = seasonal_decompose(ts, model="additive", period=365)
decomposition.plot()
plt.show()

# ============================================================
# 3. CHECK STATIONARITY (ARIMA requires stationary data)
# ============================================================
result = adfuller(ts)
print(f"ADF Statistic: {result[0]:.3f}")
print(f"p-value: {result[1]:.3f}")
print("Stationary" if result[1] < 0.05 else "Non-stationary (needs differencing)")

# ============================================================
# 4. ARIMA MODEL
# ============================================================
train_size = int(len(ts) * 0.9)
train, test = ts[:train_size], ts[train_size:]

# order=(p,d,q): p=autoregressive lag, d=differencing, q=moving average lag
arima_model = ARIMA(train, order=(5, 1, 2))
arima_fit = arima_model.fit()
print(arima_fit.summary())

arima_forecast = arima_fit.forecast(steps=len(test))

plt.figure(figsize=(14, 5))
plt.plot(train.index, train, label="Train")
plt.plot(test.index, test, label="Actual")
plt.plot(test.index, arima_forecast, label="ARIMA Forecast", color='red')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()

print("\nARIMA MAE:", mean_absolute_error(test, arima_forecast))
print("ARIMA RMSE:", np.sqrt(mean_squared_error(test, arima_forecast)))

# ============================================================
# 5. PROPHET MODEL (easier, handles seasonality automatically)
# ============================================================
prophet_df = df.rename(columns={"date": "ds", "sales": "y"})   # Prophet requires these exact column names
train_p = prophet_df[:train_size]
test_p = prophet_df[train_size:]

prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.fit(train_p)

future = prophet_model.make_future_dataframe(periods=len(test_p))
forecast = prophet_model.predict(future)

prophet_model.plot(forecast)
plt.title("Prophet Forecast")
plt.show()

prophet_model.plot_components(forecast)   # shows trend/weekly/yearly seasonality separately
plt.show()

# Evaluate on test period
prophet_test_pred = forecast.set_index("ds").loc[test_p["ds"], "yhat"]
print("\nProphet MAE:", mean_absolute_error(test_p["y"].values, prophet_test_pred.values))
print("Prophet RMSE:", np.sqrt(mean_squared_error(test_p["y"].values, prophet_test_pred.values)))

# ============================================================
# 6. FUTURE FORECAST (beyond available data)
# ============================================================
future_60 = prophet_model.make_future_dataframe(periods=60)
forecast_60 = prophet_model.predict(future_60)

plt.figure(figsize=(14, 5))
plt.plot(df["date"], df["sales"], label="Historical")
plt.plot(forecast_60["ds"].tail(60), forecast_60["yhat"].tail(60), label="Next 60 days forecast", color='green')
plt.fill_between(forecast_60["ds"].tail(60), forecast_60["yhat_lower"].tail(60), forecast_60["yhat_upper"].tail(60),
                  color='green', alpha=0.2, label="Confidence interval")
plt.legend()
plt.title("60-Day Future Sales Forecast")
plt.show()