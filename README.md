### Option B: Stock Price Forecasting


#### Forecast a stock’s next-day (or next-week) price using historical prices.

You will practice:

- Time-aware train/test splitting
- Feature engineering (lags, rolling stats, calendar features)
- Baseline forecasting
- Comparing at least two models
- Important rule: In time series, you must never use future data to predict the past.

#### Notebook Link
- [Stock Price Prediction](./time_series_stock_price_prediction.ipynb)


### Results

**Best Model: Linear Regression** 

| Model | MAE | RMSE | MAPE | R² |
|-------|-----|------|------|-----|
| Baseline | $1.65 | $2.32 | 1.64% | - |
| **Linear Regression** | **$1.96** | **$2.56** | **1.97%** | **0.924** |
| Random Forest | $5.37 | $8.06 | 4.98% | 0.2459 |
| Gradient Boosting | $5.22 | $7.88 | 4.83% | 0.2800 |

- ![Model Comparsion](./results/stock_forecast_results.png)


### Change Stock Ticker
```python
forecaster = StockPriceForecaster(
    ticker='MSFT',  # Try any ticker
    start_date='2021-01-01'
)
```

### Adjust Models
```python
# Linear Regression (works best)
forecaster.train_linear_regression()

# Random Forest (for comparison)
forecaster.train_random_forest(n_estimators=200)

# Gradient Boosting (for comparison)
forecaster.train_gradient_boosting(n_estimators=100)
```
