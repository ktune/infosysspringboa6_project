# ===============================
# Milestone 3: Machine Learning Forecasting (Linear Regression)
# ===============================

# --- Step 0: Imports ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --- Step 1: Load the dataset ---
df = pd.read_excel('air_quality_data_realistic.xlsx')

print("--- Initial DataFrame Info ---")
df.info()

print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

# --- Step 2: Convert timestamp and set index ---
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Fill missing values using forward fill
df.fillna(method='ffill', inplace=True)

print("\nCleaned DataFrame Info:")
df.info()

# --- Step 3: Visualize PM2.5 over time ---
plt.figure(figsize=(14, 6))
plt.plot(df.index, df['PM2.5'], label='PM2.5')
plt.title('Time Series of PM2.5')
plt.xlabel('Timestamp')
plt.ylabel('PM2.5 Concentration ($\mu g/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 4: Define target series ---
y = df['PM2.5']

# --- Step 5: Create lag features for supervised learning ---
df_ml = pd.DataFrame()
df_ml['t'] = y
df_ml['t-1'] = y.shift(1)
df_ml['t-2'] = y.shift(2)
df_ml['t-3'] = y.shift(3)
df_ml = df_ml.dropna()

print("\n--- Lag Features for Supervised Learning ---")
print(df_ml.head())

# --- Step 6: Define features and target ---
X = df_ml[['t-1', 't-2', 't-3']]
y_target = df_ml['t']

# --- Step 7: Train-test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, shuffle=False)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# --- Step 8: Train Linear Regression model ---
model = LinearRegression()
model.fit(X_train, y_train)

# --- Step 9: Make predictions ---
y_pred = model.predict(X_test)

# --- Step 10: Evaluate model performance ---
mae_lr = mean_absolute_error(y_test, y_pred)
rmse_lr = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nLinear Regression Model → MAE: {round(mae_lr,2)}, RMSE: {round(rmse_lr,2)}")

# --- Step 11: Visualize actual vs predicted ---
plt.figure(figsize=(14,6))
plt.plot(y_test.index, y_test, label="Actual PM2.5")
plt.plot(y_test.index, y_pred, label="Predicted PM2.5 (Linear Regression)", color="red")
plt.title("Linear Regression Forecast vs Actual PM2.5")
plt.xlabel("Timestamp")
plt.ylabel("PM2.5 Concentration ($\mu g/m^3$)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Step 12: Show model coefficients ---
print("\nModel Coefficients:")
for feature, coef in zip(['t-1','t-2','t-3'], model.coef_):
    print(f"{feature}: {round(coef, 3)}")
print(f"Model Intercept: {round(model.intercept_, 3)}")

# --- Step 13: Optional: Compare with baseline models ---
# Placeholder baseline values, replace with your own calculations
mae_persist = 10.0
rmse_persist = 12.0
mae_ma = 9.5
rmse_ma = 11.5

results = pd.DataFrame({
    "Model": ["Persistence", "Moving Average (3h)", "Linear Regression"],
    "MAE": [mae_persist, mae_ma, mae_lr],
    "RMSE": [rmse_persist, rmse_ma, rmse_lr]
})

print("\n📊 Model Performance Comparison:")
print(results.round(3))
