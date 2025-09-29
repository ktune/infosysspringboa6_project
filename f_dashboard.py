# =========================================================================
# FINAL INTERACTIVE DASHBOARD: TIME-BASED REPORTS & 24H FORECAST (M3 CORE)
# =========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# --- Configuration & Setup ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("rocket")
st.set_page_config(page_title="Final Air Quality Forecast Dashboard", layout="wide")

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the air quality data."""
    try:
        # Using pd.read_excel as per your Milestone 3 code
        df = pd.read_excel('air_quality_data_realistic.xlsx', engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading 'air_quality_data_realistic.xlsx'. Please ensure the file is present. Details: {e}")
        return pd.DataFrame() 

    # Preprocessing (M3 Steps 2)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# --- 2. Model Training (Milestone 3) ---

@st.cache_data
def run_milestone_3(df):
    """Trains the M3 lag-based Linear Regression model."""
    y = df['PM2.5']
    
    # Create lag features (M3 Step 5)
    df_ml = pd.DataFrame()
    df_ml['t'] = y
    df_ml['t-1'] = y.shift(1)
    df_ml['t-2'] = y.shift(2)
    df_ml['t-3'] = y.shift(3)
    df_ml = df_ml.dropna()

    # Define features and target (M3 Step 6)
    X = df_ml[['t-1', 't-2', 't-3']]
    y_target = df_ml['t']

    # Train-test split (M3 Step 7) - Using shuffle=False for time series
    X_train, X_test, y_train, y_test = train_test_split(X, y_target, test_size=0.2, shuffle=False)

    # Train Linear Regression model (M3 Step 8)
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions (M3 Step 9)
    y_pred = model.predict(X_test)
    
    # Evaluate model performance (M3 Step 10)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    # Prepare coefficients (M3 Step 12)
    coef_df = pd.DataFrame({
        'Feature': ['t-1', 't-2', 't-3'],
        'Coefficient': model.coef_
    })

    return round(mae, 2), round(rmse, 2), y_test, pd.Series(y_pred, index=y_test.index), coef_df, model

# --- 3. Recursive Forecasting (Tomorrow's Prediction) ---

@st.cache_data
def forecast_24h_recursive(df, _model): # <-- FIX: Changed 'model' to '_model' to enable caching
    """Performs a 24-hour recursive forecast using the M3 lag model."""
    
    # Get the last three actual PM2.5 readings
    last_actual_data = df['PM2.5'].tail(3).values.tolist()
    
    # Get the timestamp of the last reading
    last_timestamp = df.index[-1]
    
    forecasts = []
    
    # Recursive prediction loop for 24 hours
    for i in range(1, 25):
        # Input for the model: [t-1, t-2, t-3]
        X_input = np.array([last_actual_data[0], last_actual_data[1], last_actual_data[2]]).reshape(1, -1)
        
        # Predict the next step (PM2.5 at t)
        prediction = _model.predict(X_input)[0]
        
        forecasts.append(prediction)
        
        # Update the inputs for the next step (t becomes t-1, t-1 becomes t-2, etc.)
        last_actual_data = [prediction, last_actual_data[0], last_actual_data[1]]
        
    # Create the timestamp index for the forecast
    forecast_timestamps = pd.to_datetime([last_timestamp + timedelta(hours=i) for i in range(1, 25)])
    
    forecast_df = pd.DataFrame({
        'PM2.5 Forecast': forecasts
    }, index=forecast_timestamps)

    return forecast_df

# =================================================
# MAIN STREAMLIT APP LOGIC
# =================================================

st.title("🗓️ Interactive Air Quality Forecasting Dashboard")
st.markdown("Forecasting PM2.5 using a Lag-based Linear Regression Model.")
st.markdown("---")

df = load_data()
if df.empty: st.stop()

# --- Run Models and Forecast ---
with st.spinner("Training  Model and generating 24-hour forecast..."):
    mae_m3, rmse_m3, y_test_m3, y_pred_m3, coef_df_m3, model_m3 = run_milestone_3(df)
    forecast_df = forecast_24h_recursive(df, model_m3)
    
# --- SIDEBAR: Date Selector (Today's Report) ---
st.sidebar.header("Date Selection")

available_dates = df.index.normalize().unique()
if available_dates.empty:
    st.sidebar.error("No dates available in data.")
    st.stop()
    
# Use the last date in the dataset as the default
default_date_index = available_dates.get_loc(df.index.normalize()[-1])
selected_date = st.sidebar.selectbox(
    "Choose a day for the Report:",
    options=available_dates,
    index=default_date_index,
    format_func=lambda x: x.strftime('%Y-%m-%d')
)

# Filter the DataFrame for the selected day
start_time = pd.to_datetime(selected_date)
end_time = start_time + timedelta(days=1)
daily_df = df.loc[start_time:end_time].copy()
st.sidebar.success(f"Report date: {selected_date.strftime('%Y-%m-%d')}")

# =================================================
# SECTION 1: TODAY'S REPORT (Interactive Data View)
# =================================================
st.header(f"1. Today's Pollutant Report: {selected_date.strftime('%Y-%m-%d')}")
st.markdown("Hourly trends for the most common air pollutants on the selected day.")

col1, col2, col3, col4 = st.columns(4)

# KPI Metrics for the selected day
with col1:
    st.metric(label=r"Avg. PM2.5 ($\mu g/m^3$)", value=f"{daily_df['PM2.5'].mean():.2f}")
with col2:
    st.metric(label=r"Max PM10 ($\mu g/m^3$)", value=f"{daily_df['PM10'].max():.2f}")
with col3:
    st.metric(label=r"Avg. NO2 ($\mu g/m^3$)", value=f"{daily_df['NO2'].mean():.2f}")
with col4:
    st.metric(label=r"Avg. CO (ppm)", value=f"{daily_df['CO'].mean():.2f}")

# Plot of Pollutant Trends
st.subheader("Hourly Trends of Key Pollutants")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(daily_df.index.hour, daily_df['PM2.5'], label='PM2.5', linewidth=2)
ax.plot(daily_df.index.hour, daily_df['PM10'], label='PM10', alpha=0.7)
ax.plot(daily_df.index.hour, daily_df['NO2'], label='NO2', linestyle='--')
ax.set_title(f'Hourly Pollutant Trends on {selected_date.strftime("%B %d, %Y")}')
ax.set_xlabel("Hour of the Day (0-23)")
ax.set_ylabel(r'Concentration ($\mu g/m^3$ or ppm)')
ax.legend(loc='upper right')
st.pyplot(fig)

st.markdown("---")

# =================================================
# SECTION 2: TOMORROW'S PREDICTION (24H Forecast)
# =================================================
st.header("2. Tomorrow's PM2.5 Forecast (24 Hours)")
st.markdown(f"A **recursive prediction** starting from the last available data point: **{df.index[-1].strftime('%Y-%m-%d %H:%M')}**.")

col_summary, col_plot = st.columns([1, 2])

with col_summary:
    st.subheader("Forecast Summary")
    st.metric(label=r"Next 24h Average PM2.5 ($\mu g/m^3$)", value=f"{forecast_df['PM2.5 Forecast'].mean():.2f}")
    st.metric(label=r"Next 24h Max PM2.5 ($\mu g/m^3$)", value=f"{forecast_df['PM2.5 Forecast'].max():.2f}")
    st.dataframe(forecast_df.head(6).style.format("{:.2f}"))

with col_plot:
    st.subheader("24-Hour Predicted Trend")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(forecast_df.index, forecast_df['PM2.5 Forecast'], marker='o', linestyle='-', color='red', label='Predicted PM2.5')
    
    # Plot last 12 actual hours for seamless transition
    last_actual = df['PM2.5'].tail(12)
    ax.plot(last_actual.index, last_actual.values, linestyle='--', color='blue', label='Last 12 Actual Hours')
    
    ax.set_title("24-Hour PM2.5 Recursive Forecast")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.markdown("---")

# =================================================
# SECTION 3: MODEL VALIDATION (M3 Outputs)
# =================================================
st.header("3. Model Validation")
st.markdown("Evaluation of the model's performance on unseen data.")

col_m3_1, col_m3_2, col_m3_3, col_m3_4 = st.columns(4)

with col_m3_1:
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae_m3:.2f}")
with col_m3_2:
    st.metric(label="Root Mean Square Error (RMSE)", value=f"{rmse_m3:.2f}")
with col_m3_3:
    st.metric(label="Lag 1 Coef. (t-1)", value=f"{coef_df_m3.iloc[0]['Coefficient']:.3f}")
with col_m3_4:
    st.metric(label="Model Intercept", value=f"{model_m3.intercept_:.3f}")

vis_col1, vis_col2 = st.columns(2)

with vis_col1:
    st.subheader("M3: Actual vs. Predicted Time Series")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_test_m3.index, y_test_m3, label="Actual PM2.5", alpha=0.8)
    ax.plot(y_pred_m3.index, y_pred_m3, label="Predicted PM2.5", color="red", linestyle='--')
    ax.set_title("M3 Forecast: Test Set Comparison")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
    ax.legend()
    st.pyplot(fig)

with vis_col2:
    st.subheader("M3: Lag-Feature Coefficients")
    st.markdown("The coefficients show the direct influence of past hours on the current hour's prediction.")
    st.dataframe(coef_df_m3.set_index('Feature').style.format("{:.3f}"))
# =================================================