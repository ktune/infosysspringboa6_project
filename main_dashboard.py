# =========================================================================
# Dashboard 1: Main Dashboard (Milestone 4 - All Milestones Output)
# =========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# --- Configuration & Setup ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
st.set_page_config(page_title="Air Quality Predictive Dashboard", layout="wide")

# --- 0. Data Functions ---
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the air quality data."""
    try:
        # File name confirmed from user's files
        df = pd.read_excel('air_quality_data_realistic.xlsx', engine='openpyxl')
    except Exception:
        st.error("Error loading 'air_quality_data_realistic.xlsx'. Check file location.")
        return pd.DataFrame() 

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# --- M2: PM10 -> PM2.5 Linear Regression (Metrics Only) ---
def run_milestone_2(df):
    features = ['PM10']
    X = df[features]
    y = df['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return round(mae, 2), round(r2, 2)

# --- M3: Time Series Forecasting (Metrics Only) ---
def run_milestone_3(df):
    df_ts = df[['PM2.5']].copy()
    for i in range(1, 4):
        df_ts[f't-{i}'] = df_ts['PM2.5'].shift(i)
    df_ts.dropna(inplace=True)
    features = ['t-1', 't-2', 't-3']
    X = df_ts[features]
    y = df_ts['PM2.5']
    split_index = int(len(X) * 0.8)
    X_train, X_test, y_train, y_test = X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return round(mae, 2), round(rmse, 2)

# --- M3: Time Series Forecasting (Plot Data) ---
def get_milestone_3_plot_data(df):
    df_ts = df[['PM2.5']].copy()
    for i in range(1, 4):
        df_ts[f't-{i}'] = df_ts['PM2.5'].shift(i)
    df_ts.dropna(inplace=True)
    
    features = ['t-1', 't-2', 't-3']
    X = df_ts[features]
    y = df_ts['PM2.5']
    
    split_index = int(len(X) * 0.8)
    y_test = y.iloc[split_index:]
    
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return y_test, pd.Series(y_pred, index=y_test.index)

# =================================================
# MAIN STREAMLIT APP
# =================================================

st.title("Main Project Dashboard (M1-M3 Integration) 📈")
st.markdown("---")

df = load_data()
if df.empty: st.stop()

# --- Run Models and Get Metrics ---
with st.spinner("Running models and calculating metrics..."):
    mae_m2, r2_m2 = run_milestone_2(df)
    mae_m3, rmse_m3 = run_milestone_3(df)
    y_test_m3, y_pred_m3 = get_milestone_3_plot_data(df)

# =================================================
# SECTION 1: MODEL PERFORMANCE SUMMARY
# =================================================
st.header("1. Predictive Model Summary")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("M2: PM10 vs PM2.5")
    st.metric(label="R-squared (R²)", value=f"{r2_m2}")
    st.caption("Proportion of PM2.5 variance explained by PM10.")

with col2:
    st.subheader("M2: Prediction Error")
    st.metric(label=r"Mean Absolute Error ($\mu g/m^3$)", value=f"{mae_m2}")
    st.caption("Average error of PM2.5 prediction from PM10.")

with col3:
    st.subheader("M3: Time Series Error")
    st.metric(label=r"RMSE (Lag Forecast) ($\mu g/m^3$)", value=f"{rmse_m3}")
    st.caption("Standard deviation of the residuals for the time series forecast.")

st.markdown("---")

# =================================================
# SECTION 2: EDA (Milestone 1)
# =================================================
st.header("2. Exploratory Data Analysis (EDA) - M1")

tab1, tab2, tab3 = st.tabs(["Time Series Trend", "PM2.5 Distribution", "Statistical Summary"])

with tab1:
    st.subheader("Hourly Air Pollutant Concentrations")
    fig, ax = plt.subplots(figsize=(12, 5))
    df_sample = df.iloc[:24*7] # Plot first week for clarity
    ax.plot(df_sample.index, df_sample['PM2.5'], label='PM2.5')
    ax.plot(df_sample.index, df_sample['PM10'], label='PM10')
    ax.set_title('Time Series Trend (First Week)')
    ax.set_ylabel(r'Concentration ($\mu g/m^3$)')
    ax.legend()
    st.pyplot(fig)

with tab2:
    st.subheader("Distribution of PM2.5")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df['PM2.5'], bins=30, kde=True, ax=ax)
    ax.set_title('Frequency Distribution of PM2.5')
    ax.set_xlabel(r'PM2.5 Concentration ($\mu g/m^3$)')
    st.pyplot(fig)

with tab3:
    st.subheader("Key Pollutant Statistics")
    stats_df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3']].describe().T
    st.dataframe(stats_df)

st.markdown("---")

# =================================================
# SECTION 3: M3 FORECAST VISUALIZATION (Milestone 3)
# =================================================
st.header("3. Time Series Forecast Visualization - M3")

st.subheader("Actual PM2.5 vs. Lag-Feature Predicted PM2.5 (Test Set)")
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_test_m3.index, y_test_m3, label="Actual PM2.5", alpha=0.8)
ax.plot(y_pred_m3.index, y_pred_m3, label="Predicted PM2.5 (Lag Model)", color="red", linestyle='--')
ax.set_title("PM2.5 Forecast Comparison")
ax.set_xlabel("Timestamp")
ax.set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
ax.legend()
st.pyplot(fig)

# =================================================