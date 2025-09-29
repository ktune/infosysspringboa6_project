# =========================================================================
# FINAL PROJECT DASHBOARD: UNIFIED OUTPUT (STRUCTURED LIKE THE IMAGE)
# =========================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# Set Matplotlib style for a clean look
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")
st.set_page_config(page_title="Final Air Quality Prediction Dashboard", layout="wide")

# --- 1. Data Loading and Preprocessing ---
@st.cache_data
def load_data():
    """Loads, cleans, and prepares the air quality data."""
    try:
        # Using the correct pd.read_excel and file name based on your milestones
        df = pd.read_excel('air_quality_data_realistic.xlsx', engine='openpyxl')
    except Exception as e:
        st.error(f"Error loading 'air_quality_data_realistic.xlsx'. Check file location and permissions. Details: {e}")
        return pd.DataFrame() 

    # Preprocessing (M1)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    df.fillna(method='ffill', inplace=True)
    return df

# --- 2. Model Training Functions ---

def run_milestone_2(df):
    """Predictive Model: PM10 -> PM2.5 (Linear Regression) - Returns metrics and plot data."""
    features = ['PM10']
    X = df[features]
    y = df['PM2.5']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return round(mae, 2), round(r2, 2), y_test, y_pred, model

def run_milestone_3(df):
    """Time Series Forecasting: PM2.5 (Lagged Features) - Returns all necessary outputs."""
    df_ts = df[['PM2.5']].copy()
    
    # Create Lag Features (t-1, t-2, t-3)
    for i in range(1, 4):
        df_ts[f't-{i}'] = df_ts['PM2.5'].shift(i)
        
    df_ts.dropna(inplace=True)
    features = ['t-1', 't-2', 't-3']
    X = df_ts[features]
    y = df_ts['PM2.5']
    split_index = int(len(X) * 0.8)
    
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    coef_df = pd.DataFrame({
        'Feature': features,
        'Coefficient': model.coef_
    })

    return round(mae, 2), round(rmse, 2), y_test, pd.Series(y_pred, index=y_test.index), coef_df

# =================================================
# MAIN STREAMLIT APP
# =================================================

st.title("🏙️ Air Quality Prediction Dashboard (M1, M2, M3 Outputs)")
st.markdown("A unified view of Exploratory Data Analysis, Regression Modeling, and Time Series Forecasting.")
st.markdown("---")

df = load_data()
if df.empty: st.stop()

# --- Run Models and Get All Outputs ---
with st.spinner("Processing data and training models..."):
    mae_m2, r2_m2, y_test_m2, y_pred_m2, model_m2 = run_milestone_2(df)
    mae_m3, rmse_m3, y_test_m3, y_pred_m3, coef_df_m3 = run_milestone_3(df)

# =================================================
# ROW 1: KEY PERFORMANCE INDICATORS (KPls) - M2 & M3
# =================================================
st.header("1. Model Performance Summary")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.info("Milestone 2: Regression (PM10 $\\rightarrow$ PM2.5)")
    st.metric(label="R-squared ($R^2$)", value=f"{r2_m2}")
    st.caption("Variance explained by PM10.")

with col2:
    st.info("Milestone 2: Regression Error")
    st.metric(label=r"Mean Absolute Error ($\mu g/m^3$)", value=f"{mae_m2}")
    st.caption("Average prediction error.")

with col3:
    st.info("Milestone 3: Forecasting Error")
    st.metric(label=r"Root Mean Square Error ($\mu g/m^3$)", value=f"{rmse_m3}")
    st.caption("Standard deviation of forecast errors.")

with col4:
    st.info("Milestone 3: Forecast Model Key Feature")
    # Display the largest coefficient (t-1)
    st.metric(label="Lag 1 Coefficient (t-1)", value=f"{coef_df_m3.iloc[0]['Coefficient']:.3f}")
    st.caption("Impact of the previous hour's reading.")

st.markdown("---")

# =================================================
# ROW 2: VISUALIZATIONS (M1 & M2)
# =================================================
st.header("2. Regression & Correlation Analysis")

col5, col6 = st.columns(2)

# M2 Scatter Plot (Actual vs Predicted)
with col5:
    st.subheader("2.1. M2: Regression - Actual vs. Predicted PM2.5")
    st.markdown("Visualizing the model's accuracy on the test set. Points near the red line are accurate predictions.")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    
    ax.scatter(y_test_m2, y_pred_m2, alpha=0.6, color='teal', label='Predicted vs. Actual')
    
    # Line of perfect prediction (y=x)
    max_val = max(y_test_m2.max(), y_pred_m2.max())
    min_val = min(y_test_m2.min(), y_pred_m2.min())
    perfect_line = np.linspace(min_val, max_val, 100)
    ax.plot(perfect_line, perfect_line, '--', color='red', label='Perfect Prediction')
    
    ax.set_title(f"PM10 $\\rightarrow$ PM2.5 Regression Performance")
    ax.set_xlabel(r'Actual PM2.5 ($\mu g/m^3$)')
    ax.set_ylabel(r'Predicted PM2.5 ($\mu g/m^3$)')
    ax.legend()
    st.pyplot(fig)

# M1 Scatter Plot (PM2.5 vs PM10 Correlation)
with col6:
    st.subheader("2.2. M1: EDA - PM2.5 vs. PM10 Correlation")
    st.markdown("Confirming the strong linear relationship between the two key pollutants.")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(x=df['PM2.5'], y=df['PM10'], data=df, ax=ax, alpha=0.5)
    
    ax.set_title('M1 Correlation: PM2.5 vs PM10')
    ax.set_xlabel('PM2.5 Concentration')
    ax.set_ylabel('PM10 Concentration')
    st.pyplot(fig)

st.markdown("---")

# =================================================
# ROW 3: TIME SERIES & MODEL INTERPRETATION (M1 & M3)
# =================================================
st.header("3. Time Series Forecasting & Model Interpretation")

col7, col8 = st.columns(2)

# M3 Forecast vs Actual Plot
with col7:
    st.subheader("3.1. M3: Time Series Forecast (Actual vs. Predicted)")
    st.markdown("Comparing the lag-based prediction against the true future PM2.5 values in the test set.")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_test_m3.index, y_test_m3, label="Actual PM2.5", alpha=0.8)
    ax.plot(y_pred_m3.index, y_pred_m3, label="Predicted PM2.5 (Lag Model)", color="red", linestyle='--')
    ax.set_title("M3 Forecast: Test Set Comparison")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(r"PM2.5 Concentration ($\mu g/m^3$)")
    ax.legend()
    st.pyplot(fig)

# M1 Time Series Plot (Trend) & M3 Coefficients
with col8:
    # --- M1 Time Series Trend ---
    st.subheader("3.2. M1: EDA - Pollutant Trend Over Time")
    st.markdown("Showing the hourly fluctuations of PM2.5 and PM10 (Sample of first 10 days).")
    
    fig, ax = plt.subplots(figsize=(7, 5))
    df_sample = df.iloc[:24*10]
    ax.plot(df_sample.index, df_sample['PM2.5'], label='PM2.5')
    ax.plot(df_sample.index, df_sample['PM10'], label='PM10', alpha=0.7)
    ax.set_title('Time Series Trend (First 10 Days)')
    ax.set_ylabel(r'Concentration ($\mu g/m^3$)')
    ax.legend()
    st.pyplot(fig)

    # --- M3 Coefficients Table ---
    st.subheader("3.3. M3: Lag-Feature Coefficients")
    st.dataframe(coef_df_m3.set_index('Feature'))

st.markdown("---")
# =================================================
# ROW 4: DISTRIBUTION & SUMMARY (M1)
# =================================================
st.header("4. Statistical Deep Dive (Milestone 1)")

col9, col10 = st.columns(2)

with col9:
    st.subheader("4.1. PM2.5 Frequency Distribution")
    st.markdown("Understanding the typical range and skewness of PM2.5 concentrations.")
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.histplot(df['PM2.5'], bins=30, kde=True, ax=ax)
    ax.set_title('Frequency Distribution of PM2.5')
    ax.set_xlabel(r'PM2.5 Concentration ($\mu g/m^3$)')
    st.pyplot(fig)

with col10:
    st.subheader("4.2. Statistical Summary")
    st.markdown("Descriptive statistics for all key pollutants in the dataset.")
    stats_df = df[['PM2.5', 'PM10', 'NO2', 'CO', 'O3']].describe().T
    st.dataframe(stats_df)
    
# =================================================