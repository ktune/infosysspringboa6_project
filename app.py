import io
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, send_file

# Use the 'Agg' backend to avoid issues with non-GUI environments like servers.
matplotlib.use('Agg')

app = Flask(__name__, static_folder='static', template_folder='templates')

# Function to load and preprocess the data
def load_and_preprocess_data():
    """Loads, cleans, and preprocesses the air quality data from the CSV file."""
    try:
        # Check if the data file exists in the same directory
        file_path = 'air_quality_data_realistic.xlsx'
        if not os.path.exists(file_path):
            print(f"Error: File not found at '{file_path}'. Please place the CSV file in the same folder as app.py.")
            return pd.DataFrame() # Return an empty DataFrame to prevent app crash

        # Load the data using pandas
        df = pd.read_csv(file_path)

        # Convert the 'timestamp' column to a proper datetime format
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Handle missing values by forward-filling them
        df.fillna(method='ffill', inplace=True)

        # Clean column names for easier access (e.g., 'PM2.5' -> 'PM25')
        df.columns = [col.replace('.', '').replace(' ', '') for col in df.columns]

        return df

    except Exception as e:
        print(f"An error occurred during data loading: {e}")
        return pd.DataFrame()

# Load the data when the application starts
df = load_and_preprocess_data()

# Function to compute statistical summary from the loaded data
def compute_stats():
    """Calculates statistical summary for the dashboard."""
    if df.empty:
        return {}
    
    # Calculate key statistics for the PM25 column
    stats = {
        'pm25_mean': round(df['PM25'].mean(), 1),
        'pm25_median': round(df['PM25'].median(), 1),
        'pm25_max': round(df['PM25'].max(), 1),
        'pm25_min': round(df['PM25'].min(), 1),
        'pm25_std': round(df['PM25'].std(), 1),
        'data_points': len(df)
    }
    return stats

@app.route('/')
def index():
    """Renders the main dashboard page with statistical data."""
    stats = compute_stats()
    return render_template('index.html', stats=stats)

@app.route('/plot/<name>')
def plot_png(name):
    """Generates and serves different plots as images based on the URL."""
    if df.empty:
        # Handle the case where the data failed to load
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Data not available", ha='center', va='center', fontsize=14)
        ax.axis('off')
        img = io.BytesIO()
        fig.savefig(img, format='png')
        img.seek(0)
        plt.close(fig)
        return send_file(img, mimetype='image/png')

    fig = None
    if name == 'time_series':
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#f7f9f3')
        ax.set_facecolor('white')
        ax.plot(df.index, df['PM25'], label='PM2.5', color='#2e4d41')
        ax.plot(df.index, df['PM10'], label='PM10', color='#74b9a9')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel(r'Concentration ($\mu g/m^3$)')
        ax.set_title('Time Series of PM2.5 and PM10')
        ax.legend()
        plt.xticks(rotation=45, ha='right')
        ax.set_ylim([0, 300])
        fig.tight_layout()

    elif name == 'correlations':
        fig, ax = plt.subplots(figsize=(7, 7), facecolor='#f7f9f3')
        ax.set_facecolor('white')
        sns.regplot(x='PM25', y='PM10', data=df.sample(n=min(500, len(df))), ax=ax, scatter_kws={'alpha':0.6, 'color':'#44a390'}, line_kws={'color':'#2e4d41'})
        ax.set_xlabel('PM2.5 Concentration')
        ax.set_ylabel('PM10 Concentration')
        ax.set_title('Pollutant Correlations: PM2.5 vs PM10')
        ax.set_xlim([0, 250])
        ax.set_ylim([0, 350])
        fig.tight_layout()

    elif name == 'distribution':
        fig, ax = plt.subplots(figsize=(10, 4), facecolor='#f7f9f3')
        ax.set_facecolor('white')
        sns.histplot(df['PM25'], bins=20, kde=False, color='#2e4d41', ax=ax)
        ax.set_xlabel(r'PM2.5 Concentration ($\mu g/m^3$)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of PM2.5 Concentrations')
        ax.set_xlim([0, 250])
        fig.tight_layout()
    else:
        # Return a simple error image for an invalid plot request
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Plot not found", ha='center', va='center')
        ax.axis('off')
    
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    plt.close(fig)
    return send_file(img, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
    