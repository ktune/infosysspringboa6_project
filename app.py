from flask import Flask, render_template, send_file
import io
import matplotlib
matplotlib.use('Agg')  # use non-GUI backend for servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

app = Flask(__name__)

def load_mock_data():
    # Replace this with CSV load later if you want real data.
    # We generate 8760 hours of mock hourly readings (one year).
    rng = np.random.default_rng(42)
    hours = np.arange(8760)
    # base seasonal + trend + noise
    pm25 = np.clip(rng.normal(120 + 20*np.sin(2*np.pi*hours/24), 60) + hours/200, 0, None)
    pm10 = np.clip(pm25 * 1.3 + rng.normal(0, 20, size=pm25.shape), 0, None)
    no2  = np.clip(rng.normal(22, 9, size=pm25.shape), 0, None)
    df = pd.DataFrame({
        'hour': hours,
        'pm25': pm25,
        'pm10': pm10,
        'no2': no2
    })
    return df

def compute_stats(df):
    stats = {
        'pm25_mean': round(df['pm25'].mean(), 2),
        'pm25_std' : round(df['pm25'].std(), 2),
        'pm25_min' : round(df['pm25'].min(), 2),
        'pm25_max' : round(df['pm25'].max(), 2),

        'pm10_mean': round(df['pm10'].mean(), 2),
        'pm10_std' : round(df['pm10'].std(), 2),
        'pm10_min' : round(df['pm10'].min(), 2),
        'pm10_max' : round(df['pm10'].max(), 2),

        'no2_mean' : round(df['no2'].mean(), 2),
        'no2_std'  : round(df['no2'].std(), 2),
        'no2_min'  : round(df['no2'].min(), 2),
        'no2_max'  : round(df['no2'].max(), 2)
    }
    return stats

# Global data loaded once on app start (for prototyping)
DF = load_mock_data()
STATS = compute_stats(DF)

@app.route('/')
def index():
    # The template will include <img> tags pointing to /plot/time_series.png etc.
    return render_template('index.html', stats=STATS)

@app.route('/plot/<name>.png')
def plot_png(name):
    """
    Serve PNG images for:
     - time_series
     - scatter
     - histogram
    """
    buf = io.BytesIO()

    if name == 'time_series':
        # show first 500 hours to keep plot small and readable
        sample = DF.iloc[:500]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(sample['hour'], sample['pm25'], label='PM2.5')
        ax.plot(sample['hour'], sample['pm10'], label='PM10')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Concentration')
        ax.set_title('Time Series of PM2.5 and PM10 (sample)')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()

    elif name == 'scatter':
        sample = DF.sample(n=1000, random_state=1)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(sample['pm25'], sample['pm10'], s=10, alpha=0.6)
        ax.set_xlabel('PM2.5')
        ax.set_ylabel('PM10')
        ax.set_title('Scatter: PM2.5 vs PM10')
        ax.grid(True)
        fig.tight_layout()

    elif name == 'histogram':
        # histogram of PM2.5 using full data
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(DF['pm25'], bins=20, edgecolor='black', alpha=0.7)
        ax.set_xlabel('PM2.5 Concentration')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of PM2.5 Concentrations')
        ax.grid(axis='y')
        fig.tight_layout()

    else:
        # return 404-like image
        fig, ax = plt.subplots(figsize=(6,2))
        ax.text(0.5, 0.5, f'Unknown plot: {name}', ha='center', va='center')
        ax.axis('off')
        fig.tight_layout()

    # save to buffer and return
    fig.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == '__main__':
    # For local development. In VS Code terminal run: python app.py
    app.run(debug=True, port=5000)
