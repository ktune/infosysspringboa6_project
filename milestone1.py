# Import the pandas library for data manipulation
import pandas as pd

# Install the necessary library for Excel files


# Use pd.read_excel instead of pd.read_csv
df = pd.read_excel('air_quality_data_realistic.xlsx')

# Use .info() to get a quick summary of the DataFrame.
# This shows the number of entries, data types, and non-null values.
print("--- Initial DataFrame Info ---")
df.info()

# Use .head() to display the first few rows of the data.
# This helps you see what the data looks like.
print("\n--- First 5 Rows of the Dataset ---")
print(df.head())

# Convert timestamp to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Drop unnecessary columns if applicable
# df.drop(['station_id', 'city'], axis=1, inplace=True)

# Handle missing values (example: forward fill)
df.fillna(method='ffill', inplace=True)

print("\nCleaned DataFrame Info:")
df.info()

import matplotlib.pyplot as plt
import seaborn as sns

# Create a figure and axes for the plot
plt.figure(figsize=(14, 7))

# Plot the 'PM2.5' and 'PM10' columns
plt.plot(df.index, df['PM2.5'], label='PM2.5')
plt.plot(df.index, df['PM10'], label='PM10')

# Add a title, labels, and a legend for clarity
plt.title('Time Series of PM2.5 and PM10')
plt.xlabel('Timestamp')
plt.ylabel('Concentration ($\mu g/m^3$)')
plt.legend()
plt.grid(True)
plt.tight_layout() # Adjusts plot to prevent labels from overlapping
plt.show() # Display the plot

# Create a scatter plot using seaborn for better aesthetics
plt.figure(figsize=(8, 6))
sns.scatterplot(x='PM2.5', y='PM10', data=df)

# Add a title and labels
plt.title('Scatter Plot: PM2.5 vs PM10')
plt.xlabel('PM2.5 Concentration')
plt.ylabel('PM10 Concentration')
plt.grid(True)
plt.tight_layout()
plt.show()

# Use the .describe() method to get a statistical summary of the data
print("\n--- Statistical Summary of Pollutant Data ---")
print(df[['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']].describe())

# Create a histogram for PM2.5 distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['PM2.5'], bins=20, kde=True) # kde=True adds a smooth density curve

plt.title('Distribution of PM2.5 Concentrations')
plt.xlabel('PM2.5 Concentration ($\mu g/m^3$)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Create a histogram for PM2.5 distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['PM2.5'], bins=20, kde=True) # kde=True adds a smooth density curve

plt.title('Distribution of PM2.5 Concentrations')
plt.xlabel('PM2.5 Concentration ($\mu g/m^3$)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()