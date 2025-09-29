import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# A little style goes a long way.
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

print("--- Milestone 2: The Predictive Engine 🧠 ---")
print("Loading and preparing the air quality data for modeling...")

# Load the realistic air quality data.
try:
    df = pd.read_excel('air_quality_data_realistic.xlsx', engine='openpyxl')
except FileNotFoundError:
    print("Error: The file 'air_quality_data_realistic.xlsx' was not found. Make sure it's in the same directory.")
    exit()

# --- 1. Data Preprocessing ---
# Handle any remaining missing values before we start.
df.fillna(method='ffill', inplace=True)

# Define our variables.
# We're using PM10 to predict PM2.5, as our EDA showed a strong correlation.
features = ['PM10']
target = 'PM2.5'

X = df[features]
y = df[target]

# Split the data into training and testing sets.
# We hold back a portion of the data to test our model's real-world performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nData successfully partitioned:")
print(f"- Training data: {len(X_train)} samples")
print(f"- Testing data:  {len(X_test)} samples")

# --- 2. Training the Model ---
print("\nInitiating model training... Please wait.")

# Initialize the model. We'll start with a simple but effective Linear Regression.
model = LinearRegression()

# Train the model. It's learning the relationship between PM10 and PM2.5.
model.fit(X_train, y_train)

print("Model training complete! 🎉")

# --- 3. Evaluating Performance ---
print("\nAssessing the model's predictive power...")

# Make predictions on the unseen test data.
y_pred = model.predict(X_test)

# Calculate key performance metrics.
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Summary:")
print(f"- Mean Absolute Error (MAE): {mae:.2f}")
print(f"- R-squared (R²):           {r2:.2f}")

# A low MAE means our predictions are, on average, very close to the actual values.
# An R² value close to 1 means our model is a great fit for the data.

# --- 4. Visualizing the Results ---
# The best way to understand performance is to see it.
print("\nCreating a visualization of actual vs. predicted values...")

plt.figure(figsize=(10, 7))

# Plot the actual values against the predicted ones.
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color='teal', label='Predicted vs. Actual')

# Plot a dashed red line representing a perfect prediction.
perfect_line = np.linspace(min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max()), 100)
plt.plot(perfect_line, perfect_line, '--', color='red', label='Perfect Prediction')

plt.title('Actual vs. Predicted PM2.5 Concentrations', fontsize=16, fontweight='bold')
plt.xlabel('Actual PM2.5 Concentration ($\mu g/m^3$)', fontsize=12)
plt.ylabel('Predicted PM2.5 Concentration ($\mu g/m^3$)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

print("Execution of Milestone 2 is complete.")