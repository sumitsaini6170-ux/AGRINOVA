import csv
import numpy as np
import tensorflow as tf
from pathlib import Path

print("====================================")
print(" SOIL MOISTURE MODEL EVALUATION")
print("====================================")

# -----------------------------
# Load dataset
# -----------------------------

data = []

with open("data/soil_moisture.csv", "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append([
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["rainfall"]),
            float(row["soil_moisture"])
        ])

data = np.array(data, dtype=np.float32)

# -----------------------------
# Load normalization values
# -----------------------------

minimum = np.load("models/minimum.npy")
maximum = np.load("models/maximum.npy")

normalized = (data - minimum) / (maximum - minimum + 1e-8)

# -----------------------------
# Create sequences
# -----------------------------

LOOKBACK = 24

X = []
y = []

for i in range(LOOKBACK, len(normalized)):
    X.append(normalized[i - LOOKBACK:i])
    y.append(normalized[i, 3])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# -----------------------------
# Test data
# -----------------------------

split = int(len(X) * 0.8)

X_test = X[split:]
y_test = y[split:]

# -----------------------------
# Load trained model
# -----------------------------

model = tf.keras.models.load_model(
    "models/soil_moisture_lstm.keras"
)

print("\nModel loaded successfully.")

# -----------------------------
# Predictions
# -----------------------------

predicted = model.predict(
    X_test,
    verbose=0
).flatten()

# -----------------------------
# Convert back to actual %
# -----------------------------

soil_min = minimum[3]
soil_max = maximum[3]

actual_soil = y_test * (soil_max - soil_min) + soil_min
predicted_soil = predicted * (soil_max - soil_min) + soil_min

# -----------------------------
# Metrics
# -----------------------------

errors = predicted_soil - actual_soil

mae = np.mean(np.abs(errors))

rmse = np.sqrt(
    np.mean(errors ** 2)
)

ss_res = np.sum(
    (actual_soil - predicted_soil) ** 2
)

ss_tot = np.sum(
    (actual_soil - np.mean(actual_soil)) ** 2
)

r2 = 1 - (ss_res / ss_tot)

# -----------------------------
# Results
# -----------------------------

print("\n====================================")
print(" FINAL MODEL PERFORMANCE")
print("====================================")

print(f"MAE  : {mae:.2f} %")
print(f"RMSE : {rmse:.2f} %")
print(f"R²   : {r2:.4f}")

print("\nFirst 10 Predictions")
print("------------------------------------")

for i in range(10):

    print(
        f"Actual: {actual_soil[i]:6.2f}%"
        f"   Predicted: {predicted_soil[i]:6.2f}%"
    )

# -----------------------------
# Save predictions
# -----------------------------

Path("results").mkdir(exist_ok=True)

with open(
    "results/predictions.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Actual Soil Moisture",
        "Predicted Soil Moisture",
        "Absolute Error"
    ])

    for actual, pred in zip(
        actual_soil,
        predicted_soil
    ):

        writer.writerow([
            round(float(actual), 2),
            round(float(pred), 2),
            round(float(abs(actual - pred)), 2)
        ])

print("\nPredictions saved:")
print("results/predictions.csv")
