import csv
import numpy as np
import tensorflow as tf
from pathlib import Path

print("TensorFlow:", tf.__version__)

# -----------------------------
# 1. Load CSV
# -----------------------------

file_path = Path("data/soil_moisture.csv")

data = []

with open(file_path, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        data.append([
            float(row["temperature"]),
            float(row["humidity"]),
            float(row["rainfall"]),
            float(row["soil_moisture"])
        ])

data = np.array(data, dtype=np.float32)

print("Dataset shape:", data.shape)

# -----------------------------
# 2. Normalize data
# -----------------------------

minimum = data.min(axis=0)
maximum = data.max(axis=0)

normalized = (data - minimum) / (maximum - minimum + 1e-8)

# -----------------------------
# 3. Create sequences
# -----------------------------

LOOKBACK = 24

X = []
y = []

for i in range(LOOKBACK, len(normalized)):

    X.append(normalized[i - LOOKBACK:i])

    # Future soil moisture
    y.append(normalized[i, 3])

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

print("X shape:", X.shape)
print("y shape:", y.shape)

# -----------------------------
# 4. Train/Test split
# -----------------------------

split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# -----------------------------
# 5. Build LSTM model
# -----------------------------

model = tf.keras.Sequential([

    tf.keras.layers.Input(
        shape=(LOOKBACK, 4)
    ),

    tf.keras.layers.LSTM(
        64,
        return_sequences=True
    ),

    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.LSTM(
        32
    ),

    tf.keras.layers.Dense(
        16,
        activation="relu"
    ),

    tf.keras.layers.Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)

model.summary()

# -----------------------------
# 6. Train
# -----------------------------

history = model.fit(
    X_train,
    y_train,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# -----------------------------
# 7. Evaluate
# -----------------------------

loss, mae = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print()
print("==============================")
print("MODEL RESULTS")
print("==============================")
print("Test Loss:", loss)
print("Test MAE:", mae)

# -----------------------------
# 8. Save model
# -----------------------------

Path("models").mkdir(exist_ok=True)

model.save(
    "models/soil_moisture_lstm.keras"
)

# Save normalization values
np.save(
    "models/minimum.npy",
    minimum
)

np.save(
    "models/maximum.npy",
    maximum
)

print()
print("Model saved successfully!")
print("models/soil_moisture_lstm.keras")
