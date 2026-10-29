import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------
# Load prediction results
# ---------------------------------

actual = []
predicted = []

with open("results/predictions.csv", "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        actual.append(float(row["Actual Soil Moisture"]))
        predicted.append(float(row["Predicted Soil Moisture"]))

actual = np.array(actual)
predicted = np.array(predicted)

# ---------------------------------
# Create output directory
# ---------------------------------

output_dir = Path("results/graphs")
output_dir.mkdir(parents=True, exist_ok=True)

# ---------------------------------
# Graph 1: Actual vs Predicted
# ---------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    actual[:200],
    label="Actual Soil Moisture"
)

plt.plot(
    predicted[:200],
    label="Predicted Soil Moisture"
)

plt.xlabel("Time")
plt.ylabel("Soil Moisture (%)")
plt.title("Actual vs Predicted Soil Moisture")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_dir / "actual_vs_predicted.png",
    dpi=200
)

plt.close()

# ---------------------------------
# Graph 2: Prediction Error
# ---------------------------------

error = predicted - actual

plt.figure(figsize=(12, 5))

plt.plot(
    error[:200],
    label="Prediction Error"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel("Time")
plt.ylabel("Error (%)")
plt.title("Soil Moisture Prediction Error")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_dir / "prediction_error.png",
    dpi=200
)

plt.close()

# ---------------------------------
# Graph 3: Correlation
# ---------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    actual,
    predicted,
    alpha=0.5
)

minimum = min(
    actual.min(),
    predicted.min()
)

maximum = max(
    actual.max(),
    predicted.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)

plt.xlabel("Actual Soil Moisture (%)")
plt.ylabel("Predicted Soil Moisture (%)")
plt.title("Actual vs Predicted Correlation")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    output_dir / "correlation.png",
    dpi=200
)

plt.close()

# ---------------------------------
# Complete
# ---------------------------------

print()
print("================================")
print("GRAPHS CREATED SUCCESSFULLY")
print("================================")

print("Location:")
print("results/graphs/")

print()
print("Files:")

print("1. actual_vs_predicted.png")
print("2. prediction_error.png")
print("3. correlation.png")