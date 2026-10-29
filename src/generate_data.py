import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

output_file = Path("data/soil_moisture.csv")
output_file.parent.mkdir(exist_ok=True)

start_time = datetime(2026, 1, 1, 0, 0)

rows = []

soil = 55.0

for i in range(5000):

    timestamp = start_time + timedelta(hours=i)

    hour = timestamp.hour

    # Temperature pattern
    temperature = (
        24
        + 7 * math.sin((hour - 6) * math.pi / 12)
        + random.uniform(-1.5, 1.5)
    )

    # Humidity inversely related to temperature
    humidity = (
        72
        - 0.65 * (temperature - 20)
        + random.uniform(-4, 4)
    )

    humidity = max(35, min(95, humidity))

    # Random rainfall
    if random.random() < 0.08:
        rainfall = round(random.uniform(2, 25), 2)
    else:
        rainfall = 0.0

    # Soil moisture dynamics
    evaporation = max(0.1, temperature * 0.025)

    soil += rainfall * 0.9
    soil += (humidity - 60) * 0.015
    soil -= evaporation
    soil += random.uniform(-1.0, 1.0)

    soil = max(10, min(95, soil))

    rows.append([
        timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        round(temperature, 2),
        round(humidity, 2),
        rainfall,
        round(soil, 2)
    ])


with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "timestamp",
        "temperature",
        "humidity",
        "rainfall",
        "soil_moisture"
    ])

    writer.writerows(rows)


print("Dataset created successfully!")
print("Rows:", len(rows))
print("File:", output_file)