from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# Files
CSV_PATH = BASE / "data" / "soil_moisture.csv"
MODEL_PATH = BASE / "models" / "soil_moisture_lstm.keras"
PRED_PATH = BASE / "results" / "predictions.csv"
DB_PATH = BASE / "data" / "smartsoil.db"

# LSTM
FEATURES = [
    "temperature",
    "humidity",
    "rainfall",
    "soil_moisture",
]

WINDOW = 24
STEPS = 6
MOISTURE_SCALE = 1.0

# MQTT
MQTT_HOST = "localhost"
MQTT_PORT = 1883
TOPIC = "smartsoil"
NODE = "field-01"

# InfluxDB
INFLUX_URL = "http://localhost:8086"
INFLUX_TOKEN = "smartsoil-token"
INFLUX_ORG = "smartsoil"
INFLUX_BUCKET = "farm"

# Irrigation
RAIN_MM = 1.0
TANK_MIN = 10
FLOW_LPM = 12.0

FIELD_AREA_M2 = 10.0
ROOT_DEPTH_MM = 300.0
COST_PER_KL = 15.0
BASELINE_MIN_PER_DAY = 30

# Crop profiles
CROPS = {
    "Wheat": {
        "thr": 30,
        "target": 55,
        "ph": (6.0, 7.5),
    },
    "Rice": {
        "thr": 50,
        "target": 80,
        "ph": (5.5, 7.0),
    },
    "Tomato": {
        "thr": 40,
        "target": 65,
        "ph": (6.0, 6.8),
    },
    "Maize": {
        "thr": 35,
        "target": 60,
        "ph": (5.8, 7.0),
    },
    "Custom": {
        "thr": 30,
        "target": 55,
        "ph": (5.5, 7.5),
    },
}