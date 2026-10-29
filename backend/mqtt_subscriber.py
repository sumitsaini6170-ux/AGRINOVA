import os
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
CSV_PATH = BASE / "data" / "soil_moisture.csv"
MODEL_PATH = BASE / "models" / "soil_moisture_lstm.keras"
PRED_PATH = BASE / "results" / "predictions.csv"
DB_PATH = BASE / "data" / "smartsoil.db"

# LSTM contract: 24 hours x these 4 features (exact order) -> next-hour soil moisture
FEATURES = ["temperature", "humidity", "rainfall", "soil_moisture"]
WINDOW = 24
STEPS = 6                 # forecast horizon on chart (hours)
MOISTURE_SCALE = 1.0      # set to 100.0 ONLY if train_model.py divided soil_moisture by 100

RAIN_MM = 1.0             # rainfall >= this = rain detected
TANK_MIN = 10             # pump locked below this tank level (%)

# Field / pump model (edit for your setup)
FLOW_LPM = 12.0           # pump flow, litres per minute
FIELD_AREA_M2 = 10.0      # irrigated bed area
ROOT_DEPTH_MM = 300.0     # root zone depth (1 mm of water over 1 m2 = 1 litre)
COST_PER_KL = 15.0        # Rs per 1000 litres
BASELINE_MIN_PER_DAY = 30 # fixed-timer schedule used to estimate water saved

CROPS = {
    "Wheat":  {"thr": 30, "target": 55, "ph": (6.0, 7.5)},
    "Rice":   {"thr": 50, "target": 80, "ph": (5.5, 7.0)},
    "Tomato": {"thr": 40, "target": 65, "ph": (6.0, 6.8)},
    "Maize":  {"thr": 35, "target": 60, "ph": (5.8, 7.0)},
    "Custom": {"thr": 30, "target": 55, "ph": (5.5, 7.5)},
}

# --- MQTT / InfluxDB (defaults match docker-compose.yml) ---
MQTT_HOST = os.getenv("MQTT_HOST", "localhost")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
NODE = "farm01"
TOPIC = f"smartsoil/{NODE}"          # telemetry, forecast, cmd, pump/set live under this prefix
INFLUX_URL = os.getenv("INFLUX_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUX_TOKEN", "smartsoil-token")
INFLUX_ORG = "smartsoil"
INFLUX_BUCKET = "farm"