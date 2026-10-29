"""
Smart Soil - MQTT -> InfluxDB Bridge

ESP32 / Simulator
        |
        | MQTT telemetry
        v
     Bridge
        |
        +----> LSTM Forecast
        |
        +----> Open-Meteo Weather API
        |
        +----> Irrigation Decision
        |
        +----> InfluxDB
                  |
                  v
               Grafana

Bridge -> MQTT pump/set -> ESP32 relay

Run:
    python backend/bridge.py

Optional crop:
    CROP=Rice
    CROP=Tomato
    CROP=Maize
    CROP=Wheat
"""

import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------
# External libraries
# ---------------------------------------------------------

import pandas as pd
import paho.mqtt.client as mqtt

from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

# ---------------------------------------------------------
# Project modules
# ---------------------------------------------------------

from src import engine

from src.config import (
    CROPS,
    CSV_PATH,
    FLOW_LPM,
    INFLUX_BUCKET,
    INFLUX_ORG,
    INFLUX_TOKEN,
    INFLUX_URL,
    MODEL_PATH,
    MQTT_HOST,
    MQTT_PORT,
    NODE,
    RAIN_MM,
    TANK_MIN,
    TOPIC,
)

# =========================================================
# OPEN-METEO CONFIGURATION
# =========================================================

WEATHER_LAT = 28.6692
WEATHER_LON = 77.4538

# Weather API refresh interval
WEATHER_REFRESH_SECONDS = 600  # 10 minutes

# Irrigation weather rules
WEATHER_RAIN_PROB_THRESHOLD = 60.0
WEATHER_RAIN_MM_THRESHOLD = 1.0


# =========================================================
# OPEN-METEO WEATHER FUNCTION
# =========================================================

def get_weather():
    """
    Fetch current weather and short-term rain forecast
    from Open-Meteo.

    Returns a dictionary even if API fails.
    """

    try:
        url = "https://api.open-meteo.com/v1/forecast"

        params = {
            "latitude": WEATHER_LAT,
            "longitude": WEATHER_LON,

            "hourly": (
                "temperature_2m,"
                "relative_humidity_2m,"
                "precipitation_probability,"
                "rain,"
                "precipitation"
            ),

            "forecast_days": 2,
            "timezone": "Asia/Kolkata",
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
        )

        response.raise_for_status()

        data = response.json()

        hourly = data["hourly"]

        times = hourly["time"]

        # -------------------------------------------------
        # Find the current hour instead of always using
        # index 0.
        # -------------------------------------------------

        current_time = time.strftime(
            "%Y-%m-%dT%H:00",
            time.localtime()
        )

        current_index = 0

        for i, forecast_time in enumerate(times):
            if forecast_time >= current_time:
                current_index = i
                break

        # -------------------------------------------------
        # Next 3 hours
        # -------------------------------------------------

        end_index = min(
            current_index + 3,
            len(times)
        )

        next_probabilities = (
            hourly["precipitation_probability"]
            [current_index:end_index]
        )

        next_rain = (
            hourly["rain"]
            [current_index:end_index]
        )

        next_precipitation = (
            hourly["precipitation"]
            [current_index:end_index]
        )

        # -------------------------------------------------
        # Current weather
        # -------------------------------------------------

        weather_temperature = float(
            hourly["temperature_2m"][current_index]
        )

        weather_humidity = float(
            hourly["relative_humidity_2m"][current_index]
        )

        rain_probability = float(
            hourly["precipitation_probability"][current_index]
        )

        weather_rain = float(
            hourly["rain"][current_index]
        )

        weather_precipitation = float(
            hourly["precipitation"][current_index]
        )

        # -------------------------------------------------
        # Short-term forecast
        # -------------------------------------------------

        rain_probability_next_3h = float(
            max(next_probabilities)
            if next_probabilities
            else 0.0
        )

        rain_next_3h = float(
            sum(next_rain)
            if next_rain
            else 0.0
        )

        precipitation_next_3h = float(
            sum(next_precipitation)
            if next_precipitation
            else 0.0
        )

        weather = {
            "weather_temperature": weather_temperature,
            "weather_humidity": weather_humidity,
            "rain_probability": rain_probability,
            "weather_rain": weather_rain,
            "weather_precipitation": weather_precipitation,

            "rain_probability_next_3h": (
                rain_probability_next_3h
            ),

            "rain_next_3h": rain_next_3h,

            "precipitation_next_3h": (
                precipitation_next_3h
            ),
        }

        return weather

    except Exception as e:

        print(
            "[weather] API error:",
            type(e).__name__,
            e
        )

        # Safe fallback
        return {
            "weather_temperature": 0.0,
            "weather_humidity": 0.0,
            "rain_probability": 0.0,
            "weather_rain": 0.0,
            "weather_precipitation": 0.0,
            "rain_probability_next_3h": 0.0,
            "rain_next_3h": 0.0,
            "precipitation_next_3h": 0.0,
        }


# =========================================================
# DEFAULT SENSOR VALUES
# =========================================================

DEFAULTS = {
    "moisture": 50.0,
    "temperature": 25.0,
    "humidity": 50.0,
    "rainfall": 0.0,
    "light": 0.0,
    "ph": 6.5,
    "water": 100.0,
}


# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(CSV_PATH)


# =========================================================
# LOAD LSTM MODEL
# =========================================================

model, err = engine.load_model(MODEL_PATH)

print(
    "LSTM model:",
    "loaded"
    if model
    else f"NOT loaded ({err}). Using fallback estimate."
)


# =========================================================
# CROP PROFILE
# =========================================================

crop = dict(
    CROPS[
        os.getenv("CROP", "Wheat")
    ]
)


# =========================================================
# SYSTEM STATE
# =========================================================

st = {
    "mode": "AUTO",

    "pump": False,

    "since": time.time(),

    # LSTM
    "fc": [0.0] * 6,
    "std": 0.0,
    "fc_t": 0.0,

    # Alerts
    "flags": {},

    # Moisture history
    "hist": [],

    # Weather
    "weather": {},
    "weather_t": 0.0,
}


# =========================================================
# INFLUXDB CONNECTION
# =========================================================

influx = InfluxDBClient(
    url=INFLUX_URL,
    token=INFLUX_TOKEN,
    org=INFLUX_ORG,
)

write_api = influx.write_api(
    write_options=SYNCHRONOUS
)


# =========================================================
# MQTT CONNECTION
# =========================================================

mq = mqtt.Client(
    mqtt.CallbackAPIVersion.VERSION2
)

mq.enable_logger()


# =========================================================
# INFLUX WRITE HELPER
# =========================================================

def write(points):
    """
    Write one or more points to InfluxDB.
    """

    write_api.write(
        INFLUX_BUCKET,
        INFLUX_ORG,
        points
    )


# =========================================================
# EVENT LOGGER
# =========================================================

def event(level, text):
    """
    Print event and save it to InfluxDB.
    """

    print(
        f"[{level}] {text}"
    )

    write(
        Point("events")
        .tag("node", NODE)
        .tag("level", level)
        .field("text", text)
    )


# =========================================================
# ALERT FLAG
# =========================================================

def flag(name, cond, level, text):
    """
    Log an event only when a condition starts.
    """

    if cond and not st["flags"].get(name):
        event(level, text)

    st["flags"][name] = cond


# =========================================================
# PUMP CONTROL
# =========================================================

def set_pump(on, why):
    """
    Control pump through MQTT.
    """

    if st["pump"] == on:
        return

    now = time.time()

    # Pump just stopped
    if not on:

        mins = (
            now - st["since"]
        ) / 60

        litres = (
            mins * FLOW_LPM
        )

        write(
            Point("pump_run")
            .tag("node", NODE)
            .field("minutes", mins)
            .field("litres", litres)
        )

    st["pump"] = on
    st["since"] = now

    mq.publish(
        f"{TOPIC}/pump/set",
        "1" if on else "0",
        qos=1,
        retain=True
    )

    event(
        "info" if on else "ok",
        f"Pump {'ON' if on else 'OFF'} ({why})"
    )


# =========================================================
# MQTT COMMAND HANDLER
# =========================================================

def handle_cmd(c):

    # Crop selection
    if c.get("crop") in CROPS:
        crop.update(
            CROPS[c["crop"]]
        )

    # Threshold / target
    for k in ("thr", "target"):

        if k in c:
            crop[k] = float(c[k])

    # Mode
    if "mode" in c:

        st["mode"] = str(
            c["mode"]
        ).upper()

    # Manual pump command
    if "pump" in c:

        st["mode"] = "MANUAL"

        set_pump(
            bool(int(c["pump"])),
            "manual command"
        )

    event(
        "info",
        f"Command applied: {c} "
        f"(mode={st['mode']}, "
        f"thr={crop['thr']}, "
        f"target={crop['target']})"
    )


# =========================================================
# TELEMETRY HANDLER
# =========================================================

def handle_telemetry(d):

    # -----------------------------------------------------
    # Sensor data
    # -----------------------------------------------------

    r = {
        k: float(
            d.get(k, v)
        )
        for k, v in DEFAULTS.items()
    }

    # Soil temperature
    r["soil_temp"] = float(
        d.get(
            "soil_temp",
            r["temperature"] - 1.5
        )
    )

    # -----------------------------------------------------
    # WEATHER UPDATE
    # -----------------------------------------------------

    if (
        time.time()
        - st["weather_t"]
        > WEATHER_REFRESH_SECONDS
    ):

        st["weather"] = get_weather()

        st["weather_t"] = time.time()

        weather = st["weather"]

        print(
            "[weather]",
            f"Rain probability: "
            f"{weather['rain_probability']:.0f}%",
            "| Next 3h probability:",
            f"{weather['rain_probability_next_3h']:.0f}%",
            "| Next 3h rain:",
            f"{weather['rain_next_3h']:.1f} mm",
            "| Temperature:",
            f"{weather['weather_temperature']:.1f} C"
        )

    weather = st["weather"]

    # -----------------------------------------------------
    # ACTUAL RAIN + WATER TANK
    # -----------------------------------------------------

    raining = (
        r["rainfall"] >= RAIN_MM
    )

    tank_ok = (
        r["water"] >= TANK_MIN
    )

    # -----------------------------------------------------
    # LSTM FORECAST
    # -----------------------------------------------------

    if (
        time.time()
        - st["fc_t"]
        > 5
    ):

        st["fc"], st["std"] = engine.forecast(
            model,
            engine.make_window(
                df,
                r
            )
        )

        st["fc_t"] = time.time()

    fc = st["fc"]
    std = st["std"]

    # -----------------------------------------------------
    # EXISTING DECISION ENGINE
    # -----------------------------------------------------

    level, headline, advice, plan = engine.decide(
        r,
        crop,
        fc,
        raining,
        tank_ok
    )

    # -----------------------------------------------------
    # WEATHER FORECAST DECISION
    # -----------------------------------------------------

    rain_expected = False

    if weather:

        rain_expected = (
            weather["rain_probability_next_3h"]
            >= WEATHER_RAIN_PROB_THRESHOLD
            or
            weather["rain_next_3h"]
            >= WEATHER_RAIN_MM_THRESHOLD
        )

    # -----------------------------------------------------
    # AUTO IRRIGATION
    # -----------------------------------------------------

    if st["mode"] == "AUTO":

        # Actual rain detected by sensor
        if raining:

            set_pump(
                False,
                "auto: rain detected"
            )

        # Weather forecast predicts rain
        elif rain_expected:

            set_pump(
                False,
                "auto: rain expected"
            )

        # Tank too low
        elif not tank_ok:

            set_pump(
                False,
                "auto: tank low"
            )

        # Soil moisture below threshold
        elif r["moisture"] < crop["thr"]:

            set_pump(
                True,
                "auto: low moisture"
            )

        # Target reached
        elif r["moisture"] >= crop["target"]:

            set_pump(
                False,
                "auto: target reached"
            )

    # -----------------------------------------------------
    # SAFETY: PUMP ON + TANK LOW
    # -----------------------------------------------------

    elif st["pump"] and not tank_ok:

        set_pump(
            False,
            "tank empty"
        )

    # -----------------------------------------------------
    # MOISTURE HISTORY
    # -----------------------------------------------------

    st["hist"] = (
        st["hist"]
        + [r["moisture"]]
    )[-60:]

    an = engine.anomalies(
        {
            "moisture": st["hist"]
        },
        True
    )

    # -----------------------------------------------------
    # ALERTS
    # -----------------------------------------------------

    flag(
        "low",
        r["moisture"] < crop["thr"]
        and not raining
        and not rain_expected,
        "bad",
        f"WATER REQUIRED: "
        f"moisture {r['moisture']:.1f}% "
        f"< {crop['thr']:.0f}%"
    )

    flag(
        "soon",
        r["moisture"] >= crop["thr"]
        and plan["hours_to_thr"] is not None,
        "warn",
        f"LSTM: moisture below threshold "
        f"in ~{plan['hours_to_thr']} h"
    )

    flag(
        "tank",
        not tank_ok,
        "bad",
        f"Water tank low "
        f"({r['water']:.0f}%), pump locked"
    )

    flag(
        "rain",
        raining,
        "info",
        f"Rain detected "
        f"({r['rainfall']:.1f} mm)"
    )

    flag(
        "weather_rain",
        rain_expected,
        "info",
        f"Weather forecast: rain expected "
        f"within 3h "
        f"({weather.get('rain_probability_next_3h', 0):.0f}% probability, "
        f"{weather.get('rain_next_3h', 0):.1f} mm)"
    )

    flag(
        "anom",
        bool(an),
        "warn",
        an[0] if an else ""
    )

    # =====================================================
    # INFLUXDB TELEMETRY POINT
    # =====================================================

    tele = (
        Point("telemetry")
        .tag("node", NODE)
    )

    # -----------------------------------------------------
    # Normal sensor values
    # -----------------------------------------------------

    for k, v in r.items():

        tele.field(
            k,
            float(v)
        )

    # -----------------------------------------------------
    # Open-Meteo values
    # -----------------------------------------------------

    if weather:

        tele.field(
            "weather_temperature",
            float(
                weather["weather_temperature"]
            )
        )

        tele.field(
            "weather_humidity",
            float(
                weather["weather_humidity"]
            )
        )

        tele.field(
            "rain_probability",
            float(
                weather["rain_probability"]
            )
        )

        tele.field(
            "weather_rain",
            float(
                weather["weather_rain"]
            )
        )

        tele.field(
            "weather_precipitation",
            float(
                weather["weather_precipitation"]
            )
        )

        tele.field(
            "rain_probability_next_3h",
            float(
                weather[
                    "rain_probability_next_3h"
                ]
            )
        )

        tele.field(
            "rain_next_3h",
            float(
                weather["rain_next_3h"]
            )
        )

        tele.field(
            "precipitation_next_3h",
            float(
                weather[
                    "precipitation_next_3h"
                ]
            )
        )

    # =====================================================
    # LSTM FORECAST POINT
    # =====================================================

    f = (
        Point("forecast")
        .tag("node", NODE)

        .field(
            "next_hr",
            float(fc[0])
        )

        .field(
            "std",
            float(std)
        )

        .field(
            "threshold",
            float(crop["thr"])
        )

        .field(
            "target",
            float(crop["target"])
        )

        .field(
            "status",
            {
                "ok": 0,
                "warn": 1,
                "bad": 2
            }[level]
        )

        .field(
            "litres",
            float(plan["litres"])
        )

        .field(
            "minutes",
            float(plan["minutes"])
        )

        .field(
            "rain_expected",
            int(rain_expected)
        )

        .field(
            "rain_probability_next_3h",
            float(
                weather.get(
                    "rain_probability_next_3h",
                    0.0
                )
            )
        )

        .field(
            "rain_next_3h",
            float(
                weather.get(
                    "rain_next_3h",
                    0.0
                )
            )
        )
    )

    # -----------------------------------------------------
    # Store all LSTM forecast steps
    # -----------------------------------------------------

    for i, v in enumerate(
        fc,
        1
    ):

        f.field(
            f"in_{i}h",
            float(v)
        )

    # =====================================================
    # PUMP POINT
    # =====================================================

    pump = (
        Point("pump")
        .tag("node", NODE)

        .field(
            "state",
            int(st["pump"])
        )

        .field(
            "auto",
            int(
                st["mode"] == "AUTO"
            )
        )
    )

    # =====================================================
    # WRITE EVERYTHING TO INFLUXDB
    # =====================================================

    write(
        [
            tele,
            f,
            pump
        ]
    )

    # =====================================================
    # PUBLISH LSTM FORECAST
    # =====================================================

    mq.publish(
        f"{TOPIC}/forecast",
        json.dumps(
            {
                "next_hr": round(
                    fc[0],
                    2
                ),

                "std": round(
                    std,
                    2
                ),

                "status": headline,

                "advice": advice,

                "rain_probability": round(
                    weather.get(
                        "rain_probability",
                        0.0
                    ),
                    1
                ),

                "rain_probability_next_3h": round(
                    weather.get(
                        "rain_probability_next_3h",
                        0.0
                    ),
                    1
                ),

                "rain_next_3h": round(
                    weather.get(
                        "rain_next_3h",
                        0.0
                    ),
                    2
                ),

                "rain_expected": rain_expected,

                "pump": st["pump"],
            }
        )
    )


# =========================================================
# MQTT CONNECT
# =========================================================

def on_connect(
    client,
    userdata,
    flags,
    rc,
    props=None
):

    print(
        "MQTT connected:",
        rc
    )

    client.subscribe(
        [
            (
                f"{TOPIC}/telemetry",
                0
            ),
            (
                f"{TOPIC}/cmd",
                1
            ),
        ]
    )

    # Safe start
    client.publish(
        f"{TOPIC}/pump/set",
        "0",
        qos=1,
        retain=True
    )


# =========================================================
# MQTT MESSAGE
# =========================================================

def on_message(
    client,
    userdata,
    msg
):

    try:

        d = json.loads(
            msg.payload
        )

        if msg.topic.endswith(
            "/cmd"
        ):

            handle_cmd(d)

        else:

            handle_telemetry(d)

    except Exception as e:

        # One bad message must not
        # stop the complete bridge.
        print(
            "Message error:",
            type(e).__name__,
            e
        )


# =========================================================
# MQTT CALLBACKS
# =========================================================

mq.on_connect = on_connect
mq.on_message = on_message


# =========================================================
# START BRIDGE
# =========================================================

print(
    f"Connecting to MQTT "
    f"{MQTT_HOST}:{MQTT_PORT}, "
    f"InfluxDB {INFLUX_URL}, "
    f"crop profile {crop}"
)

print(
    f"Weather API: Open-Meteo "
    f"({WEATHER_LAT}, {WEATHER_LON})"
)

print(
    f"Weather rain threshold: "
    f"{WEATHER_RAIN_PROB_THRESHOLD:.0f}% "
    f"or "
    f"{WEATHER_RAIN_MM_THRESHOLD:.1f} mm "
    f"within 3h"
)

mq.connect(
    MQTT_HOST,
    MQTT_PORT
)

mq.loop_forever()