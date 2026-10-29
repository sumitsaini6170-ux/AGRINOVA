"""
Virtual ESP32 for testing without hardware (closed loop: soil dries, bridge starts pump, soil recovers).
Run:  python pipeline/simulator.py --scenario Heatwave --speed 4
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import paho.mqtt.client as mqtt  # noqa: E402

from src.config import MQTT_HOST, MQTT_PORT, TANK_MIN, TOPIC  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", default="Normal", choices=["Normal", "Heatwave", "Rain", "SensorFault"])
ap.add_argument("--speed", type=int, default=4, help="1-10, how fast the soil changes")
ap.add_argument("--interval", type=float, default=2.0, help="seconds between messages")
ap.add_argument("--start", type=float, default=60.0, help="starting soil moisture %")
a = ap.parse_args()

s = {"m": a.start, "water": 80.0, "pump": False, "t": 0}


def on_connect(client, userdata, flags, rc, props=None):
    client.subscribe(f"{TOPIC}/pump/set", qos=1)


def on_message(client, userdata, msg):
    s["pump"] = msg.payload == b"1"


mq = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
mq.on_connect, mq.on_message = on_connect, on_message
mq.connect(MQTT_HOST, MQTT_PORT)
mq.loop_start()
print(f"Simulating '{a.scenario}' -> {TOPIC}/telemetry every {a.interval}s (Ctrl+C to stop)")

while True:
    s["t"] += 1
    t, k = s["t"], a.speed
    temp = {"Heatwave": 38, "Rain": 22}.get(a.scenario, 27) + 3 * np.sin(t / 40)
    rain = float(5 + 3 * np.random.rand()) if a.scenario == "Rain" else 0.0
    watering = s["pump"] and s["water"] >= TANK_MIN
    s["m"] = float(np.clip(s["m"] - (0.12 + 0.012 * max(temp - 20, 0)) * k + 0.5 * rain * k + (1.5 * k if watering else 0)
                           + np.random.normal(0, 0.05), 0, 100))
    if watering:
        s["water"] = max(0.0, s["water"] - 0.25 * k)
    shown = s["m"]
    if a.scenario == "SensorFault" and np.random.rand() < 0.08:
        shown = float(np.clip(shown + np.random.choice([-45, 45]), 0, 100))
    msg = {"moisture": round(shown, 2), "temperature": round(temp, 2), "rainfall": round(rain, 2),
           "humidity": round(float(np.clip(75 - (temp - 20) * 1.3 + np.random.normal(0, 1), 15, 98)), 2),
           "light": round(float(max(0, 450 + 350 * np.sin(t / 30))), 0), "ph": round(6.5 + 0.15 * np.sin(t / 50), 2),
           "water": round(s["water"], 1)}
    mq.publish(f"{TOPIC}/telemetry", json.dumps(msg))
    print(f"moisture={msg['moisture']:5.1f}%  temp={msg['temperature']:4.1f}  rain={msg['rainfall']:3.1f}  "
          f"tank={msg['water']:5.1f}%  pump={'ON ' if s['pump'] else 'OFF'}")
    time.sleep(a.interval)