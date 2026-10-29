from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import time
import paho.mqtt.publish as publish

MQTT_HOST = "127.0.0.1"
MQTT_PORT = 1883
TOPIC = "smartsoil/telemetry"


TEST_CASES = {
    "normal": {
        "moisture": 50,
        "temperature": 25,
        "humidity": 60,
        "rainfall": 0,
        "light": 700,
        "ph": 6.5,
        "water": 80
    },

    "dry": {
        "moisture": 20,
        "temperature": 28,
        "humidity": 50,
        "rainfall": 0,
        "light": 700,
        "ph": 6.5,
        "water": 80
    },

    "very_dry": {
        "moisture": 15,
        "temperature": 30,
        "humidity": 40,
        "rainfall": 0,
        "light": 800,
        "ph": 6.5,
        "water": 80
    },

    "rain": {
        "moisture": 20,
        "temperature": 25,
        "humidity": 80,
        "rainfall": 5,
        "light": 300,
        "ph": 6.5,
        "water": 80
    },

    "low_tank": {
        "moisture": 15,
        "temperature": 28,
        "humidity": 50,
        "rainfall": 0,
        "light": 700,
        "ph": 6.5,
        "water": 5
    },

    "target": {
        "moisture": 55,
        "temperature": 25,
        "humidity": 60,
        "rainfall": 0,
        "light": 700,
        "ph": 6.5,
        "water": 80
    },

    "hot_dry": {
        "moisture": 22,
        "temperature": 38,
        "humidity": 35,
        "rainfall": 0,
        "light": 900,
        "ph": 6.5,
        "water": 80
    }
}


def send(data):
    publish.single(
        TOPIC,
        payload=json.dumps(data),
        hostname=MQTT_HOST,
        port=MQTT_PORT
    )


class Handler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        body = json.dumps(data).encode()

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):

        if self.path != "/test-case":
            self.send_json({"error": "Invalid endpoint"}, 404)
            return

        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)

        try:
            request = json.loads(body)
            case = request.get("case")

            if case == "sensor_fault":
                send(TEST_CASES["target"])
                time.sleep(1)
                send(TEST_CASES["dry"])

            elif case in TEST_CASES:
                send(TEST_CASES[case])

            else:
                self.send_json({
                    "error": "Unknown test case",
                    "available": list(TEST_CASES.keys())
                }, 400)
                return

            self.send_json({
                "success": True,
                "test_case": case
            })

        except Exception as e:
            self.send_json({"error": str(e)}, 500)


print("Smart Soil Test API running on http://localhost:5000")

server = ThreadingHTTPServer(("0.0.0.0", 5000), Handler)
server.serve_forever()