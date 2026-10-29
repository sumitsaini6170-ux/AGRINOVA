// Smart Soil AI - ESP32 firmware
// Libraries: DHT sensor library (Adafruit), Adafruit Unified Sensor
// Sends: moisture, temperature, humidity, rainfall, light. The backend fills pH / water level with defaults.
#include <WiFi.h>
#include <HTTPClient.h>
#include <DHT.h>

const char* WIFI_SSID = "YOUR_WIFI";
const char* WIFI_PASS = "YOUR_PASSWORD";
const char* SERVER_URL = "http://192.168.1.10:5000/data";  // PC running backend/server.py

#define DHT_PIN    4
#define SOIL_PIN   34   // capacitive soil moisture sensor (analog)
#define RAIN_PIN   35   // rain sensor module AO (low value = wet)
#define LDR_PIN    32   // optional LDR
#define RELAY_PIN  26   // pump relay (active HIGH; flip if your module is active LOW)

const int SOIL_DRY = 3200;  // ADC in dry air  -> calibrate
const int SOIL_WET = 1300;  // ADC in water    -> calibrate

DHT dht(DHT_PIN, DHT22);

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  dht.begin();
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println("\nWiFi connected");
}

void loop() {
  float t = dht.readTemperature();
  float h = dht.readHumidity();
  if (isnan(t) || isnan(h)) { Serial.println("DHT read failed"); delay(2000); return; }

  float moisture = constrain(map(analogRead(SOIL_PIN), SOIL_DRY, SOIL_WET, 0, 100), 0, 100);
  // A rain module gives wetness, not mm. Mapped to 0-10 "mm" so the dashboard can detect rain.
  float rain = constrain(map(analogRead(RAIN_PIN), 4095, 1500, 0, 10), 0, 10);
  float light = analogRead(LDR_PIN) * (2000.0 / 4095.0);  // rough lux scale

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(SERVER_URL);
    http.addHeader("Content-Type", "application/json");
    String body = "{\"moisture\":" + String(moisture, 1) + ",\"temperature\":" + String(t, 1) +
                  ",\"humidity\":" + String(h, 1) + ",\"rainfall\":" + String(rain, 1) +
                  ",\"light\":" + String(light, 0) + "}";
    int code = http.POST(body);
    if (code == 200) {
      String resp = http.getString();  // {"ok":true,"pump":1}
      digitalWrite(RELAY_PIN, resp.indexOf("\"pump\":1") >= 0 ? HIGH : LOW);
    }
    Serial.printf("POST %d  moisture=%.1f temp=%.1f\n", code, moisture, t);
    http.end();
  }
  delay(5000);
}