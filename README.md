# AGRINOVA – AI-Powered Smart Farming Assistant

**AI-based soil-moisture forecasting and intelligent irrigation support using IoT, MQTT, machine learning and real-time dashboards.**

**Project Type:** Individual Academic Project  
**Developer:** Sumit Kumar  
**GitHub Repository:** https://github.com/sumitsaini6170-ux/AGRINOVA

## 1. Project Overview

AGRINOVA is an IoT-enabled smart farming project designed to monitor soil and environmental conditions, forecast soil moisture, and support irrigation decisions. Sensor telemetry is communicated through MQTT, processed by a Python application, stored in InfluxDB, and visualized through Grafana.

The Smart Soil AI component uses an LSTM model to forecast soil moisture from a sequence of recent sensor observations. Weather information and irrigation rules are used alongside sensor readings to support irrigation decisions.

## 2. Problem Statement

Conventional irrigation can lead to inefficient water use and requires frequent manual monitoring. Farmers need timely information about soil moisture, weather conditions and water availability to make better irrigation decisions.

AGRINOVA addresses this problem through sensor-based monitoring, soil-moisture forecasting, rain-aware irrigation logic and a centralized dashboard.

## 3. Project Objectives

- Monitor soil moisture and environmental conditions.
- Forecast soil moisture using an LSTM model.
- Incorporate weather information into irrigation decisions.
- Provide rain-aware pump protection logic.
- Store and visualize sensor and forecast data.
- Support monitoring of water-tank level and irrigation status where the corresponding hardware is configured.

## 4. Main Features

- IoT telemetry over MQTT.
- LSTM-based soil-moisture forecasting.
- Python-based data processing and irrigation decision logic.
- InfluxDB time-series data storage.
- Grafana dashboard for monitoring.
- Weather data integration using Open-Meteo.
- Rain and low-water safeguards in the irrigation decision logic.
- Configurable crop thresholds and irrigation settings.

Feature availability depends on the configured hardware, firmware and services.

## 5. System Architecture

```text
ESP32 / Sensor Node
        |
        | MQTT telemetry
        v
Mosquitto MQTT Broker
        |
        v
Python Processing Bridge
   |             |
   |             +--> LSTM Soil-Moisture Forecast
   |             |
   |             +--> Weather Data / Irrigation Logic
   |
   +--> InfluxDB
           |
           v
      Grafana Dashboard

Python / Irrigation Logic
        |
        | MQTT command topics
        v
Mosquitto --> ESP32 / Pump Controller
```

The project SRS describes these default service ports:

| Service | Default port |
|---|---:|
| Mosquitto MQTT | 1883 |
| InfluxDB | 8086 |
| Grafana | 3000 |
| Test API | 5000 |

## 6. Technology Stack

| Layer | Technology |
|---|---|
| Microcontroller / IoT | ESP32 |
| Firmware | C++ / Arduino environment |
| Backend and processing | Python |
| Machine learning | TensorFlow / Keras, LSTM |
| Data processing | NumPy, Pandas |
| Messaging | MQTT, Mosquitto |
| Time-series database | InfluxDB |
| Visualization | Grafana |
| Weather source | Open-Meteo API |
| Service orchestration | Docker Compose |

Use the versions and configuration files committed in this repository as the source of truth for the current development environment.

## 7. Hardware

The project documentation describes an ESP32-based sensor node and irrigation setup. Components described include:

- ESP32 development board
- Soil-moisture sensor(s)
- Temperature and humidity sensor
- Rain sensor
- Water-level / tank-level sensing
- Relay or pump-control module
- Water pump and suitable external power supply

Confirm exact sensor models, GPIO pin mapping, electrical ratings and final bill of materials against the actual prototype wiring before assembling or powering the circuit. Do not connect a pump directly to an ESP32 GPIO pin.

## 8. Machine Learning Model

The Smart Soil AI SRS describes an LSTM model with:

- Input window: 24 time steps
- Input features: 4
- Output: one predicted soil-moisture value

The SRS records model diagnostics of MAE 1.75 and RMSE 2.97. These are reported model results from the documented evaluation and are not a guarantee of field performance. Refer to the model evaluation outputs and dataset details in the repository for context.

## 9. Irrigation Decision Logic

The documented default Wheat profile uses:

- Irrigation start threshold: 30% soil moisture
- Target moisture: 55%
- Documented pH range: 6.0–7.5

The SRS describes this decision priority:

1. If the water tank is low, keep the pump OFF.
2. If rain is detected, keep the pump OFF.
3. If rain is expected within the next three hours, keep the pump OFF.
4. If the pump is OFF and soil moisture is at or below the configured start threshold, turn it ON.
5. If the pump is ON and moisture is below the target, keep it ON.
6. At or above the target moisture, turn the pump OFF.

The SRS documents a rain forecast condition of precipitation probability at least 60% or expected rainfall of at least 1 mm in the next three hours. These values are configurable project rules, not universal agronomic recommendations.

## 10. Repository Structure

```text
AGRINOVA/
├── backend/
├── dashboard/
├── data/
├── esp32/
├── grafana/
├── influxdb/
├── models/
├── mosquitto/
├── results/
├── src/
├── docs/
├── main.py
├── requirements.txt
├── docker-compose.yml
└── README.md
```

## 11. Setup

### Prerequisites

- Git
- Python version compatible with the repository dependencies
- Docker and Docker Compose
- Arduino IDE or another supported ESP32 development environment (for hardware firmware)
- Configured ESP32 and sensors, if running with physical hardware

### Clone the repository

```bash
git clone https://github.com/sumitsaini6170-ux/AGRINOVA.git
cd AGRINOVA
```

### Install Python dependencies

**Windows:**

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Start containerized services

From the repository root, if the Docker Compose configuration is complete:

```bash
docker compose up -d
```

Check service status:

```bash
docker compose ps
```

Open Grafana and InfluxDB using the host and ports configured in your environment. Configure credentials and tokens locally; never commit secrets to GitHub.

## 12. Configuration

Before running the application, review the repository configuration for:

- MQTT broker host, port and topic names.
- InfluxDB URL, organization, bucket and token.
- Grafana data-source configuration.
- Weather API settings and location coordinates.
- ESP32 Wi-Fi and MQTT settings.
- Sensor calibration values and irrigation thresholds.

The SRS identifies example MQTT topics including:

- `smartsoil/telemetry`
- `smartsoil/cmd`
- `smartsoil/pump/set`

Use the exact topic names and payload formats implemented in the current source code.

## 13. Running the Project

The exact start commands depend on the scripts and configuration currently present in the repository.

A typical development sequence is:

1. Start the required Docker services using `docker compose up -d`.
2. Configure and start the MQTT broker, InfluxDB and Grafana.
3. Configure the ESP32 firmware with the correct Wi-Fi and MQTT settings, or use the project's supported test/simulator input.
4. Start the Python processing or bridge script specified by the repository.
5. Open Grafana and select the configured dashboard.
6. Check telemetry, forecast output and irrigation status.
7. Test pump-control behavior safely with the correct power supply and relay wiring.

Check the source files and their command-line arguments before launching a script. Do not assume every service is enabled by default.

## 14. Testing and Validation

The SRS defines scenarios covering:

- Normal sensor conditions.
- Low and very low soil moisture.
- Rain detection.
- Low water-tank level.
- Target moisture reached.
- Hot and dry conditions.
- Sensor fault handling.
- LSTM input/output shape.

Record each test with its input, expected result, actual result, date and evidence. Mark a test **PASS** only after it has been executed and its result recorded. Keep incomplete tests marked **PENDING**.

Physical pump/relay operation should be described as verified only when supported by actual hardware test evidence.

## 15. Project Documents and Evidence

Recommended contents for the repository `docs/` directory:

```text
docs/
├── Problem_Statement.pdf
├── SRS.pdf
├── Project_Report.pdf
├── Architecture_Diagram.png
├── Test_Results.pdf
└── evidence/
    ├── dashboard_screenshot_1.png
    ├── dashboard_screenshot_2.png
    └── hardware_photos/
```

Only list files here after they have actually been added to the repository.

## 16. Demo

- GitHub repository: https://github.com/sumitsaini6170-ux/AGRINOVA
- Demo video: **Add the accessible video-sharing URL here after uploading the demo video.**
- Dashboard screenshots: Add final screenshots under `docs/evidence/`.

Ensure that the video-sharing permission allows the evaluator to view the video.

## 17. Security and Safety

- Do not commit Wi-Fi passwords, API keys, InfluxDB tokens or other credentials.
- Store secrets in local environment variables or an untracked configuration file.
- Use an appropriately rated external supply and driver/relay for the pump.
- Verify wiring and common-ground requirements against the hardware module documentation.
- Test pump control with a safe setup before unattended operation.

## 18. Project Ownership

**Project Type:** Individual Academic Project  
**Developer:** Sumit Kumar

This project is developed and maintained individually. The implementation, documentation and project submission are presented as an individual academic submission.

## 19. Acknowledgement

Developed as an individual academic smart-agriculture project. Project guidance, implementation details and verification status are documented in the accompanying project documents.
