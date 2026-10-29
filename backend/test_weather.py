import requests

LAT = 28.6692
LON = 77.4538

url = "https://api.open-meteo.com/v1/forecast"

params = {
    "latitude": LAT,
    "longitude": LON,
    "hourly": (
        "temperature_2m,"
        "relative_humidity_2m,"
        "precipitation_probability,"
        "rain,"
        "precipitation"
    ),
    "forecast_days": 1,
    "timezone": "Asia/Kolkata"
}

try:
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()
    hourly = data["hourly"]

    print("\n--- OPEN-METEO WEATHER TEST ---")
    print("Temperature:", hourly["temperature_2m"][0], "C")
    print("Humidity:", hourly["relative_humidity_2m"][0], "%")
    print("Rain Probability:", hourly["precipitation_probability"][0], "%")
    print("Rain:", hourly["rain"][0], "mm")
    print("Precipitation:", hourly["precipitation"][0], "mm")
    print("--------------------------------")

except Exception as e:
    print("Weather API Error:", e)