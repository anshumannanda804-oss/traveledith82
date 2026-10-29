"""
train_weather_model.py

Train a Random Forest model using live weather data
from Open-Meteo API.

Author: Anshuman
"""

import requests
import pandas as pd
import random
import warnings
from sklearn.ensemble import RandomForestClassifier

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_weather(latitude, longitude):
    """
    Fetch current weather from Open-Meteo.
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": [
            "temperature_2m",
            "wind_speed_10m",
            "precipitation"
        ]
    }

    response = requests.get(OPEN_METEO_URL, params=params, timeout=10)
    response.raise_for_status()

    data = response.json()

    current = data["current"]

    return {
        "temperature": current["temperature_2m"],
        "wind_speed": current["wind_speed_10m"],
        "precipitation": current["precipitation"]
    }


def create_training_dataset():

    locations = [

        # India
        (20.2961, 85.8245),   # Bhubaneswar
        (19.8135, 85.8312),   # Puri
        (28.6139, 77.2090),   # Delhi
        (19.0760, 72.8777),   # Mumbai
        (13.0827, 80.2707),   # Chennai
        (22.5726, 88.3639),   # Kolkata

        # International
        (51.5074, -0.1278),   # London
        (48.8566, 2.3522),    # Paris
        (40.7128, -74.0060),  # New York
        (35.6762, 139.6503),  # Tokyo
        (25.2048, 55.2708),   # Dubai
        (1.3521, 103.8198)    # Singapore
    ]

    features = []
    labels = []

    for lat, lon in locations:

        try:

            weather = fetch_weather(lat, lon)

            temp = weather["temperature"]
            wind = weather["wind_speed"]
            precip = weather["precipitation"]

            safe = True

            if temp < -5 or temp > 40:
                safe = False

            if wind > 60:
                safe = False

            if precip > 40:
                safe = False

            if random.random() < 0.05:
                safe = not safe

            features.append([temp, wind, precip])
            labels.append(1 if safe else 0)

        except Exception as e:
            print("Weather API Error:", e)

    return pd.DataFrame(
        features,
        columns=[
            "temperature",
            "wind_speed",
            "precipitation"
        ]
    ), labels


def train_weather_model():

    X, y = create_training_dataset()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X.values, y)

    return model


weather_ai_model = train_weather_model()


if __name__ == "__main__":

    model = train_weather_model()

    sample = fetch_weather(20.2961, 85.8245)

    prediction = model.predict([[
        sample["temperature"],
        sample["wind_speed"],
        sample["precipitation"]
    ]])[0]

    print(sample)

    print("SAFE" if prediction == 1 else "UNSAFE")