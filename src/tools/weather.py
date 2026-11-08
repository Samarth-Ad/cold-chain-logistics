
import requests

from langchain_core.tools import tool


OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


@tool
def fetch_corridor_conditions(
    latitude: float,
    longitude: float,
) -> str:
    """
    Fetch current weather and corridor conditions using Open-Meteo.

    Inputs:
        latitude: Target GPS latitude.
        longitude: Target GPS longitude.

    Returns:
        Current temperature, wind speed and computed
        corridor disruption indicator.
    """

    if not -90 <= latitude <= 90:
        return "Invalid latitude. Must be between -90 and 90."

    if not -180 <= longitude <= 180:
        return "Invalid longitude. Must be between -180 and 180."

    try:
        response = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current_weather": "true",
            },
            timeout=10,
        )

        response.raise_for_status()

        payload = response.json().get(
            "current_weather", {}
        )

        if not payload:
            return "No current weather data available."

        temperature = payload.get("temperature", "N/A")
        wind_speed = payload.get("windspeed", 0.0)

        congestion_index = (
            8.5 if wind_speed > 10.0 else 2.5
        )

        status = (
            "High Transit Disruption"
            if wind_speed > 10.0
            else "Corridor Normal"
        )

        return (
            "--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temperature: {temperature}°C\n"
            f"Wind Speed: {wind_speed} km/h\n"
            f"Corridor Risk: {status}\n"
            f"Congestion Index: {congestion_index}/10\n"
            "-------------------------------"
        )

    except requests.RequestException as exc:
        return f"Corridor API Communication Failure: {exc}"

    except Exception as exc:
        return f"Weather Tool Error: {exc}"
