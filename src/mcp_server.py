from math import asin, cos, isfinite, pi, radians, sin, sqrt

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from sqlalchemy import text

from src.tools.telemetry import get_db_engine

mcp = FastMCP("Cold Chain Logistics")


class TemperatureReading(BaseModel):
    timestamp: str | None
    temperature_c: float


class FleetTelemetryReading(BaseModel):
    timestamp: str | None
    latitude: float | None
    longitude: float | None
    temperature_c: float | None
    cargo_condition_code: str | None
    risk_classification: str | None
    delay_probability: float | None
    port_congestion_level: str | None
    route_risk_index: float | None


class NearbyFleetReading(FleetTelemetryReading):
    distance_km: float


def _fleet_reading(row) -> FleetTelemetryReading:
    return FleetTelemetryReading(
        timestamp=str(row["Timestamp"]) if row["Timestamp"] is not None else None,
        latitude=row["Latitude"],
        longitude=row["Longitude"],
        temperature_c=row["Current_Temperature_C"],
        cargo_condition_code=str(row["Cargo_Condition_Code"]) if row["Cargo_Condition_Code"] is not None else None,
        risk_classification=row["Risk_Classification"],
        delay_probability=row["Delay_Probability"],
        port_congestion_level=str(row["Port_Congestion_Level"]) if row["Port_Congestion_Level"] is not None else None,
        route_risk_index=row["Route_Risk_Index"],
    )


def _distance_km(latitude: float, longitude: float, other_latitude: float, other_longitude: float) -> float:
    """Great-circle distance between two coordinates."""
    lat_delta = radians(other_latitude - latitude)
    lon_delta = radians(other_longitude - longitude)
    haversine = sin(lat_delta / 2) ** 2 + cos(radians(latitude)) * cos(radians(other_latitude)) * sin(lon_delta / 2) ** 2
    return 6371.0 * 2 * asin(min(1.0, sqrt(haversine)))


@mcp.tool()
def find_fleet_readings_near(
    latitude: float,
    longitude: float,
    radius_km: float = 50.0,
    limit: int = 10,
) -> list[NearbyFleetReading]:
    """Find historical telemetry readings within radius_km of a location, nearest first.

    Use this for any question about shipments near a city or GPS point. The default
    radius is 50 km. The view contains historical readings, not verified live assets.
    An empty list means no matching readings within the radius, not no assets anywhere.
    """
    if not isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("Latitude must be between -90 and 90.")
    if not isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("Longitude must be between -180 and 180.")
    if not isfinite(radius_km) or not 0 < radius_km <= 500:
        raise ValueError("Radius must be greater than 0 and at most 500 km.")
    if not 1 <= limit <= 50:
        raise ValueError("The limit must be between 1 and 50.")

    # A latitude band is a safe prefilter; exact distance is checked below.
    latitude_delta = radius_km / 6371.0 * 180 / pi
    statement = text("""
        SELECT [Timestamp], Latitude, Longitude, Current_Temperature_C,
            Cargo_Condition_Code, Risk_Classification, Delay_Probability,
            Port_Congestion_Level, Route_Risk_Index
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Latitude BETWEEN :min_latitude AND :max_latitude
          AND Longitude IS NOT NULL
    """)
    params = {
        "min_latitude": max(-90.0, latitude - latitude_delta),
        "max_latitude": min(90.0, latitude + latitude_delta),
    }

    with get_db_engine().connect() as connection:
        rows = connection.execute(statement, params).mappings()
        matches = []
        for row in rows:
            distance = _distance_km(latitude, longitude, float(row["Latitude"]), float(row["Longitude"]))
            if distance <= radius_km:
                matches.append((distance, row))

    matches.sort(key=lambda item: item[0])
    return [
        NearbyFleetReading(
            **_fleet_reading(row).model_dump(),
            distance_km=round(distance, 2),
        )
        for distance, row in matches[:limit]
    ]


@mcp.tool()
def list_fleet_telemetry(
    limit: int = 10,
    min_temp_c: float | None = None,
    risk_classification: str | None = None,
) -> list[FleetTelemetryReading]:
    """List recent historical fleet telemetry across all locations.

    Optionally filter readings strictly above min_temp_c or by exact risk classification.
    Returns at most 50 readings, newest first. For location questions, use
    find_fleet_readings_near instead. No SQL input is accepted.
    """
    if not 1 <= limit <= 50:
        raise ValueError("The limit must be between 1 and 50.")
    if min_temp_c is not None and not isfinite(min_temp_c):
        raise ValueError("The temperature threshold must be finite.")

    query = """
        SELECT TOP (:limit) [Timestamp], Latitude, Longitude,
            Current_Temperature_C, Cargo_Condition_Code, Risk_Classification,
            Delay_Probability, Port_Congestion_Level, Route_Risk_Index
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE 1 = 1
    """
    params = {"limit": limit}
    if min_temp_c is not None:
        query += " AND Current_Temperature_C > :min_temp_c"
        params["min_temp_c"] = min_temp_c
    if risk_classification is not None:
        query += " AND Risk_Classification = :risk_classification"
        params["risk_classification"] = risk_classification
    query += " ORDER BY [Timestamp] DESC"

    with get_db_engine().connect() as connection:
        rows = connection.execute(text(query), params).mappings()
        return [_fleet_reading(row) for row in rows]


@mcp.tool()
def list_temperature_readings_above(min_temp_c: float, max_temp_c: float | None = None) -> list[TemperatureReading]:
    """List up to 10 historical fleet readings, hottest first.
    Use min_temp_c for a strict lower bound in °C.
    When the user gives a range, also pass max_temp_c as a strict upper bound.
    """
    if not isfinite(min_temp_c):
        raise ValueError("The lower temperature bound must be finite.")

    query = """
        SELECT TOP (10) [Timestamp], Current_Temperature_C
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Current_Temperature_C > :min_temp_c
    """
    params = {"min_temp_c": min_temp_c}

    if max_temp_c is not None:
        if not isfinite(max_temp_c) or max_temp_c <= min_temp_c:
            raise ValueError("The upper bound must be finite and above the lower bound.")
        query += " AND Current_Temperature_C < :max_temp_c"
        params["max_temp_c"] = max_temp_c

    query += " ORDER BY Current_Temperature_C DESC, [Timestamp] DESC"

    with get_db_engine().connect() as connection:
        rows = connection.execute(text(query), params).mappings()
        return [
            TemperatureReading(
                timestamp=str(row["Timestamp"]) if row["Timestamp"] is not None else None,
                temperature_c=row["Current_Temperature_C"],
            )
            for row in rows
        ]


@mcp.tool()
def count_temperature_readings_above(min_temp_c: float) -> dict[str, int]:
    """Count all historical fleet readings strictly above a threshold in °C."""
    if not isfinite(min_temp_c):
        raise ValueError("Temperature threshold must be finite.")

    statement = text("""
        SELECT COUNT_BIG(*)
        FROM FDE_VIEWS.VW_ACTIVE_FLEET
        WHERE Current_Temperature_C > :min_temp_c
    """)

    with get_db_engine().connect() as connection:
        count = connection.execute(
            statement, {"min_temp_c": min_temp_c}
        ).scalar_one()

    return {"matching_readings": count}

if __name__ == "__main__":
    mcp.run(transport="stdio")
