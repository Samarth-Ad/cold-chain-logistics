"""Compare nearby source CSV rows with the database results returned through MCP."""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_client import get_mcp_tools


def distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2) ** 2
    return 6371.0 * 2 * math.asin(min(1.0, math.sqrt(a)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latitude", type=float, default=33.8)
    parser.add_argument("--longitude", type=float, default=-118.1)
    parser.add_argument("--radius-km", type=float, default=50.0)
    parser.add_argument("--csv-only", action="store_true", help="Check the source CSV without connecting to SQL Server")
    args = parser.parse_args()

    source = PROJECT_ROOT / "data" / "raw" / "dynamic_supply_chain_logistics_dataset.csv"
    nearby = []
    with source.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            latitude = float(row["vehicle_gps_latitude"])
            longitude = float(row["vehicle_gps_longitude"])
            distance = distance_km(args.latitude, args.longitude, latitude, longitude)
            if distance <= args.radius_km:
                nearby.append({
                    "timestamp": row["timestamp"],
                    "latitude": latitude,
                    "longitude": longitude,
                    "temperature_c": float(row["iot_temperature"]),
                    "distance_km": round(distance, 2),
                })

    nearby.sort(key=lambda row: row["distance_km"])
    print(f"Source CSV: {len(nearby)} readings within {args.radius_km:g} km")
    print(json.dumps(nearby[:10], indent=2))

    if args.csv_only:
        return

    tool = next(tool for tool in get_mcp_tools() if tool.name == "find_fleet_readings_near")
    result = tool.invoke({
        "latitude": args.latitude,
        "longitude": args.longitude,
        "radius_km": args.radius_km,
        "limit": 10,
    })
    if result.startswith("Telemetry lookup failed"):
        raise SystemExit(result)
    print("Database via MCP (up to 10 nearest readings):")
    print(json.dumps(json.loads(result), indent=2))


if __name__ == "__main__":
    main()
