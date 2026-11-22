from math import isfinite

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel
from sqlalchemy import text

from src.tools.telemetry import get_db_engine

mcp = FastMCP("Cold Chain Logistics")


class TemperatureReading(BaseModel):
    timestamp: str | None
    temperature_c: float


@mcp.tool()
def list_temperature_readings_above(min_temp_c: float,max_temp_c: float | None = None) -> list[TemperatureReading]:
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