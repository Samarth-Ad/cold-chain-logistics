"""Checks for the MCP telemetry boundary and its agent adapter."""

import unittest
from unittest.mock import MagicMock, patch

from src.mcp_client import get_mcp_tools
from src.mcp_server import find_fleet_readings_near, list_fleet_telemetry


class FleetTelemetryTests(unittest.TestCase):
    def test_fleet_query_uses_bound_filters_and_bounded_result(self):
        connection = MagicMock()
        connection.execute.return_value.mappings.return_value = []
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection

        with patch("src.mcp_server.get_db_engine", return_value=engine):
            self.assertEqual(
                list_fleet_telemetry(
                    limit=3, min_temp_c=4.0, risk_classification="HIGH"
                ),
                [],
            )

        statement, params = connection.execute.call_args.args
        self.assertIn("TOP (:limit)", str(statement))
        self.assertIn("Risk_Classification = :risk_classification", str(statement))
        self.assertNotIn("HIGH", str(statement))
        self.assertEqual(
            params,
            {"limit": 3, "min_temp_c": 4.0, "risk_classification": "HIGH"},
        )

    def test_limit_is_bounded_before_database_access(self):
        with patch("src.mcp_server.get_db_engine") as engine:
            with self.assertRaises(ValueError):
                list_fleet_telemetry(limit=51)
            engine.assert_not_called()

    def test_nearby_lookup_uses_distance_and_returns_timestamps(self):
        base = {
            "Timestamp": "2021-03-31 13:00:00",
            "Current_Temperature_C": -8.29,
            "Cargo_Condition_Code": "1",
            "Risk_Classification": "High Risk",
            "Delay_Probability": 0.7,
            "Port_Congestion_Level": 2.5,
            "Route_Risk_Index": 4.0,
        }
        near = {**base, "Latitude": 33.8655, "Longitude": -118.0623}
        far = {**base, "Latitude": 33.8, "Longitude": -116.1}
        connection = MagicMock()
        connection.execute.return_value.mappings.return_value = [far, near]
        engine = MagicMock()
        engine.connect.return_value.__enter__.return_value = connection

        with patch("src.mcp_server.get_db_engine", return_value=engine):
            results = find_fleet_readings_near(33.8, -118.1, radius_km=50)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].timestamp, "2021-03-31 13:00:00")
        self.assertLess(results[0].distance_km, 10)
        statement, params = connection.execute.call_args.args
        self.assertIn("Latitude BETWEEN :min_latitude AND :max_latitude", str(statement))
        self.assertLess(params["min_latitude"], 33.8)
        self.assertGreater(params["max_latitude"], 33.8)

    def test_nearby_lookup_rejects_invalid_coordinates(self):
        with patch("src.mcp_server.get_db_engine") as engine:
            with self.assertRaises(ValueError):
                find_fleet_readings_near(91, -118.1)
            engine.assert_not_called()

    def test_agent_discovers_mcp_tools_without_sql_argument(self):
        tools = {tool.name: tool for tool in get_mcp_tools()}
        self.assertEqual(
            set(tools),
            {
                "find_fleet_readings_near",
                "list_fleet_telemetry",
                "list_temperature_readings_above",
                "count_temperature_readings_above",
            },
        )
        self.assertNotIn("sql_query", tools["list_fleet_telemetry"].args)


if __name__ == "__main__":
    unittest.main()
