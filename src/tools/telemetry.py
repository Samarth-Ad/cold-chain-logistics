
import os
import urllib

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_core.tools import tool


PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("SQL_SERVER_HOST", "localhost")
DB_PORT = os.getenv("SQL_SERVER_PORT", "1433")
DB_USER = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
DB_PASSWORD = os.getenv("SQL_AGENT_PASSWORD")


@lru_cache(maxsize=1)
def get_db_engine() -> Engine:
    """Create and cache the SQL Server connection engine."""

    if not DB_PASSWORD:
        raise ValueError(
            "SQL_AGENT_PASSWORD is missing from .env"
        )

    connection_string = (
        "DRIVER={ODBC Driver 18 for SQL Server};"
        f"SERVER={DB_HOST},{DB_PORT};"
        "DATABASE=master;"
        f"UID={DB_USER};"
        f"PWD={DB_PASSWORD};"
        "Encrypt=no;"
        "TrustServerCertificate=yes;"
    )

    params = urllib.parse.quote_plus(connection_string)

    return create_engine(
        f"mssql+pyodbc:///?odbc_connect={params}",
        pool_pre_ping=True,
    )


def validate_sql_query(sql_query: str) -> tuple[bool, str]:
    """
    Perform basic read-only query validation.

    The dedicated database user must also have SELECT-only
    permissions on the semantic view.
    """

    query = sql_query.strip()

    if not query:
        return False, "SQL query cannot be empty."

    normalized = query.upper()

    if not normalized.startswith("SELECT"):
        return False, "Only SELECT queries are authorized."

    # Permit one optional trailing semicolon.
    if ";" in query.rstrip(";"):
        return False, "Multiple SQL statements are not permitted."

    # Block common SQL Server mutation and execution constructs.
    forbidden = [
        "INSERT ",
        "UPDATE ",
        "DELETE ",
        "DROP ",
        "ALTER ",
        "CREATE ",
        "TRUNCATE ",
        "EXEC ",
        "EXECUTE ",
        "MERGE ",
        "INTO ",
        "OPENROWSET",
        "OPENDATASOURCE",
    ]

    for keyword in forbidden:
        if keyword in normalized:
            return False, (
                f"Query contains restricted SQL keyword: {keyword.strip()}"
            )

    return True, query


@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Execute a read-only SQL query against the active fleet view.

    Available columns:
    Timestamp, Latitude, Longitude, Current_Temperature_C,
    Cargo_Condition_Code, Risk_Classification,
    Delay_Probability, Port_Congestion_Level,
    Route_Risk_Index.

    Always use standard SQL Server T-SQL syntax.
    Only SELECT queries are supported.
    """

    is_valid, result = validate_sql_query(sql_query)

    if not is_valid:
        return f"SECURITY BLOCK: {result}"

    try:
        engine = get_db_engine()

        with engine.connect() as connection:
            cursor = connection.execute(text(result))

            columns = list(cursor.keys())
            rows = cursor.fetchmany(10)

            if not rows:
                return "No records matched the query criteria."

            formatted_output = (
                f"COLUMNS: {', '.join(columns)}\n"
            )

            for row in rows:
                formatted_output += f"{tuple(row)}\n"

            return formatted_output

    except Exception as exc:
        return f"Database Error: {exc}"
