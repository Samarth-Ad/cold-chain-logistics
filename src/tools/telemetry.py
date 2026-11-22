
import os
import urllib

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


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
