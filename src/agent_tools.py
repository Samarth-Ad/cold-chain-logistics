import os
import urllib
import requests
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy import create_engine, text
from functools import lru_cache
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from src.config.settings import (
    EMBEDDING_MODEL_PROVIDER,
    OLLAMA_EMBEDDING_MODEL_NAME,
    OLLAMA_BASE_URL,
)

from src.vector_store.qdrant import initialize_qdrant
from src.embeddings.openrouter_embeddings import OpenRouterNemotronEmbeddings

# 1. ENVIRONMENT CONFIGURATION
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DB_HOST = os.getenv("SQL_SERVER_HOST", "localhost")
DB_PORT = os.getenv("SQL_SERVER_PORT", "1433")
DB_USER = os.getenv("SQL_AGENT_USER", "USR_FDE_RO")
DB_PASSWORD = os.getenv("SQL_AGENT_PASSWORD")

# 2. initializing embedding model based on the provider
@lru_cache(maxsize=1)
def get_embeddings(provider: str):
    """
    Initialize and cache the configured embedding model.

    The same provider and model configuration must be used
    during both ingestion and retrieval.
    """

    if provider == "openrouter":
        print("Initializing OpenRouter embeddings...")

        return OpenRouterNemotronEmbeddings()

    if provider == "ollama":
        print("Initializing Ollama embeddings...")

        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider}"
    )
    
def initialize_agent_vector_store():
    """
    Initialize the same Qdrant collection used by ingestion.
    """

    provider = EMBEDDING_MODEL_PROVIDER

    if provider not in {"openrouter", "ollama"}:
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    embedding_model = None

    # Try the configured provider first.
    if provider == "openrouter":
        try:
            embedding_model = get_embeddings("openrouter")

            # Verify embedding connectivity and dimension.
            test_embedding = embedding_model.embed_query(
                "Embedding connectivity test"
            )

            print(
                f"OpenRouter embeddings ready: "
                f"{len(test_embedding)} dimensions"
            )

        except Exception as exc:
            print(
                f"OpenRouter initialization failed: {exc}"
            )

            print("Falling back to Ollama...")

            embedding_model = None

    # Use Ollama when configured or when OpenRouter fails.
    if embedding_model is None:
        provider = "ollama"

        embedding_model = get_embeddings("ollama")

        test_embedding = embedding_model.embed_query(
            "Embedding connectivity test"
        )

        print(
            f"Ollama embeddings ready: "
            f"{len(test_embedding)} dimensions"
        )

    # Attach to the existing provider-specific collection.
    client, vector_store, collection_name = initialize_qdrant(
        embedding_model=embedding_model,
        provider=provider,
        vector_dimension=len(test_embedding),
    )

    print(
        f"Agent connected to Qdrant collection: "
        f"{collection_name}"
    )

    return client, vector_store 
    
# Initialize the vector store once when this module loads.
qdrant_client, vector_store = initialize_agent_vector_store()

retriever = vector_store.as_retriever(
    search_kwargs={"k": 2}
)

# 3. MSSQL CONNECTION
@lru_cache(maxsize=1)
def get_db_engine():
    """
    Create a reusable connection engine for Docker MSSQL.
    """

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

# 4. CORE FDE AGENT TOOLS
@tool
def query_telemetry_db(sql_query: str) -> str:
    """
    Executes a SQL SELECT query against the
    FDE_VIEWS.VW_ACTIVE_FLEET view.

    Available columns:
    Timestamp, Latitude, Longitude, Current_Temperature_C,
    Cargo_Condition_Code, Risk_Classification,
    Delay_Probability, Port_Congestion_Level,
    Route_Risk_Index.

    Always write standard T-SQL queries.
    """

    normalized_query = sql_query.strip()

    # Basic query validation.
    if not normalized_query.upper().startswith("SELECT"):
        return (
            "SECURITY BLOCK: Only SELECT operations "
            "are authorized."
        )

    # Reject multiple SQL statements.
    if ";" in normalized_query.rstrip(";"):
        return (
            "SECURITY BLOCK: Multiple SQL statements "
            "are not permitted."
        )

    try:
        engine = get_db_engine()

        with engine.connect() as conn:

            cursor = conn.execute(text(normalized_query))

            columns = list(cursor.keys())

            rows = cursor.fetchmany(10)

            if not rows:
                return "No records matched the query criteria."

            formatted_output = (
                f"COLUMNS: {', '.join(columns)}\n"
            )

            for row in rows:
                formatted_output += (
                    str(tuple(row)) + "\n"
                )

            return formatted_output

    except Exception as exc:
        return f"Database Error: {str(exc)}"    
    
@tool
def fetch_corridor_conditions(latitude: float,longitude: float) -> str:
    """
    Fetches real-time weather and corridor conditions
    from Open-Meteo for given GPS coordinates.

    Provides temperature, wind speed, and computed
    corridor congestion index.
    """

    try:
        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current_weather=true"
        )

        response = requests.get(url, timeout=6)

        response.raise_for_status()

        payload = response.json().get(
            "current_weather", {}
        )

        temp = payload.get("temperature", "N/A")
        wind = payload.get("windspeed", 0.0)

        congestion_index = (
            8.5 if wind > 10.0 else 2.5
        )

        status_note = (
            "High Transit Disruption"
            if wind > 10.0
            else "Corridor Normal"
        )

        return (
            "--- LIVE CORRIDOR TELEMETRY ---\n"
            f"Target GPS: {latitude}, {longitude}\n"
            f"External Temp: {temp}°C | "
            f"Wind Speed: {wind} km/h\n"
            f"Corridor Risk: {status_note} "
            f"(Congestion Index: {congestion_index}/10)\n"
            "-------------------------------"
        )

    except Exception as exc:
        return (
            f"Corridor API Communication Failure: {exc}"
        )    
    
@tool
def search_compliance_sop(query: str) -> str:
    """
    Searches enterprise Standard Operating Procedures
    stored in the Qdrant vector database.

    Use this to retrieve regulatory thresholds,
    cold-chain breach mitigations, and rerouting rules.
    """

    try:
        matched_docs = retriever.invoke(query)

        if not matched_docs:
            return "No matching compliance clauses found."

        formatted_context = "\n\n".join(
            [
                (
                    f"[Source: {doc.metadata.get('source_file', 'SOP')} "
                    f"| Format: {doc.metadata.get('file_format', 'RAW')}]\n"
                    f"{doc.page_content}"
                )
                for doc in matched_docs
            ]
        )

        return (
            "--- COMPLIANCE SOP CONTEXT ---\n"
            f"{formatted_context}\n"
            "------------------------------"
        )

    except Exception as exc:
        return (
            f"Vector Store Retrieval Error: {str(exc)}"
        )
    
# 5. LOCAL VERIFICATION
if __name__ == "__main__":

    print("\n--- Testing Tool 1: SQL Telemetry View ---")

    print(
        query_telemetry_db.invoke(
            "SELECT TOP 2 Latitude, Longitude, "
            "Current_Temperature_C "
            "FROM FDE_VIEWS.VW_ACTIVE_FLEET"
        )
    )

    print("\n--- Testing Tool 2: Live Corridor API ---")

    print(
        fetch_corridor_conditions.invoke(
            {
                "latitude": 33.77,
                "longitude": -118.19,
            }
        )
    )

    print("\n--- Testing Tool 3: Qdrant SOP Retrieval ---")

    print(
        search_compliance_sop.invoke(
            "What are the temperature rules "
            "for fresh perishables?"
        )
    )
    
    
    
    