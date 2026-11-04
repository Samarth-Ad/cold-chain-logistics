
import os
from pathlib import Path

from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
POLICY_DIR = DATA_DIR / "policy"
CACHE_DIR = DATA_DIR / "cache"

HASH_CACHE_FILE = CACHE_DIR / "ingestion_hash_cache.json"


# ---------------------------------------------------------------------------
# DOCUMENT INGESTION CONFIGURATION
# ---------------------------------------------------------------------------

SUPPORTED_FILE_EXTENSIONS = {
    ".md",
    ".txt",
    ".pdf",
    ".csv",
    ".xlsx",
}

CHUNK_SIZE = 600
CHUNK_OVERLAP = 60
BATCH_SIZE = 100

DOCUMENT_TYPE = "Compliance Asset"


# ---------------------------------------------------------------------------
# EMBEDDING CONFIGURATION
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_PROVIDER = os.getenv(
    "EMBEDDING_MODEL_PROVIDER",
    "openrouter",
).strip().lower()

OLLAMA_EMBEDDING_MODEL_NAME = os.getenv(
    "OLLAMA_EMBEDDING_MODEL_NAME",
    "nomic-embed-text",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)


# ---------------------------------------------------------------------------
# QDRANT CONFIGURATION
# ---------------------------------------------------------------------------

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or None

BASE_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "fde-sop-index",
)


# ---------------------------------------------------------------------------
# DIRECTORY INITIALIZATION
# ---------------------------------------------------------------------------

def initialize_directories() -> None:
    """Create directories required by the ingestion pipeline."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
