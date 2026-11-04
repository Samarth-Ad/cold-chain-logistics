
import hashlib
import json
from pathlib import Path

from src.config.settings import HASH_CACHE_FILE, CACHE_DIR


def load_hash_cache() -> dict[str, str]:
    """Load previously ingested document hashes."""

    if not HASH_CACHE_FILE.exists():
        return {}

    try:
        with HASH_CACHE_FILE.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        if not isinstance(cache, dict):
            raise ValueError("Hash cache must contain a JSON object.")

        return cache

    except (json.JSONDecodeError, OSError, ValueError) as exc:
        print(f"⚠️ Could not load hash cache: {exc}")
        return {}


def save_hash_cache(cache: dict[str, str]) -> None:
    """Save the hash cache atomically."""

    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    temporary_file = HASH_CACHE_FILE.with_suffix(".tmp")

    with temporary_file.open("w", encoding="utf-8") as file:
        json.dump(cache, file, indent=4)

    temporary_file.replace(HASH_CACHE_FILE)


def calculate_file_hash(file_path: Path) -> str:
    """Calculate the SHA-256 hash of a file."""

    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()
