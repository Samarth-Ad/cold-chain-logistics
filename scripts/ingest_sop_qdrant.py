
import sys
from pathlib import Path


# Ensure project root is available for src imports.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.config.settings import initialize_directories

from src.ingestion.hash_cache import (
    load_hash_cache,
    save_hash_cache,
)

from src.ingestion.pipeline import synchronize_documents

from src.vector_store.qdrant import initialize_qdrant


def initialize_embeddings():
    """Initialize the configured embedding provider."""

    from src.config.settings import (
        EMBEDDING_MODEL_PROVIDER,
        OLLAMA_EMBEDDING_MODEL_NAME,
        OLLAMA_BASE_URL,
    )

    provider = EMBEDDING_MODEL_PROVIDER

    if provider not in {"openrouter", "ollama"}:
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    embedding_model = None

    if provider == "openrouter":

        try:
            from src.embeddings.openrouter_embeddings import (
                OpenRouterNemotronEmbeddings,
            )

            print("☁️ Initializing OpenRouter embeddings...")

            embedding_model = OpenRouterNemotronEmbeddings()

            test_embedding = embedding_model.embed_query(
                "Embedding connectivity test"
            )

        except Exception as exc:
            print(f"⚠️ OpenRouter initialization failed: {exc}")
            print("🔄 Falling back to Ollama...")

            embedding_model = None

    if embedding_model is None:

        from langchain_ollama import OllamaEmbeddings

        embedding_model = OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
        )

        test_embedding = embedding_model.embed_query(
            "Embedding connectivity test"
        )

        provider = "ollama"

    dimension = len(test_embedding)

    print(
        f"✅ Embedding provider: {provider} "
        f"({dimension} dimensions)"
    )

    return embedding_model, provider, dimension


def main() -> None:
    """Execute the incremental SOP ingestion pipeline."""

    print("\n========== SOP INGESTION PIPELINE ==========\n")

    initialize_directories()

    # Load previous ingestion state.
    hash_cache = load_hash_cache()

    # Initialize embedding provider.
    embedding_model, provider, dimension = initialize_embeddings()

    # Initialize Qdrant.
    client, vector_store, collection_name = initialize_qdrant(
        embedding_model=embedding_model,
        provider=provider,
        vector_dimension=dimension,
    )

    # Synchronize documents.
    updated_cache = synchronize_documents(
        client=client,
        vector_store=vector_store,
        collection_name=collection_name,
        hash_cache=hash_cache,
    )

    # Persist successful ingestion state.
    if updated_cache != hash_cache:

        save_hash_cache(updated_cache)

        print("\n✅ Hash cache synchronized.")

    else:
        print("\n🌴 No cache changes required.")

    print("\n========== INGESTION COMPLETED ==========\n")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n⚠️ Ingestion interrupted by user.")
        sys.exit(130)

    except Exception as exc:
        print(f"\n❌ Ingestion pipeline failed: {exc}")
        raise
