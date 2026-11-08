
from functools import lru_cache

from langchain_core.documents import Document

from src.config.settings import (
    EMBEDDING_MODEL_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBEDDING_MODEL_NAME,
)
from src.embeddings.openrouter_embeddings import (
    OpenRouterNemotronEmbeddings,
)
from src.vector_store.qdrant import initialize_qdrant


@lru_cache(maxsize=2)
def get_embeddings(provider: str):
    """Initialize and cache the configured embedding provider."""

    if provider == "openrouter":
        return OpenRouterNemotronEmbeddings()

    if provider == "ollama":
        from langchain_ollama import OllamaEmbeddings

        return OllamaEmbeddings(
            model=OLLAMA_EMBEDDING_MODEL_NAME,
            base_url=OLLAMA_BASE_URL,
        )

    raise ValueError(f"Unsupported embedding provider: {provider}")


@lru_cache(maxsize=1)
def get_sop_retriever():
    """
    Initialize the Qdrant retriever.

    Uses the same provider-specific collection as ingestion.
    Falls back to Ollama if OpenRouter initialization fails.
    """

    provider = EMBEDDING_MODEL_PROVIDER

    if provider not in {"openrouter", "ollama"}:
        raise ValueError(
            f"Unsupported embedding provider: {provider}"
        )

    embedding_model = None

    if provider == "openrouter":
        try:
            embedding_model = get_embeddings("openrouter")

            test_embedding = embedding_model.embed_query(
                "Embedding connectivity test"
            )

        except Exception as exc:
            print(f"OpenRouter embedding initialization failed: {exc}")
            print("Falling back to Ollama...")

            embedding_model = None

    if embedding_model is None:
        provider = "ollama"
        embedding_model = get_embeddings("ollama")

        test_embedding = embedding_model.embed_query(
            "Embedding connectivity test"
        )

    dimension = len(test_embedding)

    client, vector_store, collection_name = initialize_qdrant(
        embedding_model=embedding_model,
        provider=provider,
        vector_dimension=dimension,
    )

    print(
        f"SOP retriever connected to collection: {collection_name}"
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 2}
    )


def retrieve_sop_documents(query: str) -> list[Document]:
    """Retrieve relevant SOP documents from Qdrant."""

    if not query.strip():
        return []

    retriever = get_sop_retriever()

    return retriever.invoke(query)
