
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    VectorParams,
)
from langchain_qdrant import QdrantVectorStore

from src.config.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    BASE_COLLECTION_NAME,
)


def create_qdrant_client() -> QdrantClient:
    """Initialize the Qdrant client."""

    print(f"Connecting to Qdrant: {QDRANT_URL}")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )


def get_collection_name(provider: str) -> str:
    """Generate a provider-specific collection name."""

    return f"{BASE_COLLECTION_NAME}-{provider}"


def create_or_validate_collection(
    client: QdrantClient,
    collection_name: str,
    vector_dimension: int,
) -> None:
    """Create a collection or validate its existing configuration."""

    if client.collection_exists(collection_name):

        collection_info = client.get_collection(collection_name)

        existing_vectors = collection_info.config.params.vectors

        if isinstance(existing_vectors, dict):
            raise ValueError(
                "Named vectors are not supported by this pipeline."
            )

        existing_dimension = existing_vectors.size

        if existing_dimension != vector_dimension:
            raise ValueError(
                f"Vector dimension mismatch. "
                f"Collection: {existing_dimension}, "
                f"Model: {vector_dimension}."
            )

        print(f"Using existing collection: {collection_name}")
        return

    print(
        f"Creating collection: {collection_name} "
        f"({vector_dimension} dimensions)"
    )

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_dimension,
            distance=Distance.COSINE,
        ),
    )


def create_vector_store(
    client: QdrantClient,
    collection_name: str,
    embedding_model: Any,
) -> QdrantVectorStore:
    """Initialize the LangChain Qdrant integration."""

    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
        distance=Distance.COSINE,
    )


def initialize_qdrant(
    embedding_model: Any,
    provider: str,
    vector_dimension: int,
) -> tuple[QdrantClient, QdrantVectorStore, str]:
    """Initialize Qdrant and return the client and vector store."""

    client = create_qdrant_client()

    collection_name = get_collection_name(provider)

    create_or_validate_collection(
        client=client,
        collection_name=collection_name,
        vector_dimension=vector_dimension,
    )

    vector_store = create_vector_store(
        client=client,
        collection_name=collection_name,
        embedding_model=embedding_model,
    )

    print("✅ Qdrant vector store initialized.")

    return client, vector_store, collection_name


def delete_document_vectors(
    client: QdrantClient,
    collection_name: str,
    file_name: str,
) -> None:
    """Delete all vectors associated with a source document."""

    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_file",
                        match=MatchValue(value=file_name),
                    )
                ]
            )
        ),
    )

    print(f"🗑️ Deleted vectors for: {file_name}")


def delete_stale_document_vectors(
    client: QdrantClient,
    collection_name: str,
    file_name: str,
    current_hash: str,
) -> None:
    """Delete older versions of a document after successful ingestion."""

    client.delete(
        collection_name=collection_name,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_file",
                        match=MatchValue(value=file_name),
                    )
                ],
                must_not=[
                    FieldCondition(
                        key="metadata.content_hash",
                        match=MatchValue(value=current_hash),
                    )
                ],
            )
        ),
    )
