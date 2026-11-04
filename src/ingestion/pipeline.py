
import uuid
from pathlib import Path

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.config.settings import (
    POLICY_DIR,
    SUPPORTED_FILE_EXTENSIONS,
    BATCH_SIZE,
    DOCUMENT_TYPE,
)

from src.ingestion.chunker import parse_and_chunk_document
from src.ingestion.hash_cache import calculate_file_hash

from src.vector_store.qdrant import (
    delete_document_vectors,
    delete_stale_document_vectors,
)


def discover_policy_files() -> dict[str, Path]:
    """Discover supported documents inside the policy directory."""

    if not POLICY_DIR.exists():
        raise FileNotFoundError(
            f"Policy directory does not exist: {POLICY_DIR}"
        )

    files = {}

    for file_path in POLICY_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
            continue

        relative_path = file_path.relative_to(
            POLICY_DIR
        ).as_posix()

        files[relative_path] = file_path

    return files


def build_chunk_ids(
    file_name: str,
    file_hash: str,
    chunk_count: int,
) -> list[str]:
    """Generate deterministic UUIDs for document chunks."""

    return [
        str(
            uuid.uuid5(
                uuid.NAMESPACE_DNS,
                f"{file_name}:{file_hash}:chunk:{index}",
            )
        )
        for index in range(chunk_count)
    ]


def prepare_document_metadata(
    chunks: list[Document],
    file_name: str,
    file_path: Path,
    file_hash: str,
) -> list[Document]:
    """Attach ingestion metadata to every document chunk."""

    for chunk in chunks:
        chunk.metadata.update(
            {
                "source_file": file_name,
                "file_format": file_path.suffix.lstrip(".").upper(),
                "document_type": DOCUMENT_TYPE,
                "content_hash": file_hash,
            }
        )

    return chunks


def upsert_document_chunks(
    vector_store: QdrantVectorStore,
    chunks: list[Document],
    chunk_ids: list[str],
) -> None:
    """Upsert document chunks into Qdrant in batches."""

    total_chunks = len(chunks)

    for start in range(0, total_chunks, BATCH_SIZE):

        end = start + BATCH_SIZE

        vector_store.add_documents(
            documents=chunks[start:end],
            ids=chunk_ids[start:end],
        )

        print(
            f"   Uploaded {min(end, total_chunks)}/{total_chunks} chunks"
        )


def ingest_document(
    client: QdrantClient,
    vector_store: QdrantVectorStore,
    collection_name: str,
    file_name: str,
    file_path: Path,
    file_hash: str,
) -> bool:
    """Parse, chunk and ingest a single document."""

    print(f"\n🔄 Processing: {file_name}")

    chunks = parse_and_chunk_document(file_path)

    if not chunks:
        print(f"⚠️ No valid chunks extracted: {file_name}")
        return False

    prepare_document_metadata(
        chunks=chunks,
        file_name=file_name,
        file_path=file_path,
        file_hash=file_hash,
    )

    chunk_ids = build_chunk_ids(
        file_name=file_name,
        file_hash=file_hash,
        chunk_count=len(chunks),
    )

    print(f"📤 Upserting {len(chunks)} chunks...")

    # Upload the new version before deleting the old one.
    upsert_document_chunks(
        vector_store=vector_store,
        chunks=chunks,
        chunk_ids=chunk_ids,
    )

    # Delete previous versions only after all batches succeed.
    delete_stale_document_vectors(
        client=client,
        collection_name=collection_name,
        file_name=file_name,
        current_hash=file_hash,
    )

    print(f"✅ Successfully ingested: {file_name}")

    return True


def synchronize_documents(
    client: QdrantClient,
    vector_store: QdrantVectorStore,
    collection_name: str,
    hash_cache: dict[str, str],
) -> dict[str, str]:
    """Synchronize local policy documents with Qdrant."""

    current_files = discover_policy_files()

    current_names = set(current_files.keys())
    cached_names = set(hash_cache.keys())

    deleted_files = cached_names - current_names

    updated_cache = {}

    print(f"\nFound {len(current_files)} policy document(s).")

    # -----------------------------------------------------------------------
    # DELETE REMOVED DOCUMENTS
    # -----------------------------------------------------------------------

    for file_name in sorted(deleted_files):

        print(f"\n🗑️ Detected deleted document: {file_name}")

        try:
            delete_document_vectors(
                client=client,
                collection_name=collection_name,
                file_name=file_name,
            )

        except Exception as exc:
            print(f"❌ Could not delete {file_name}: {exc}")

            # Keep its cache entry so deletion is retried next time.
            updated_cache[file_name] = hash_cache[file_name]

    # -----------------------------------------------------------------------
    # PROCESS NEW AND MODIFIED DOCUMENTS
    # -----------------------------------------------------------------------

    for file_name, file_path in sorted(current_files.items()):

        try:
            file_hash = calculate_file_hash(file_path)

            # Skip documents that haven't changed.
            if hash_cache.get(file_name) == file_hash:

                print(f"✨ Skipped (unchanged): {file_name}")

                updated_cache[file_name] = file_hash
                continue

            print(f"\n🔄 Change detected: {file_name}")

            successful = ingest_document(
                client=client,
                vector_store=vector_store,
                collection_name=collection_name,
                file_name=file_name,
                file_path=file_path,
                file_hash=file_hash,
            )

            if successful:
                updated_cache[file_name] = file_hash

            else:
                # Preserve the previous successful version.
                if file_name in hash_cache:
                    updated_cache[file_name] = hash_cache[file_name]

        except Exception as exc:

            print(f"❌ Failed to process {file_name}: {exc}")

            # Do not mark a failed document as successfully ingested.
            if file_name in hash_cache:
                updated_cache[file_name] = hash_cache[file_name]

    return updated_cache
