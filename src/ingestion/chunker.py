
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def create_text_splitter() -> RecursiveCharacterTextSplitter:
    """Create the recursive character text splitter."""

    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )


def chunk_documents(documents: list[Document]) -> list[Document]:
    """
    Split extracted documents into smaller chunks.

    Document metadata, including page numbers, row indexes,
    and Markdown headings, is propagated to the resulting chunks.
    """

    if not documents:
        return []

    splitter = create_text_splitter()

    chunks = splitter.split_documents(documents)

    return [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]


def parse_and_chunk_document(file_path) -> list[Document]:
    """Parse a document and return its chunks."""

    from src.ingestion.document_parser import parse_document

    documents = parse_document(file_path)

    return chunk_documents(documents)
