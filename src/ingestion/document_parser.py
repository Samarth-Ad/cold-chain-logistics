
from pathlib import Path

import pandas as pd
import pypdf

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


# ---------------------------------------------------------------------------
# MARKDOWN PARSER
# ---------------------------------------------------------------------------

def parse_markdown(file_path: Path) -> list[Document]:
    """Parse Markdown while preserving heading metadata."""

    headers_to_split_on = [
        ("#", "Header_1"),
        ("##", "Header_2"),
        ("###", "Header_3"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
    )

    content = file_path.read_text(encoding="utf-8")

    return splitter.split_text(content)


# ---------------------------------------------------------------------------
# TEXT PARSER
# ---------------------------------------------------------------------------

def parse_text(file_path: Path) -> list[Document]:
    """Read a plain-text document."""

    content = file_path.read_text(encoding="utf-8")

    if not content.strip():
        return []

    return [
        Document(page_content=content)
    ]


# ---------------------------------------------------------------------------
# PDF PARSER
# ---------------------------------------------------------------------------

def parse_pdf(file_path: Path) -> list[Document]:
    """Extract text from PDF pages while retaining page numbers."""

    documents = []

    with file_path.open("rb") as pdf_file:
        reader = pypdf.PdfReader(pdf_file)

        for page_number, page in enumerate(reader.pages, start=1):
            page_content = page.extract_text() or ""

            if not page_content.strip():
                continue

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "page_number": page_number,
                    },
                )
            )

    return documents


# ---------------------------------------------------------------------------
# TABULAR DATA PARSER
# ---------------------------------------------------------------------------

def parse_tabular_file(file_path: Path) -> list[Document]:
    """
    Convert CSV or Excel rows into individual LangChain documents.

    Each row becomes a document containing:
        column_name: value | column_name: value
    """

    extension = file_path.suffix.lower()

    if extension == ".csv":
        dataframe = pd.read_csv(file_path)

    elif extension == ".xlsx":
        dataframe = pd.read_excel(file_path)

    else:
        raise ValueError(
            f"Unsupported tabular format: {extension}"
        )

    documents = []

    for row_index, row in dataframe.iterrows():

        row_items = [
            f"{column}: {value}"
            for column, value in row.items()
            if pd.notna(value) and str(value).strip()
        ]

        if not row_items:
            continue

        documents.append(
            Document(
                page_content=" | ".join(row_items),
                metadata={
                    "row_index": int(row_index),
                },
            )
        )

    return documents


# ---------------------------------------------------------------------------
# DOCUMENT PARSER DISPATCHER
# ---------------------------------------------------------------------------

PARSERS = {
    ".md": parse_markdown,
    ".txt": parse_text,
    ".pdf": parse_pdf,
    ".csv": parse_tabular_file,
    ".xlsx": parse_tabular_file,
}


def parse_document(file_path: Path) -> list[Document]:
    """
    Identify the document format and invoke its corresponding parser.

    Returns extracted LangChain documents before chunking.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Document not found: {file_path}"
        )

    extension = file_path.suffix.lower()

    parser = PARSERS.get(extension)

    if parser is None:
        raise ValueError(
            f"Unsupported document format: {extension}"
        )

    documents = parser(file_path)

    return [
        document
        for document in documents
        if document.page_content.strip()
    ]
