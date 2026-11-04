import os 
import hashlib
from dotenv import load_dotenv
import json
from pathlib import Path
import pandas as pd
import pypdf
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
    FilterSelector
)
from langchain_qdrant import QdrantVectorStore
import uuid
import sys
script_dir = Path(__file__).parent
project_root = script_dir.parent
load_dotenv(project_root / ".env")

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from src.embeddings.openrouter_embeddings import OpenRouterNemotronEmbeddings

cache_dir = project_root / "data" / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
HASH_CACHE_FILE = cache_dir / "ingestion_hash_cache.json"

hash_cache = {}
if HASH_CACHE_FILE.exists():
    try:
        with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
            hash_cache = json.load(f)

    except (json.JSONDecodeError, OSError) as e:
        print(f"⚠️ Could not load hash cache: {e}")
        hash_cache = {}
        
# ==========================================
# 2. EMBEDDING PROVIDER INITIALIZATION
# ==========================================

EMBEDDING_MODEL_PROVIDER = os.getenv(
    "EMBEDDING_MODEL_PROVIDER",
    "openrouter"
).strip().lower()

embedding_model = None
active_provider = None

if EMBEDDING_MODEL_PROVIDER == "openrouter":

    print("☁️ Attempting OpenRouter Nemotron embeddings...")

    try:
        from src.embeddings.openrouter_embeddings import (
            OpenRouterNemotronEmbeddings
        )

        embedding_model = OpenRouterNemotronEmbeddings()

        # Test connectivity and determine dimension
        test_embedding = embedding_model.embed_query(
            "Testing Nemotron embedding connectivity"
        )

        TARGET_DIMENSION = len(test_embedding)
        active_provider = "openrouter"

        print(
            f"✅ OpenRouter connected. "
            f"Dimension: {TARGET_DIMENSION}"
        )

    except Exception as e:

        print(f"⚠️ OpenRouter initialization failed: {e}")
        print("🔄 Falling back to Ollama...")

        embedding_model = None


if embedding_model is None:

    if EMBEDDING_MODEL_PROVIDER not in ("openrouter", "ollama"):
        raise ValueError(
            f"Unsupported embedding provider: "
            f"{EMBEDDING_MODEL_PROVIDER}"
        )

    from langchain_ollama import OllamaEmbeddings

    embedding_model = OllamaEmbeddings(
        model=os.getenv(
            "OLLAMA_EMBEDDING_MODEL_NAME",
            "nomic-embed-text"
        ),
        base_url=os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434"
        )
    )

    # Test local embedding model
    test_embedding = embedding_model.embed_query(
        "Embedding initialization test"
    )

    TARGET_DIMENSION = len(test_embedding)
    active_provider = "ollama"

    print(
        f"✅ Ollama connected. "
        f"Embedding dimension: {TARGET_DIMENSION}"
    )


embeddings = embedding_model
        

# ==========================================
# 3. QDRANT PROVISIONING
# ==========================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333"
)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

BASE_COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION_NAME",
    "fde-sop-index"
)

# Separate collection for each embedding provider
COLLECTION_NAME = f"{BASE_COLLECTION_NAME}-{active_provider}"

print(f"Connecting to Qdrant: [{QDRANT_URL}]...")

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY or None
)

# Check whether collection exists
if qdrant_client.collection_exists(COLLECTION_NAME):

    print(f"Collection already exists: {COLLECTION_NAME}")

    collection_info = qdrant_client.get_collection(
        COLLECTION_NAME
    )

    existing_vectors = collection_info.config.params.vectors

    if isinstance(existing_vectors, dict):
        raise ValueError(
            "Existing collection uses named vectors. "
            "This script expects a single unnamed vector."
        )

    existing_dimension = existing_vectors.size

    if existing_dimension != TARGET_DIMENSION:
        raise ValueError(
            f"Dimension mismatch! "
            f"Collection: {existing_dimension}, "
            f"Model: {TARGET_DIMENSION}. "
            f"Use a new collection or migrate existing vectors."
        )

else:

    print(
        f"Creating collection: {COLLECTION_NAME} "
        f"({TARGET_DIMENSION} dimensions)"
    )

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=TARGET_DIMENSION,
            distance=Distance.COSINE
        )
    )

# Initialize LangChain integration
vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embeddings,
    distance=Distance.COSINE
)

print("✅ Qdrant vector store initialized successfully.")
        

# =========================================
# 4. DOCUMENT PARSING AND CHUNKING
# ========================================
def parse_and_chunk_document(doc_path: Path) -> list[Document]:
    ext = doc_path.suffix.lower()
    raw_chunks: list[Document] = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=60)
    
    if ext == ".md":
        headers_to_split_on = [("#", "Header_1"), ("##", "Header_2"), ("###", "Header_3")]
        md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        raw_text = doc_path.read_text(encoding="utf-8")
        header_docs = md_splitter.split_text(raw_text)
        raw_chunks = text_splitter.split_documents(header_docs)
        
    elif ext == ".txt":
        raw_text = doc_path.read_text(encoding="utf-8")
        raw_docs = [Document(page_content=raw_text)]
        raw_chunks = text_splitter.split_documents(raw_docs)
        
    elif ext == ".pdf":
        pdf_docs = []
        try:
            with open(doc_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for page_num, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        pdf_docs.append(Document(page_content=page_text, metadata={"page_number": page_num + 1}))
            raw_chunks = text_splitter.split_documents(pdf_docs)
        except Exception as e:
            print(f"  ❌ Error parsing PDF {doc_path.name}: {e}")
            return []
        
    elif ext in [".csv", ".xlsx"]:
        try:
            df = pd.read_csv(doc_path) if ext == ".csv" else pd.read_excel(doc_path)
        except Exception as e:
            print(f"  ❌ Error reading table: {e}")
            return []
            
        for idx, row in df.iterrows():
            row_dict = row.to_dict()
            row_items = [
                f"{str(col)}: {str(val)}" 
                for col, val in row_dict.items() 
                if pd.notna(val) and str(val).strip() != ""
            ]
            
            if row_items:
                row_text = " | ".join(row_items)
                doc_item = Document(page_content=row_text, metadata={"row_index": int(idx)})
                raw_chunks.append(doc_item)

    # sanity check
    valid_chunks = []
    for chunk in raw_chunks:
        clean_text = chunk.page_content.strip()
        if clean_text:
            chunk.page_content = clean_text
            valid_chunks.append(chunk)
            
    return valid_chunks

# ==========================================
# 5. INCREMENTAL PIPELINE WITH BATCHING
# ==========================================
policy_dir = project_root / "data" / "policy"
target_patterns = ["*.md", "*.txt", "*.pdf", "*.csv", "*.xlsx"]
current_files = {}
for pattern in target_patterns:
    for file_path in policy_dir.glob(pattern):
        current_files[file_path.name] = file_path

print(f"Found {len(current_files)} policy file(s) in {policy_dir}...")

updated_cache = {}
cache_modified = False

cached_filenames = set(hash_cache.keys())
current_filenames = set(current_files.keys())
deleted_files = cached_filenames - current_filenames

for deleted_file in deleted_files:
    print(f"🗑️ Detected deleted file: {deleted_file}. Purging from Pinecone...")
    try:
        index_client.delete(filter={"source_file": {"$eq": deleted_file}})
        cache_modified = True
    except Exception as e:
        print(f"  ❌ Failed to purge {deleted_file}: {e}")

def delete_document_vectors(file_name: str):
    """
    Delete all Qdrant points associated with a source document.
    """

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source_file",
                        match=MatchValue(value=file_name)
                    )
                ]
            )
        )
    )

    print(f"🗑️ Deleted existing vectors for: {file_name}")

for file_name, file_path in current_files.items():

    file_bytes = file_path.read_bytes()
    file_hash = hashlib.md5(file_bytes).hexdigest()

    if hash_cache.get(file_name) == file_hash:
        print(f"✨ Skipped (Unchanged): {file_name}")
        updated_cache[file_name] = file_hash
        continue

    print(f"🔄 Processing updates: {file_name}...")

    try:

        chunks = parse_and_chunk_document(file_path)

        if not chunks:
            print(f"⚠️ No valid chunks extracted from {file_name}")
            continue

        explicit_ids = []

        for idx, chunk in enumerate(chunks):

            chunk.metadata["source_file"] = file_name
            chunk.metadata["file_format"] = (
                file_path.suffix.replace(".", "").upper()
            )
            chunk.metadata["document_type"] = "Compliance Asset"

            explicit_ids.append(
                str(
                    uuid.uuid5(
                        uuid.NAMESPACE_DNS,
                        f"{file_name}-chunk-{idx}"
                    )
                )
            )

        total_chunks = len(chunks)
        batch_size = 100

        print(f"📤 Upserting {total_chunks} chunks...")

        # Delete old vectors only after parsing succeeds.
        delete_document_vectors(file_name)

        for i in range(0, total_chunks, batch_size):

            vector_store.add_documents(
                documents=chunks[i:i + batch_size],
                ids=explicit_ids[i:i + batch_size]
            )

        # Commit hash only after successful ingestion
        updated_cache[file_name] = file_hash

        print(f"✅ Successfully ingested {file_name}")

    except Exception as e:

        print(f"❌ Error during ingestion of {file_name}: {e}")

        # Do not mark failed files as successfully ingested

# ==========================================
# 6. SYNC HASH CACHE
# ==========================================
if cache_modified:
    with open(HASH_CACHE_FILE, "w") as f:
        json.dump(updated_cache, f, indent=4)
    print("✅ Ingestion & cache update complete.")
else:
    print("🌴 Index is already up-to-date.")       
    
if updated_cache != hash_cache:

    with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(updated_cache, f, indent=4)

    print("✅ Hash cache synchronized.")
else:
    print("🌴 No cache changes required.") 