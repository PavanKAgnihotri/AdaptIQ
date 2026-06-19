import os, json, uuid, time, requests
from pathlib import Path
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# ── Constants ─────────────────────────────────────────
COLLECTION_NAME  = "adaptiq_docs"
EMBEDDING_MODEL  = "BAAI/bge-small-en-v1.5"
VECTOR_SIZE      = 384
CHUNK_SIZE       = 400
CHUNK_OVERLAP    = 80

DOC_SOURCES = [
    {"url": "https://python.langchain.com/docs/concepts/agents/",
     "category": "Agents", "source": "LangChain Docs — Agents"},
    {"url": "https://python.langchain.com/docs/concepts/rag/",
     "category": "RAG Systems", "source": "LangChain Docs — RAG"},
    {"url": "https://python.langchain.com/docs/concepts/vectorstores/",
     "category": "RAG Systems", "source": "LangChain Docs — Vector Stores"},
    {"url": "https://python.langchain.com/docs/concepts/retrievers/",
     "category": "RAG Systems", "source": "LangChain Docs — Retrievers"},
    {"url": "https://huggingface.co/docs/transformers/main/en/model_doc/bert",
     "category": "NLP & Transformers", "source": "HuggingFace — BERT"},
    {"url": "https://huggingface.co/learn/nlp-course/chapter1/4",
     "category": "NLP & Transformers", "source": "HuggingFace NLP Course"},
    {"url": "https://huggingface.co/docs/peft/main/en/conceptual_guides/lora",
     "category": "LLMs", "source": "HuggingFace — LoRA"},
    {"url": "https://huggingface.co/learn/nlp-course/chapter3/1",
     "category": "LLMs", "source": "HuggingFace — Fine-tuning"},
]

EXTRA_AGENT_SOURCES = [
    {
        "url":      "https://python.langchain.com/docs/concepts/tools/",
        "category": "Agents",
        "source":   "LangChain Docs — Tools"
    },
    {
        "url":      "https://python.langchain.com/docs/concepts/memory/",
        "category": "Agents",
        "source":   "LangChain Docs — Memory"
    },
    {
        "url":      "https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/",
        "category": "Agents",
        "source":   "LangGraph — Agentic Concepts"
    },
]

# ── Qdrant setup ──────────────────────────────────────
qdrant   = QdrantClient(":memory:")
embedder = SentenceTransformer(EMBEDDING_MODEL)

DRIVE_CHUNKS = "/content/drive/MyDrive/adaptiq/chunks.json" #<----change Path here
LOCAL_CHUNKS = "data/chunks.json"#<----change Path here

def get_chunks_path():
    if os.path.exists("/content/drive/MyDrive"):
        Path("/content/drive/MyDrive/adaptiq").mkdir(
            parents=True, exist_ok=True)
        return DRIVE_CHUNKS
    return LOCAL_CHUNKS

def save_chunks(chunks):
    path = get_chunks_path()
    with open(path, "w") as f:
        json.dump(chunks, f)
    print(f"✅ {len(chunks)} chunks saved to Drive")

def load_chunks_from_drive():
    path = get_chunks_path()
    if not os.path.exists(path):
        return []
    with open(path) as f:
        chunks = json.load(f)
    print(f"✅ {len(chunks)} chunks loaded from Drive")
    return chunks

def chunks_saved():
    return os.path.exists(get_chunks_path())

print("✅ Chunk store ready")
print(f"   Drive chunks exist: {chunks_saved()}")

# ── Collection ────────────────────────────────────────

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=VECTOR_SIZE,
                distance=Distance.COSINE
            )
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created")
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' already exists")
        

def fetch_page_text(url):
    try:
        headers = {"User-Agent": "AdaptIQ-RAG-Bot/1.0"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script","style","nav","footer","header"]):
            tag.decompose()
        text  = soup.get_text(separator="\n")
        lines = [l.strip() for l in text.splitlines()]
        return "\n".join(l for l in lines if len(l) > 30)
    except Exception as e:
        print(f"  ⚠️  Failed: {url[:50]} — {e}")
        return ""

def chunk_text(text, source, category):
    chunks, start, idx = [], 0, 0
    while start < len(text):
        end   = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 100:
            chunks.append({
                "id":        str(uuid.uuid4()),
                "text":      chunk,
                "source":    source,
                "category":  category,
                "chunk_idx": idx
            })
            idx += 1
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def ingest_explanations(questions):
    chunks = []
    for q in questions:
        if q.explanation and len(q.explanation) > 30:
            chunks.append({
                "id":        str(uuid.uuid4()),
                "text":      f"Topic: {q.category}\n"
                             f"Question: {q.text}\n"
                             f"Explanation: {q.explanation}",
                "source":    "AdaptIQ Question Bank",
                "category":  q.category,
                "chunk_idx": 0
            })
    return chunks

def embed_and_store(chunks):
    if not chunks:
        return
    texts      = [c["text"] for c in chunks]
    embeddings = embedder.encode(
        texts, batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    points = [
        PointStruct(
            id      = c["id"],
            vector  = emb.tolist(),
            payload = {"text": c["text"], "source": c["source"],
                       "category": c["category"]}
        )
        for c, emb in zip(chunks, embeddings)
    ]
    for i in range(0, len(points), 50):
        qdrant.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i:i+50]
        )

def ingest_extra_agent_docs():
    """Fetches additional Agent documentation (Tools, Memory, LangGraph concepts)"""
    new_chunks = []
    for source in EXTRA_AGENT_SOURCES:
        print(f"  Fetching: {source['source']}...")
        try:
            text = fetch_page_text(source["url"])
            if text:
                chunks = chunk_text(text, source["source"], source["category"])
                embed_and_store(chunks)
                new_chunks.extend(chunks)
                print(f"    ✅ {len(chunks)} chunks added")
        except Exception as e:
            print(f"    ⚠️  Failed: {e}")
        time.sleep(0.5)
    print(f"\n✅ Added {len(new_chunks)} new Agent chunks")
    return new_chunks


def run_ingestion(questions=None, skip_web=False, force=False):
    print("\n" + "="*55)
    print("AdaptIQ RAG Ingestion Pipeline")
    print("="*55)

    ensure_collection()
    all_chunks = []

    # Smart reload — skip web if already saved
    if chunks_saved() and not force:
        print("\n📂 Loading saved chunks from Drive...")
        all_chunks = load_chunks_from_drive()
        embed_and_store(all_chunks)
    else:
        # Static explanations
        if questions:
            print("\n📚 Ingesting question explanations...")
            static = ingest_explanations(questions)
            all_chunks.extend(static)
            print(f"  ✅ {len(static)} explanation chunks")

        # Web docs
        if not skip_web:
            print("\n🌐 Fetching web docs...")
            for src in DOC_SOURCES:
                print(f"  Fetching: {src['source']}...")
                text = fetch_page_text(src["url"])
                if text:
                    chunks = chunk_text(text, src["source"], src["category"])
                    all_chunks.extend(chunks)
                    print(f"    ✅ {len(chunks)} chunks")
                time.sleep(0.5)
                
        extra = ingest_extra_agent_docs()
        all_chunks.extend(extra)

        # Embed into memory
        print(f"\n⚡ Embedding {len(all_chunks)} chunks...")
        embed_and_store(all_chunks)

        # Save to Drive
        save_chunks(all_chunks)

    count = qdrant.get_collection(COLLECTION_NAME).points_count
    print(f"\n{'='*55}")
    print(f"✅ Done! {count} chunks in memory")
    print(f"{'='*55}\n")

print("✅ Ingestion functions ready")


        

