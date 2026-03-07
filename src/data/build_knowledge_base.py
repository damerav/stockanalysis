"""Knowledge Base Builder — chunks and embeds docs/code into pgvector.

Run once to populate the knowledge_vectors table:
    python -m src.data.build_knowledge_base

Re-run after significant changes to docs/ or src/.
"""
import os
import logging
from sentence_transformers import SentenceTransformer
from src.data.db_router import get_router

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
BATCH_SIZE = 64

SOURCE_DIRS = [
    ("docs", [".md"]),
    ("src", [".py"]),
]


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping character-level chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += size - overlap
    return chunks


def _collect_files() -> list[str]:
    """Walk SOURCE_DIRS and return all matching file paths."""
    paths = []
    for directory, extensions in SOURCE_DIRS:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    paths.append(os.path.join(root, fname))
    return sorted(paths)


def build_knowledge_base(truncate: bool = True):
    """Main entry point. Embeds all source files into knowledge_vectors."""
    logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
    model = SentenceTransformer(EMBEDDING_MODEL)

    files = _collect_files()
    logger.info("Found %d source files to embed.", len(files))

    all_chunks: list[tuple[str, int, str]] = []
    for path in files:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            for idx, chunk in enumerate(_chunk_text(content)):
                all_chunks.append((path, idx, chunk))
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)

    logger.info("Total chunks to embed: %d", len(all_chunks))

    texts = [c[2] for c in all_chunks]
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_embs.tolist())
        logger.info("Embedded %d / %d chunks", min(i + BATCH_SIZE, len(texts)), len(texts))

    router = get_router()
    conn = router._pg_conn
    if not conn:
        logger.error("PostgreSQL required for knowledge base. Aborting.")
        return
    cur = conn.cursor()
    try:
        if truncate:
            cur.execute("TRUNCATE TABLE knowledge_vectors RESTART IDENTITY;")
            logger.info("Truncated existing knowledge_vectors rows.")

        insert_sql = """
            INSERT INTO knowledge_vectors (source_path, chunk_index, chunk_text, embedding)
            VALUES (%s, %s, %s, %s)
        """
        rows = [
            (all_chunks[i][0], all_chunks[i][1], all_chunks[i][2], embeddings[i])
            for i in range(len(all_chunks))
        ]
        cur.executemany(insert_sql, rows)
        conn.commit()
        logger.info("Successfully inserted %d chunks into knowledge_vectors.", len(rows))
    except Exception as e:
        logger.error("Failed to insert knowledge base rows: %s", e)
        conn.rollback()
    finally:
        cur.close()
        router.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_knowledge_base()
