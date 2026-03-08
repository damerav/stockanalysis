"""Knowledge Base Builder — chunks and embeds docs/code into pgvector.

Run once to populate the knowledge_vectors table:
    python -m src.data.build_knowledge_base

Re-run after significant changes to docs/ or src/.
"""
import os
import re
import logging
from sentence_transformers import SentenceTransformer
from src.data.db_router import get_router

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
BATCH_SIZE = 64

# Directories to walk recursively
SOURCE_DIRS = [
    ("docs", [".md", ".txt"]),
    ("src", [".py"]),
    (".kiro/steering", [".md"]),
    ("scripts", [".sh", ".py"]),
]

# Individual root-level files to include
ROOT_FILES = [
    "PROMPT.md",
    "kiro-dgx-spark-setup.md",
    "config.yaml",
    "requirements.txt",
    "README.md",
]

# Directories/patterns to skip
SKIP_PATTERNS = [
    "__pycache__",
    ".git",
    "node_modules",
    ".venv",
    "data/",
]


def _should_skip(path: str) -> bool:
    """Check if a path should be excluded."""
    for pat in SKIP_PATTERNS:
        if pat in path:
            return True
    return False


def _chunk_markdown(text: str, source_path: str) -> list[str]:
    """Split markdown files by sections, then by size if needed.

    Preserves section headers as context in each chunk for better
    semantic search relevance.
    """
    # Split on markdown headers (##, ###, etc.)
    sections = re.split(r'\n(?=#{1,4}\s)', text)
    chunks = []
    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) <= CHUNK_SIZE:
            chunks.append(section)
        else:
            # Large section — split by paragraphs first
            header_match = re.match(r'(#{1,4}\s.*?\n)', section)
            header = header_match.group(1) if header_match else ""
            for sub in _chunk_text(section, CHUNK_SIZE, CHUNK_OVERLAP):
                # Prepend header to sub-chunks for context
                if header and not sub.startswith(header):
                    sub = header + sub
                chunks.append(sub)
    return chunks if chunks else _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)


def _chunk_python(text: str, source_path: str) -> list[str]:
    """Split Python files by class/function boundaries, then by size.

    Preserves the module docstring and imports as context prefix.
    """
    lines = text.split('\n')

    # Extract module header (docstring + imports, up to 30 lines)
    header_lines = []
    for i, line in enumerate(lines[:30]):
        if line.startswith(('import ', 'from ', '"""', "'''", '#')) or not line.strip():
            header_lines.append(line)
        elif i > 0 and header_lines:
            break
    module_header = '\n'.join(header_lines[:15])  # Cap at 15 lines

    # Split on class/function definitions
    boundaries = []
    for i, line in enumerate(lines):
        if re.match(r'^(class |def |async def )', line):
            boundaries.append(i)

    if not boundaries:
        return _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)

    chunks = []
    # Add module header as first chunk if substantial
    if len(module_header.strip()) > 50:
        chunks.append(f"[{source_path}]\n{module_header}")

    for idx, start in enumerate(boundaries):
        end = boundaries[idx + 1] if idx + 1 < len(boundaries) else len(lines)
        block = '\n'.join(lines[start:end]).strip()
        if not block:
            continue
        # Prefix with file path for context
        block = f"[{source_path}]\n{block}"
        if len(block) <= CHUNK_SIZE:
            chunks.append(block)
        else:
            for sub in _chunk_text(block, CHUNK_SIZE, CHUNK_OVERLAP):
                chunks.append(sub)

    return chunks if chunks else _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)


def _chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Fallback: split text into overlapping character-level chunks."""
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
    """Walk SOURCE_DIRS + ROOT_FILES and return all matching file paths."""
    paths = []

    # Root-level files
    for fname in ROOT_FILES:
        if os.path.isfile(fname):
            paths.append(fname)

    # Directory walks
    for directory, extensions in SOURCE_DIRS:
        if not os.path.isdir(directory):
            continue
        for root, _, files in os.walk(directory):
            if _should_skip(root):
                continue
            for fname in files:
                if any(fname.endswith(ext) for ext in extensions):
                    fpath = os.path.join(root, fname)
                    if not _should_skip(fpath):
                        paths.append(fpath)

    return sorted(set(paths))


def _smart_chunk(content: str, path: str) -> list[str]:
    """Route to the appropriate chunker based on file type."""
    if path.endswith(('.md', '.txt')):
        return _chunk_markdown(content, path)
    elif path.endswith('.py'):
        return _chunk_python(content, path)
    else:
        # YAML, shell scripts, etc. — prefix with path
        prefixed = f"[{path}]\n{content}"
        return _chunk_text(prefixed, CHUNK_SIZE, CHUNK_OVERLAP)


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
            if not content.strip():
                continue
            chunks = _smart_chunk(content, path)
            for idx, chunk in enumerate(chunks):
                if len(chunk.strip()) > 20:  # Skip trivially small chunks
                    all_chunks.append((path, idx, chunk.strip()))
        except Exception as e:
            logger.warning("Skipping %s: %s", path, e)

    logger.info("Total chunks to embed: %d from %d files", len(all_chunks), len(files))

    # Embed in batches
    texts = [c[2] for c in all_chunks]
    embeddings = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_embs = model.encode(batch, show_progress_bar=False, normalize_embeddings=True)
        embeddings.extend(batch_embs.tolist())
        if (i // BATCH_SIZE) % 10 == 0:
            logger.info("Embedded %d / %d chunks", min(i + BATCH_SIZE, len(texts)), len(texts))

    logger.info("Embedding complete. Inserting into PostgreSQL...")

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
        # Insert in batches to avoid memory issues
        batch_insert = 500
        for i in range(0, len(rows), batch_insert):
            cur.executemany(insert_sql, rows[i:i + batch_insert])
            conn.commit()

        logger.info("Successfully inserted %d chunks into knowledge_vectors.", len(rows))

        # Log summary by source type
        from collections import Counter
        ext_counts = Counter()
        for path, _, _ in all_chunks:
            ext = os.path.splitext(path)[1] or path
            ext_counts[ext] += 1
        for ext, count in ext_counts.most_common():
            logger.info("  %s: %d chunks", ext, count)

    except Exception as e:
        logger.error("Failed to insert knowledge base rows: %s", e)
        conn.rollback()
    finally:
        cur.close()
        router.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    build_knowledge_base()
