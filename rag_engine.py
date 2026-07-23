import os
import re
import sys
import sqlite3
import logging
import argparse
import subprocess
import tempfile
from pathlib import Path

import requests
import pypdfium2 as pdfium
import sqlite_vec

logger = logging.getLogger("rag_engine")
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s [rag_engine] %(message)s"))
    logger.addHandler(_handler)
    logger.propagate = False
logger.setLevel(logging.INFO)


# ===== Embedding server connection =====

# A second llama-server instance, loaded with an embedding-only model
# (e.g. bge-small, nomic-embed-text), reachable somewhere on the local
# network -- same client pattern as llm_engine's chat server.
EMBED_HOST = os.environ.get("EMBED_HOST", "127.0.0.1")
EMBED_PORT = os.environ.get("EMBED_PORT", "8081")
CONNECT_TIMEOUT = 5
REQUEST_TIMEOUT = float(os.environ.get("EMBED_TIMEOUT", 30))

def set_embed_server_target(host, port=None):
    """Override EMBED_HOST/PORT at runtime."""
    global EMBED_HOST, EMBED_PORT
    EMBED_HOST = host
    if port is not None:
        EMBED_PORT = str(port)

def _embed_base_url():
    # noinspection HttpUrlsUsage
    return f"http://{EMBED_HOST}:{EMBED_PORT}"

_session = None

def get_session():
    global _session
    if _session is None:
        _session = requests.Session()
    return _session

def check_embed_server():
    """Return True if the embedding llama.cpp server is reachable."""
    try:
        resp = get_session().get(f"{_embed_base_url()}/health", timeout=CONNECT_TIMEOUT)
        return resp.ok
    except requests.RequestException as e:
        logger.error(f"Embedding server unreachable at {_embed_base_url()}: {e}")
        return False


_embed_dim = None

def embed_texts(texts):
    """Embed a list of strings, returning a list of float vectors in the same order."""
    resp = get_session().post(
        f"{_embed_base_url()}/v1/embeddings",
        json={"input": texts},
        timeout=(CONNECT_TIMEOUT, REQUEST_TIMEOUT),
    )
    resp.raise_for_status()
    return [row["embedding"] for row in resp.json()["data"]]

def embed_text(text):
    return embed_texts([text])[0]

def get_embed_dim():
    """Determine the embedding vector size by asking the server once, then cache it."""
    global _embed_dim
    if _embed_dim is None:
        _embed_dim = len(embed_text("dimension probe"))
    return _embed_dim


# ===== Storage (sqlite-vec) =====

KNOWLEDGE_BASE_DIR = Path(__file__).resolve().parent / "knowledge_base"
DB_PATH = KNOWLEDGE_BASE_DIR / "rag.sqlite3"

_db = None

def get_db():
    global _db
    if _db is None:
        KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
        _db = sqlite3.connect(DB_PATH)
        _db.enable_load_extension(True)
        sqlite_vec.load(_db)
        _db.enable_load_extension(False)
        _init_schema(_db)
    return _db

def _init_schema(db):
    db.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            book TEXT NOT NULL,
            language TEXT NOT NULL,
            level TEXT NOT NULL,
            lesson TEXT NOT NULL,
            text TEXT NOT NULL
        )
    """)
    db.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunk_vectors USING vec0(
            embedding FLOAT[{get_embed_dim()}]
        )
    """)
    db.commit()


# ===== Ingestion =====

CHUNK_SIZE = 800     # characters, roughly 200-300 tokens
CHUNK_OVERLAP = 150  # characters of overlap when a paragraph must be hard-split

# Below this ratio of letters to non-whitespace characters, a chunk is
# treated as OCR noise (misread dot-leaders/underlines from worksheet
# layouts) and dropped rather than fed to the tutor as "reference material".
MIN_ALPHA_RATIO = 0.6

def _is_garbage(text):
    stripped = "".join(text.split())
    if not stripped:
        return True
    alpha = sum(1 for c in stripped if c.isalpha())
    return (alpha / len(stripped)) < MIN_ALPHA_RATIO

def _split_into_chunks(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Paragraph-aware chunking: pack paragraphs up to `size` chars per chunk,
    hard-splitting (with overlap) any single paragraph longer than `size`."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = ""
    for para in paragraphs:
        if buf and len(buf) + len(para) + 1 > size:
            chunks.append(buf)
            buf = ""
        if len(para) > size:
            start = 0
            while start < len(para):
                chunks.append(para[start:start + size])
                start += size - overlap
            continue
        buf = f"{buf}\n{para}".strip()
    if buf:
        chunks.append(buf)
    return chunks

# Scanned-book fallback: pages with less than this much embedded text are
# treated as image-only and OCR via the tesseract CLI instead. Requires
# `tesseract` on PATH with the target language pack installed
# (e.g. `brew install tesseract-lang` for `deu`).
OCR_LANG = os.environ.get("OCR_LANG", "deu")
OCR_DPI = 300
MIN_TEXT_LAYER_CHARS = 20

def set_ocr_lang(lang):
    """Override OCR_LANG at runtime (e.g. from main.py), using tesseract's
    3-letter language codes (e.g. "deu", not "de")."""
    global OCR_LANG
    OCR_LANG = lang

def _ocr_page(page, page_number):
    """Render `page` to an image and OCR it via the tesseract CLI. Returns ""
    (and logs) if tesseract isn't available or fails on this page."""
    bitmap = page.render(scale=OCR_DPI / 72)
    image = bitmap.to_pil()
    bitmap.close()
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        image.save(tmp_path)
        result = subprocess.run(
            ["tesseract", tmp_path, "stdout", "-l", OCR_LANG],
            capture_output=True, text=True, timeout=60,
        )
        if result.returncode != 0:
            logger.warning(f"tesseract OCR failed on page {page_number}: {result.stderr.strip()}")
            return ""
        return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.error(f"Could not run tesseract for OCR (page {page_number}): {e}")
        return ""
    finally:
        os.unlink(tmp_path)

def _extract_pdf_text(path, page_start=None, page_end=None):
    """Extract text from `path`. Pages are 1-indexed and inclusive; omit both
    to read the whole document. Pages with no real embedded text layer
    (scanned PDFs) are OCR via tesseract as a fallback."""
    doc = pdfium.PdfDocument(path)
    start = (page_start - 1) if page_start else 0
    end = page_end if page_end else len(doc)

    pages_text = []
    for i in range(start, end):
        page = doc[i]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        textpage.close()
        if len(text.strip()) < MIN_TEXT_LAYER_CHARS:
            text = _ocr_page(page, i + 1)
        pages_text.append(text)
        page.close()
    doc.close()
    return "\n\n".join(pages_text)

def ingest_pdf(path, book, language, level, lesson, page_start=None, page_end=None):
    """Extract, chunk, embed and store one PDF (or a page range of it) under the
    given (book, language, level, lesson) metadata. `language` should be a
    short code (e.g. "de") so it lines up with the rest of the app's language
    handling. Safe to re-run: any existing chunks for that exact
    (book, language, level, lesson) quadruple are replaced first."""
    text = _extract_pdf_text(path, page_start, page_end)
    chunks = _split_into_chunks(text)
    n_before = len(chunks)
    chunks = [c for c in chunks if not _is_garbage(c)]
    if n_before - len(chunks):
        logger.info(f"Dropped {n_before - len(chunks)} of {n_before} chunks as OCR noise")
    if not chunks:
        logger.warning(f"No usable text extracted from {path}")
        return 0

    db = get_db()
    db.execute(
        "DELETE FROM chunk_vectors WHERE rowid IN "
        "(SELECT id FROM chunks WHERE book = ? AND language = ? AND level = ? AND lesson = ?)",
        (book, language, level, lesson),
    )
    db.execute(
        "DELETE FROM chunks WHERE book = ? AND language = ? AND level = ? AND lesson = ?",
        (book, language, level, lesson),
    )

    vectors = embed_texts(chunks)
    for chunk_text, vector in zip(chunks, vectors):
        cur = db.execute(
            "INSERT INTO chunks (book, language, level, lesson, text) VALUES (?, ?, ?, ?, ?)",
            (book, language, level, lesson, chunk_text),
        )
        db.execute(
            "INSERT INTO chunk_vectors (rowid, embedding) VALUES (?, ?)",
            (cur.lastrowid, sqlite_vec.serialize_float32(vector)),
        )
    db.commit()
    logger.info(f"Ingested {len(chunks)} chunks from {path} ({book}, {language}, {level}, {lesson})")
    return len(chunks)


# ===== Retrieval =====

OVERFETCH = 5  # candidates fetched per requested result, before language/level filtering

def retrieve_context(query, k=3, language=None, level=None):
    """Return up to `k` relevant chunk strings for `query`, each labeled with its
    source (book/language/lesson/level) so the tutor can ground its answer in
    it. Pass `language` (e.g. "de") to only consider chunks from books in that
    language, and/or `level` (e.g. "A2") to only consider chunks tagged at
    that CEFR level. Returns [] if the embedding server or knowledge base is
    unreachable/empty."""
    try:
        query_vector = embed_text(query)
    except (requests.RequestException, KeyError, IndexError) as e:
        logger.error(f"Could not embed query: {e}")
        return []

    # A shared, multi-language knowledge base means most candidates can be in
    # the "wrong" language/level, so overfetch harder whenever a filter is
    # active to still have enough left after filtering.
    fetch_k = k * OVERFETCH * (3 if (language or level) else 1)

    db = get_db()
    rows = db.execute(
        """
        SELECT c.text, c.book, c.language, c.level, c.lesson
        FROM chunk_vectors v
        JOIN chunks c ON c.id = v.rowid
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
        """,
        (sqlite_vec.serialize_float32(query_vector), fetch_k),
    ).fetchall()

    if language:
        rows = [r for r in rows if r[2] == language]
    if level:
        rows = [r for r in rows if r[3] == level]
    rows = rows[:k]

    return [f"({book} - {lesson}, {lang}, {lvl}): {text}" for text, book, lang, lvl, lesson in rows]


def init_engine():
    """Verify the embedding server is reachable and the knowledge base has at
    least one chunk stored. Returns True if RAG is usable right now."""
    if not check_embed_server():
        return False
    try:
        db = get_db()
        count = db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.Error as e:
        logger.error(f"Could not read knowledge base at {DB_PATH}: {e}")
        return False
    if count == 0:
        logger.warning(f"Knowledge base at {DB_PATH} is empty; run `python rag_engine.py ingest ...` first.")
        return False
    logger.info(f"RAG ready: {count} chunks indexed, embedding server at {_embed_base_url()}.")
    return True


# ===== CLI =====

def main():
    parser = argparse.ArgumentParser(description="RAG engine - manage the tutor's PDF knowledge base")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest_p = sub.add_parser("ingest", help="Ingest a PDF (or a page range of it) under given metadata")
    ingest_p.add_argument("pdf", help="Path to the PDF file")
    ingest_p.add_argument("--book", required=True, help="Book name, e.g. 'Begegnungen'")
    ingest_p.add_argument("--language", required=True, help="Language code, e.g. 'de'")
    ingest_p.add_argument("--level", required=True, help="CEFR level, e.g. 'A2'")
    ingest_p.add_argument("--lesson", required=True, help="Lesson/chapter, e.g. 'Kapitel 3'")
    ingest_p.add_argument("--pages", help="1-indexed inclusive page range, e.g. '10-25'. Omit for the whole PDF.")

    query_p = sub.add_parser("query", help="Test retrieval against the knowledge base")
    query_p.add_argument("text", help="Query text")
    query_p.add_argument("-k", type=int, default=3)
    query_p.add_argument("--language", default=None)
    query_p.add_argument("--level", default=None)

    for p in (ingest_p, query_p):
        p.add_argument("--embed-host", default=EMBED_HOST)
        p.add_argument("--embed-port", default=EMBED_PORT)
        p.add_argument("--debug", action="store_true", help="Enable debug-level logging to logs/rag_engine.log")

    args = parser.parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    set_embed_server_target(args.embed_host, args.embed_port)

    if not check_embed_server():
        print(f"Could not reach embedding server at {_embed_base_url()}")
        sys.exit(1)

    if args.command == "ingest":
        page_start = page_end = None
        if args.pages:
            page_start, page_end = (int(x) for x in args.pages.split("-"))
        n = ingest_pdf(args.pdf, args.book, args.language, args.level, args.lesson, page_start, page_end)
        print(f"Ingested {n} chunks.")
    elif args.command == "query":
        chunks = retrieve_context(args.text, k=args.k, language=args.language, level=args.level)
        if not chunks:
            print("(no results)")
        for chunk in chunks:
            print(f"- {chunk}\n")

if __name__ == "__main__":
    main()
