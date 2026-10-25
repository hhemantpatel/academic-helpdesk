#!/usr/bin/env python3
"""
ingest/build_index.py - Build the persistent ChromaDB index with nomic-embed-text.

Reads ingest/manifest.csv, extracts text per document (PDF pages, with tesseract OCR for
scanned pages; HTML; Markdown split by section), chunks it, embeds it with Ollama and
stores it in one cosine-distance collection. Idempotent; use --rebuild to re-index.
"""

import argparse
import csv
import html
import re
import subprocess
import sys
import time
from pathlib import Path

import chromadb
import ollama
import pymupdf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from helpdesk.config import (  # noqa: E402
    CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, OLLAMA_API_BASE, RAW_DIR,
)

MANIFEST_FILE = Path(__file__).resolve().parent / "manifest.csv"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 32
ZERO_WIDTH = re.compile("[​‌‍﻿]")


def clean_html(raw_html: str) -> str:
    text = re.sub(r"(?is)<(script|style|nav|header|footer)\b.*?</\1>", " ", raw_html)
    text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h[1-6])>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = ZERO_WIDTH.sub("", html.unescape(text))
    lines = [re.sub(r"[ \t]+", " ", l).strip() for l in text.splitlines()]
    return "\n".join(l for l in lines if l)


def ocr_page(page) -> str:
    """OCR a scanned PDF page with the tesseract CLI (English)."""
    png = page.get_pixmap(dpi=300).tobytes("png")
    try:
        out = subprocess.run(["tesseract", "stdin", "stdout", "-l", "eng"],
                             input=png, capture_output=True, check=True, timeout=120)
    except (FileNotFoundError, subprocess.SubprocessError) as e:
        print(f"  OCR failed: {e}")
        return ""
    return out.stdout.decode("utf-8", errors="ignore")


def chunk_text(text: str, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Prefer to break at a paragraph, then a sentence, then a line, then a space.
        break_pt = -1
        for sep, offset in (("\n\n", 0), (". ", 1), ("\n", 0), (" ", 0)):
            idx = text.rfind(sep, start + chunk_size // 2, end)
            if idx != -1:
                break_pt = idx + offset
                break
        if break_pt == -1:
            break_pt = end
        chunk = text[start:break_pt].strip()
        if chunk:
            chunks.append(chunk)
        start = max(start + 1, break_pt - overlap)
    return [c for c in chunks if c]


def _chunk_record(doc_id, idx, text, title, page, section, date, doc_type):
    return {
        "id": f"{doc_id}_p{page}_c{idx}",
        "text": text,
        "metadata": {"doc_id": doc_id, "title": title, "page": page,
                     "section": section, "date": date, "type": doc_type},
    }


def load_and_chunk_documents() -> list[dict]:
    with open(MANIFEST_FILE, encoding="utf-8") as f:
        manifest = list(csv.DictReader(f))
    print(f"Loaded manifest with {len(manifest)} documents.")

    chunks = []
    for row in manifest:
        doc_id, title = row["doc_id"].strip(), row["title"].strip()
        doc_type, date = row["type"].strip().upper(), row.get("date", "").strip()
        path = RAW_DIR / row["file"].strip()
        if not path.exists():
            print(f"Warning: file not found: {path}, skipping.")
            continue

        n_before = len(chunks)
        if doc_type == "PDF":
            with pymupdf.open(path) as doc:
                for page_num, page in enumerate(doc, start=1):
                    text = page.get_text("text").strip()
                    if len(text) < 20:  # scanned page: no text layer
                        text = ocr_page(page).strip()
                    for i, c in enumerate(chunk_text(text)):
                        chunks.append(_chunk_record(doc_id, i, c, title, page_num, f"Page {page_num}", date, "PDF"))
        elif doc_type == "HTML":
            text = clean_html(path.read_text(encoding="utf-8", errors="ignore"))
            for i, c in enumerate(chunk_text(text)):
                chunks.append(_chunk_record(doc_id, i, c, title, 1, "General", date, "HTML"))
        elif doc_type in ("MD", "TEXT"):
            counter = 0
            for sec in re.split(r"(?m)^(?=##?\s+)", path.read_text(encoding="utf-8")):
                sec = sec.strip()
                if not sec:
                    continue
                header = re.match(r"^##?\s+(.*)", sec)
                sec_title = header.group(1).strip() if header else "Section"
                for c in chunk_text(sec):
                    chunks.append(_chunk_record(doc_id, counter, c, title, 1, sec_title, date, "MD"))
                    counter += 1
        else:
            print(f"Warning: unknown type {doc_type} for {doc_id}, skipping.")

        print(f"  {doc_id}: {len(chunks) - n_before} chunks")
        if len(chunks) == n_before:
            print(f"  WARNING: {doc_id} produced no text")

    print(f"Total chunks extracted: {len(chunks)}")
    return chunks


def build_index(rebuild: bool = False):
    client = ollama.Client(host=OLLAMA_API_BASE)
    chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if rebuild:
        try:
            chroma.delete_collection(name=COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'.")
        except Exception:
            pass

    collection = chroma.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    if collection.count() > 0 and not rebuild:
        print(f"Collection '{COLLECTION_NAME}' already has {collection.count()} chunks. Use --rebuild to re-index.")
        return collection

    chunks = load_and_chunk_documents()
    if not chunks:
        print("No chunks to index!")
        return collection

    print(f"Embedding {len(chunks)} chunks with {EMBEDDING_MODEL}...")
    t0 = time.time()
    for b in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[b:b + EMBED_BATCH]
        emb = client.embed(model=EMBEDDING_MODEL, input=[f"search_document: {c['text']}" for c in batch])
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
            embeddings=emb["embeddings"],
        )
        print(f"Indexed {min(b + EMBED_BATCH, len(chunks))}/{len(chunks)} chunks...")
    print(f"Indexed {collection.count()} chunks in {time.time() - t0:.1f}s.")
    return collection


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the persistent vector store for helpdesk documents")
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the vector store from scratch")
    build_index(rebuild=parser.parse_args().rebuild)
