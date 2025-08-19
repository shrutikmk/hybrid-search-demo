#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arxiv RAG Preprocessing Pipeline — VERBOSE
- Spark-parallelized scraping (CPU) up to N papers
- GPU (MPS) semantic chunking via Chonkie + all-MiniLM-L6-v2
- Caching + resume
- Saves papers.pkl and chunks.pkl
- EXTREMELY CHATTY: timestamps for all major steps

Usage (Mac M2, local Spark):
    nohup python /Users/shrutikmk/Documents/90DAYS/hybrid-search-demo/src/arxiv-pipeline.py \
      --query "cat:cs.CL" \
      --max-results 1000 \
      --out-dir data/processed \
      --raw-dir data/raw/arxiv \
      --partitions 8 \
      --mps-auto \
      > arxiv_pipeline.out 2>&1 &
"""

import argparse
import os
import sys
import time
import traceback
import signal
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import re
import requests
import pandas as pd
import fitz  # PyMuPDF
import arxiv
from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer  # noqa: F401 (kept for clarity)

# ---- Chonkie load early
try:
    import importlib
    chonkie = importlib.import_module("chonkie")
except Exception as e:
    print("FATAL: Chonkie is required. Install with `pip install chonkie`.", file=sys.stderr)
    raise

ARXIV_REQUEST_DELAY_S_DEFAULT = 3.0  # be polite


# -----------------------------------
# Verbose logging helpers
# -----------------------------------
def ts() -> str:
    """UTC timestamp string."""
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

def log(msg: str) -> None:
    print(f"[{ts()}] {msg}", flush=True)

def human_bytes(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


# -----------------------------------
# Data structures / config
# -----------------------------------
@dataclass
class PaperMeta:
    paper_id: str
    title: str
    authors: str
    abstract: str
    primary_category: str
    categories: str
    published: Any
    updated: Any
    pdf_url: str
    entry_id: str
    source: str = "arxiv"


# -----------------------------------
# Utilities
# -----------------------------------
def ensure_dirs(*paths: Path):
    for p in paths:
        if not p.exists():
            log(f"Creating directory: {p}")
        p.mkdir(parents=True, exist_ok=True)

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name)

def download_pdf(pdf_url: str, out_path: Path, headers: Optional[dict] = None, retry: int = 3, backoff: float = 2.0):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retry):
        try:
            log(f"[download] GET {pdf_url} -> {out_path} (attempt {attempt+1}/{retry})")
            with requests.get(pdf_url, stream=True, headers=headers, timeout=60) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            size = out_path.stat().st_size if out_path.exists() else 0
            log(f"[download] OK {out_path} ({human_bytes(size)})")
            return
        except Exception as e:
            log(f"[download] ERROR {e.__class__.__name__}: {e}")
            if attempt == retry - 1:
                log(f"[download] Giving up on {pdf_url}")
                raise
            sleep_s = backoff * (attempt + 1)
            log(f"[download] Retrying in {sleep_s:.1f}s...")
            time.sleep(sleep_s)

def extract_pdf_text(pdf_path: Path) -> str:
    log(f"[extract] Opening PDF: {pdf_path}")
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        txt = page.get_text("text")
        pages.append(txt)
        if i % 10 == 0:
            log(f"[extract] Read page {i+1}/{len(doc)}...")
    full_text = "\n".join(pages)
    log(f"[extract] Extracted text from {len(doc)} pages ({human_bytes(len(full_text.encode('utf-8')))})")
    return full_text

def pick_device(mps_auto: bool, force_cpu: bool) -> str:
    if force_cpu:
        log("[device] Forcing CPU per flag")
        return "cpu"
    if mps_auto:
        try:
            import torch
            if torch.backends.mps.is_available() and torch.backends.mps.is_built():
                log("[device] Using Apple Metal (MPS)")
                return "mps"
            else:
                log("[device] MPS not available/built; falling back to CPU")
        except Exception as e:
            log(f"[device] Torch/MPS check failed: {e}; falling back to CPU")
    return "cpu"

def build_semantic_chunker(device: str, hf_token: Optional[str]) -> Any:
    # Only pass kwargs SentenceTransformer understands.
    st_kwargs: Dict[str, Any] = {"device": device}
    if hf_token:
        st_kwargs["use_auth_token"] = hf_token

    log(f"[chunker] Building SemanticChunker (device={device})")
    t0 = time.time()
    if hasattr(chonkie, "SemanticChunker"):
        sc = chonkie.SemanticChunker(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            **st_kwargs
        )
    elif hasattr(chonkie, "semantic") and hasattr(chonkie.semantic, "SemanticChunker"):
        sc = chonkie.semantic.SemanticChunker(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            **st_kwargs
        )
    else:
        raise ImportError("Could not locate Chonkie's SemanticChunker in this version.")
    log(f"[chunker] Ready in {time.time() - t0:.2f}s")
    return sc

def semantic_chunk_text(sc: Any, full_text: str) -> List[Dict[str, Any]]:
    log(f"[chunk] Begin semantic chunking (chars={len(full_text):,})")
    t0 = time.time()
    if hasattr(sc, "chunk_text") and callable(sc.chunk_text):
        pieces = sc.chunk_text(full_text)
    elif hasattr(sc, "split") and callable(sc.split):
        pieces = sc.split(full_text)
    elif hasattr(sc, "chunk") and callable(sc.chunk):
        pieces = sc.chunk(full_text)
    else:
        raise AttributeError("SemanticChunker has no method chunk_text/split/chunk.")
    dt = time.time() - t0
    log(f"[chunk] Produced {len(pieces)} chunks in {dt:.2f}s")

    out = []
    for i, p in enumerate(pieces):
        if isinstance(p, dict):
            text = p.get("text") or p.get("chunk") or ""
            title = p.get("title") or p.get("section_title") or f"Chunk {i}"
            sp = p.get("start_page")
            ep = p.get("end_page")
        else:
            text = getattr(p, "text", str(p))
            title = getattr(p, "title", f"Chunk {i}")
            sp = getattr(p, "start_page", None)
            ep = getattr(p, "end_page", None)
        out.append({"section_title": title, "text": text, "start_page": sp, "end_page": ep})
        if i % 20 == 0:
            log(f"[chunk] Sample chunk {i}: title='{title[:60]}', len={len(text)}")
    return out


# -----------------------------------
# Spark partition worker
# -----------------------------------
def partition_worker(iterable: Iterable[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    """
    Each partition:
    - Downloads PDF (if not cached)
    - Extracts text (if not cached)
    - Returns a dict for each paper with metadata + raw_text
    Chunking happens on driver (GPU/CPU).
    """
    session = requests.Session()
    headers = {'User-Agent': 'Arxiv-RAG-Preproc/0.1 (contact: pipeline@example.com)'}
    results = []
    for item in iterable:
        paper_id = item.get("paper_id", "UNKNOWN")
        try:
            pdf_url = item["pdf_url"]
            pdf_path = Path(item["pdf_path"])
            txt_path = Path(item["txt_path"])
            delay = float(item["delay"])

            # PDF cache
            if not pdf_path.exists():
                print(f"[{ts()}] [worker] {paper_id}: downloading PDF...")
                download_pdf(pdf_url, pdf_path, headers=headers)
                time.sleep(delay)  # be polite per request
            else:
                print(f"[{ts()}] [worker] {paper_id}: PDF cache hit ({pdf_path})")

            # Text cache
            if not txt_path.exists():
                print(f"[{ts()}] [worker] {paper_id}: extracting text...")
                raw_text = extract_pdf_text(pdf_path)
                txt_path.write_text(raw_text, encoding="utf-8")
                print(f"[{ts()}] [worker] {paper_id}: wrote text cache ({txt_path})")
            else:
                raw_text = txt_path.read_text(encoding="utf-8")
                print(f"[{ts()}] [worker] {paper_id}: text cache hit ({txt_path}, {human_bytes(len(raw_text.encode('utf-8')))})")

            meta = item["meta"]
            meta["raw_text"] = raw_text
            results.append(meta)
            print(f"[{ts()}] [worker] {paper_id}: DONE")
        except Exception as e:
            print(f"[{ts()}] [worker] {paper_id}: ERROR {e.__class__.__name__}: {e}", file=sys.stderr)
            traceback.print_exc()
            continue
    return results


# -----------------------------------
# Signal handling
# -----------------------------------
def _handle_sigterm(sig, frame):
    log("Received SIGTERM, exiting gracefully...")
    sys.exit(0)

def _handle_sigint(sig, frame):
    log("Received SIGINT (Ctrl-C), exiting gracefully...")
    sys.exit(0)


# -----------------------------------
# Main
# -----------------------------------
def main():
    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigint)

    ap = argparse.ArgumentParser(description="Arxiv RAG Preprocessing (Spark + MPS chunking) — VERBOSE")
    ap.add_argument("--query", type=str, default="cat:cs.CL", help="arXiv query (e.g., 'cat:cs.CL')")
    ap.add_argument("--max-results", type=int, default=1000, help="Max number of results to fetch")
    ap.add_argument("--out-dir", type=str, default="data/processed", help="Output directory for PKL files")
    ap.add_argument("--raw-dir", type=str, default="data/raw/arxiv", help="Raw cache directory (pdfs/texts)")
    ap.add_argument("--partitions", type=int, default=8, help="Spark parallelism")
    ap.add_argument("--delay", type=float, default=ARXIV_REQUEST_DELAY_S_DEFAULT, help="Per-request arXiv delay (sec)")
    ap.add_argument("--force-cpu", action="store_true", help="Force CPU for chunking")
    ap.add_argument("--mps-auto", action="store_true", help="Use MPS on Apple Silicon if available")
    ap.add_argument("--hf-token-env", type=str, default="HUGGINGFACE_HUB_TOKEN", help="Env var name with HF token")
    ap.add_argument("--ids-file", type=str, default="", help="Optional path to a newline-delimited list of arXiv IDs")
    args = ap.parse_args()

    t_start = time.time()
    log("========== Arxiv RAG Preprocessing — START ==========")
    log(f"Args: {args}")

    raw_dir = Path(args.raw_dir)
    pdf_dir = raw_dir / "pdfs"
    txt_dir = raw_dir / "texts"
    out_dir = Path(args.out_dir)
    ensure_dirs(raw_dir, pdf_dir, txt_dir, out_dir)

    # ----- Discover IDs / results
    if args.ids_file:
        ids = [ln.strip() for ln in Path(args.ids_file).read_text().splitlines() if ln.strip()]
        log(f"Using {len(ids)} IDs from {args.ids_file}")
        client = arxiv.Client(page_size=25, delay_seconds=args.delay, num_retries=3)
        results: List[arxiv.Result] = []
        for arx_id in ids:
            try:
                log(f"[metadata] Fetching {arx_id} ...")
                search = arxiv.Search(id_list=[arx_id])
                for r in client.results(search):
                    results.append(r)
                time.sleep(args.delay)
            except Exception as e:
                log(f"[metadata] Failed for {arx_id}: {e}")
    else:
        log(f"[metadata] Query='{args.query}', max_results={args.max_results}")
        client = arxiv.Client(page_size=100, delay_seconds=args.delay, num_retries=3)
        search = arxiv.Search(query=args.query, max_results=args.max_results,
                              sort_by=arxiv.SortCriterion.SubmittedDate)
        results = list(client.results(search))

    log(f"[metadata] Fetched {len(results)} results")

    # ----- Build job list for Spark
    job_items: List[Dict[str, Any]] = []
    for r in results:
        paper_id = r.get_short_id()
        pdf_path = pdf_dir / f"{sanitize_filename(paper_id)}.pdf"
        txt_path = txt_dir / f"{sanitize_filename(paper_id)}.txt"

        meta = PaperMeta(
            paper_id=paper_id,
            title=r.title,
            authors=", ".join(a.name for a in r.authors),
            abstract=r.summary,
            primary_category=r.primary_category,
            categories=", ".join(r.categories),
            published=r.published,
            updated=r.updated,
            pdf_url=r.pdf_url,
            entry_id=r.entry_id,
            source="arxiv",
        )
        job_items.append({
            "paper_id": paper_id,
            "pdf_url": r.pdf_url,
            "pdf_path": str(pdf_path),
            "txt_path": str(txt_path),
            "delay": args.delay,
            "meta": asdict(meta),
        })

    log(f"[plan] Built {len(job_items)} job items")

    # ----- Spark session (local)
    log("[spark] Starting local Spark session...")
    spark = SparkSession.builder \
        .appName("ArxivRAGPreprocVerbose") \
        .master("local[*]") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()
    sc = spark.sparkContext
    sc.setLogLevel("WARN")
    log("[spark] Session ready")

    # ----- Parallel scrape + extract
    rdd = sc.parallelize(job_items, numSlices=max(1, args.partitions))
    log(f"[spark] Parallelizing across {max(1, args.partitions)} partitions")
    results_meta: List[Dict[str, Any]] = rdd.mapPartitions(partition_worker).collect()
    log("[spark] Shutting down Spark...")
    spark.stop()
    log("[spark] Stopped")

    log(f"[scrape] Completed downloads/extraction for {len(results_meta)} papers")

    # ----- Build papers_df (driver)
    log("[df] Building papers_df...")
    papers_df = pd.DataFrame(results_meta)
    log(f"[df] papers_df shape: {papers_df.shape}")

    # ----- Semantic chunking (driver, uses GPU if available)
    hf_token = os.getenv(args.hf_token_env, "").strip() or None
    device = pick_device(mps_auto=args.mps_auto, force_cpu=args.force_cpu)
    log(f"[device] Chunking device selected: {device}")

    try:
        scnk = build_semantic_chunker(device=device, hf_token=hf_token)
    except Exception as e:
        log(f"[chunker] Build FAILED on {device}: {e}")
        log("[chunker] Falling back to CPU...")
        scnk = build_semantic_chunker(device="cpu", hf_token=hf_token)
        device = "cpu"

    chunk_rows: List[Dict[str, Any]] = []
    log("[chunk] Starting per-paper chunking...")
    for idx, row in enumerate(papers_df.itertuples(index=False), start=1):
        try:
            pid = getattr(row, "paper_id")
            full_text = getattr(row, "raw_text") or ""
            log(f"[chunk] ({idx}/{len(papers_df)}) {pid}: text_len={len(full_text):,}")
            if not full_text.strip():
                log(f"[chunk] ({pid}) SKIP (empty text)")
                continue
            pieces = semantic_chunk_text(scnk, full_text)
            for j, ch in enumerate(pieces):
                chunk_rows.append({
                    "paper_id": pid,
                    "chunk_id": f"{pid}::chunk_{j:04d}",
                    "section_title": ch.get("section_title"),
                    "text": ch.get("text"),
                    "start_page": ch.get("start_page"),
                    "end_page": ch.get("end_page"),
                })
        except Exception as e:
            log(f"[chunk] ({pid}) ERROR: {e}")
            traceback.print_exc()
            continue

    chunks_df = pd.DataFrame(chunk_rows)
    log(f"[df] chunks_df shape: {chunks_df.shape}")

    # ----- Save PKLs
    out_dir.mkdir(parents=True, exist_ok=True)
    papers_path = out_dir / "papers.pkl"
    chunks_path = out_dir / "chunks.pkl"
    log(f"[save] Writing {papers_path} ...")
    papers_df.to_pickle(papers_path)
    log(f"[save] Writing {chunks_path} ...")
    chunks_df.to_pickle(chunks_path)
    log(f"[save] DONE")

    # ----- Summary
    log("========== SUMMARY ==========")
    log(f"Papers processed: {len(papers_df)}")
    log(f"Total chunks:     {len(chunks_df)}")
    log(f"Output files:     {papers_path} , {chunks_path}")
    log(f"Elapsed:          {time.time() - t_start:.2f}s")
    log("========== Arxiv RAG Preprocessing — END ==========")


if __name__ == "__main__":
    main()