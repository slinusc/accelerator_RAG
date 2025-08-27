#!/usr/bin/env python3
import argparse, json, re, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple, Any, Optional

from dateutil import parser as dtp
from elasticsearch import Elasticsearch, helpers

def strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "").strip()

def to_iso(meta: dict) -> str:
    # prefer RFC-2822 "Date", fallback to epoch "When"
    d = meta.get("Date")
    if d:
        try:
            dt = dtp.parse(d)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            pass
    w = meta.get("When")
    if w:
        try:
            ts = float(w)
            return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        except Exception:
            pass
    return datetime.now(timezone.utc).isoformat()

def normalize_attachment(att: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(att, dict):
        out["filename"] = att.get("filename") or att.get("name") or att.get("file")
        out["mime"]     = att.get("mime") or att.get("content_type")
        out["caption"]  = att.get("caption") or ""
        out["url"]      = att.get("url")
        out["path"]     = att.get("path")
    elif isinstance(att, (list, tuple)) and att:
        out["filename"] = att[0]
        if len(att) > 2: out["mime"] = att[2]
    return {k: v for k, v in out.items() if v}

def record_to_doc(rec: Any, default_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Accepts either:
      - triplet [html, meta, attachments]
      - dict {"html":..., "meta":..., "attachments":[...]}
    Returns (_id, _source)
    """
    if isinstance(rec, (list, tuple)) and len(rec) >= 2:
        html, meta = rec[0], rec[1]
        atts = rec[2] if len(rec) > 2 else []
    elif isinstance(rec, dict) and "html" in rec and "meta" in rec:
        html, meta = rec["html"], rec["meta"]
        atts = rec.get("attachments", [])
    else:
        raise ValueError("Unsupported record shape")

    doc_id = str(meta.get("$@MID@$") or meta.get("id") or default_id)
    doc = {
        "doc_id": doc_id,
        "date": to_iso(meta),
        "title": meta.get("Title", ""),
        "body": strip_html(html),
        "html": html,
        "category": meta.get("Category", ""),
        "domain": meta.get("Domain", ""),
        "system": meta.get("System", ""),
        "section": meta.get("Section", ""),
        "author": meta.get("Author", ""),
        "attachments": [normalize_attachment(a) for a in (atts or []) if a],
        "_raw_meta": meta,  # keep raw meta for debugging (optional, remove if not needed)
    }
    return doc_id, doc

def iter_json_array(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for rec in data:
        yield rec

def iter_jsonl(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: 
                continue
            yield json.loads(ln)

def action_stream(records: Iterable[Any], index: str) -> Iterator[Dict[str, Any]]:
    for i, rec in enumerate(records):
        try:
            _id, _src = record_to_doc(rec, default_id=str(i))
        except Exception as e:
            print(f"[WARN] Skip record {i}: {e}", file=sys.stderr)
            continue
        yield {"_op_type": "index", "_index": index, "_id": _id, "_source": _src}

def ensure_index(es: Elasticsearch, index: str, create_if_missing: bool = True):
    if es.indices.exists(index=index):
        return
    if not create_if_missing:
        return
    # minimal mapping (tune as you like)
    body = {
        "mappings": {
            "properties": {
                "doc_id": {"type": "keyword"},
                "date": {"type": "date"},
                "title": {"type": "text"},
                "body": {"type": "text"},
                "html": {"type": "text"},
                "category": {"type": "keyword"},
                "domain": {"type": "keyword"},
                "system": {"type": "keyword"},
                "section": {"type": "keyword"},
                "author": {"type": "keyword"},
                "attachments": {
                    "type": "nested",
                    "properties": {
                        "filename": {"type": "keyword"},
                        "mime": {"type": "keyword"},
                        "caption": {"type": "text"},
                        "url": {"type": "keyword"},
                        "path": {"type": "keyword"},
                    },
                },
            }
        }
    }
    es.indices.create(index=index, body=body)

def main():
    ap = argparse.ArgumentParser(description="Ingest ELOG mock data (JSON or JSONL) into Elasticsearch")
    ap.add_argument("--input", required=True, help="Path to .json (array) or .jsonl (one JSON per line)")
    ap.add_argument("--index", default="elog", help="Elasticsearch index name (default: elog)")
    ap.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Bulk chunk size")
    ap.add_argument("--request-timeout", type=int, default=120, help="Elasticsearch request timeout (s)")
    ap.add_argument("--create-index", action="store_true", help="Create index with basic mapping if missing")
    args = ap.parse_args()

    es = Elasticsearch(args.es_url, request_timeout=args.request_timeout)
    if not es.ping():
        print("ERROR: Cannot reach Elasticsearch at", args.es_url, file=sys.stderr)
        sys.exit(2)

    if args.create_index:
        ensure_index(es, args.index, create_if_missing=True)

    path = Path(args.input)
    if not path.exists():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        sys.exit(2)

    # Choose iterator based on extension
    ext = path.suffix.lower()
    if ext == ".jsonl":
        records = iter_jsonl(path)
    elif ext == ".json":
        records = iter_json_array(path)
    else:
        # try JSONL first if unknown
        try:
            records = iter_jsonl(path)
            # Touch the iterator by peeking one item
            first = next(records)
            # re-chain with the first item
            def _chain():
                yield first
                for r in iter_jsonl(path):
                    yield r
            records = _chain()
        except Exception:
            records = iter_json_array(path)

    # Stream with helpers.streaming_bulk to avoid large memory usage
    ok = 0
    fail = 0
    for success, info in helpers.streaming_bulk(
        es,
        action_stream(records, args.index),
        chunk_size=args.chunk_size,
        raise_on_error=False,
        max_retries=3,
    ):
        if success:
            ok += 1
        else:
            fail += 1
            print(f"[ERR] {info}", file=sys.stderr)

    print(f"Ingest finished. ok={ok}, fail={fail}, index='{args.index}'")

if __name__ == "__main__":
    main()

## Example usage for JSONL
# python ingest_elog_es.py --input full_elog_data.jsonl --index elog --create-index

## Example usage for JSON array
# python ingest_elog_es.py --input mock_elog_data.json --index elog --create-index
