#!/usr/bin/env python3
"""
Fetch real ELOG entries and serialize them as mock data.

Supports:
- Full crawl (all IDs 1..last) with --all or an ID range with --ids-from/--ids-to
- Streaming output to JSONL for low memory usage
- Periodic checkpoints for fault-tolerant resume
- Optional materialization to a single JSON file and/or a Python module

Outputs (depending on flags):
- mock_elog_data.py     -> Python module with variable `MOCK_ELOG = [(html, meta, attachments), ...]`
- mock_elog_data.json   -> JSON array of [html, meta, attachments]
- mock_elog_data.jsonl  -> One JSON object per line: {"html": ..., "meta": ..., "attachments": ...}

Examples:
    # Crawl everything once, stream to JSONL (recommended for ~39k)
    python fetch_elog_all.py --all --out-jsonl mock_elog_data.jsonl --checkpoint crawl.ckpt

    # Later, materialize a single JSON and Python module from the JSONL:
    python fetch_elog_all.py --materialize-json --in-jsonl mock_elog_data.jsonl \
        --out-json mock_elog_data.json --out-py mock_elog_data.py
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from email.utils import parsedate_to_datetime

import warnings
warnings.filterwarnings("ignore")

try:
    import elog  # pip install python-elog   (or your org's elog client)
except Exception as e:
    print("ERROR: Could not import 'elog' library. Install your ELOG client first.")
    raise

# ------------------------ Defaults ------------------------

DEFAULT_URL = "https://elog-gfa.psi.ch/SwissFEL+commissioning/"

DEFAULT_QUERIES = [
    "DRM reset",
    "DRM limit",
    "DRM alarm",
    "emittance",
    "measurement summary",
    "shift summary",
]

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff"}
IMAGE_MIMES = {"image/png", "image/jpeg", "image/gif", "image/bmp", "image/tiff"}

# ------------------------ Helpers -------------------------

def is_recent_enough(meta: Dict[str, Any], since_ts: Optional[float], until_ts: Optional[float]) -> bool:
    """Filter by Date (RFC 2822) or When (epoch seconds)."""
    ts = None
    date_str = meta.get("Date")
    if date_str:
        try:
            dt = parsedate_to_datetime(date_str)
            ts = dt.timestamp()
        except Exception:
            ts = None
    if ts is None:
        when = meta.get("When")
        try:
            ts = float(when)
        except Exception:
            ts = None

    if ts is None:
        return True  # no timestamp -> keep

    if since_ts is not None and ts < since_ts:
        return False
    if until_ts is not None and ts > until_ts:
        return False
    return True


def safe_filename(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name.strip())
    return name[:200] or "file"


def attachment_is_image(att: Dict[str, Any]) -> bool:
    """Best-effort check if the attachment is an image by mime or filename."""
    mime = (att.get("mime") or att.get("content_type") or "").lower()
    if mime in IMAGE_MIMES:
        return True
    fn = (att.get("filename") or att.get("name") or att.get("file") or "")
    ext = Path(fn).suffix.lower()
    if ext in IMAGE_EXT:
        return True
    return False


def write_python_module(entries: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Auto-generated from ELOG; shareable mock tuples\n")
        f.write("MOCK_ELOG = [\n")
        for html, meta, atts in entries:
            f.write("  (\n")
            f.write(f"    {html!r},\n")
            f.write(f"    {json.dumps(meta, ensure_ascii=False)},\n")
            f.write(f"    {json.dumps(atts, ensure_ascii=False)}\n")
            f.write("  ),\n")
        f.write("]\n")


def open_elog(url: str, user: Optional[str], password: Optional[str]):
    """Open ELOG with your installed client."""
    kwargs = {}
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password
    return elog.open(url, **kwargs)


def fetch_ids_for_query(client, query: str, scope: str, n_results: int) -> List[str]:
    try:
        return list(client.search(query, scope=scope, n_results=n_results))
    except TypeError:
        # Older clients may not accept named args
        return list(client.search(query, scope, n_results))


def read_entry(client, entry_id: str) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Normalize read() output to (html, meta, attachments)."""
    data = client.read(entry_id)
    if not isinstance(data, (tuple, list)) or len(data) < 2:
        raise ValueError(f"Unexpected ELOG read() response for id {entry_id}: {type(data)}")
    html = data[0]
    meta = data[1] or {}
    attachments = data[2] if len(data) > 2 else []

    # Normalize attachments to list of dicts
    norm_atts = []
    for a in attachments or []:
        if isinstance(a, dict):
            norm_atts.append(a)
        else:
            # try to coerce tuples/lists -> dict
            try:
                # Common shapes: (filename, url) or (filename, data, mime)
                if isinstance(a, (tuple, list)):
                    d = {}
                    if len(a) >= 1: d["filename"] = a[0]
                    if len(a) >= 2 and isinstance(a[1], (str, bytes)):
                        if isinstance(a[1], str) and a[1].startswith("http"):
                            d["url"] = a[1]
                        else:
                            d["data"] = a[1]
                    if len(a) >= 3: d["mime"] = a[2]
                    norm_atts.append(d)
                else:
                    norm_atts.append({"value": str(a)})
            except Exception:
                norm_atts.append({"value": str(a)})
    return html, meta, norm_atts


def maybe_download_attachments(client, entry_id: str, attachments: List[Dict[str, Any]], out_dir: Path) -> List[Dict[str, Any]]:
    """Download image attachments where possible; keep originals otherwise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    session = getattr(client, "session", None)  # some clients expose an HTTP session

    rewritten: List[Dict[str, Any]] = []
    for idx, att in enumerate(attachments):
        try:
            if not attachment_is_image(att):
                rewritten.append(att)
                continue

            fn = safe_filename(att.get("filename") or att.get("name") or f"{entry_id}_{idx}.bin")
            mime = att.get("mime") or att.get("content_type")
            target = out_dir / fn

            if "data" in att and isinstance(att["data"], (bytes, bytearray)):
                target.write_bytes(att["data"])
                rewritten.append({"filename": fn, "path": str(target), "mime": mime or ""})
                continue

            url = att.get("url")
            if url and session is not None:
                try:
                    r = session.get(url, timeout=30)
                    r.raise_for_status()
                    target.write_bytes(r.content)
                    if not mime:
                        mime = r.headers.get("Content-Type", "")
                    rewritten.append({"filename": fn, "path": str(target), "mime": (mime or "")})
                    continue
                except Exception:
                    pass

            rewritten.append(att)
        except Exception:
            rewritten.append(att)

    return rewritten


def resilient_read(client, entry_id: int, retries: int = 3, backoff: float = 0.5):
    """Read with retries and simple backoff. Returns (html, meta, atts) or raises."""
    last_exc = None
    for i in range(retries):
        try:
            return read_entry(client, str(entry_id))
        except Exception as e:
            last_exc = e
            time.sleep(backoff * (2 ** i))
    raise last_exc


def load_checkpoint(path: Optional[Path]) -> Optional[int]:
    if not path or not path.exists():
        return None
    try:
        return int(path.read_text().strip())
    except Exception:
        return None


def save_checkpoint(path: Optional[Path], last_id: int):
    if not path:
        return
    path.write_text(str(last_id), encoding="utf-8")


def stream_jsonl_write(jsonl_path: Path, obj: Dict[str, Any]):
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def materialize_from_jsonl(in_jsonl: Path) -> List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]]:
    out: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    with in_jsonl.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            out.append((rec["html"], rec["meta"], rec["attachments"]))
    return out


# ------------------------ Main ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Create mock ELOG data from the real ELOG.")

    # Auth / base
    ap.add_argument("--url", default=DEFAULT_URL, help="ELOG base URL")
    ap.add_argument("--user", default=None, help="Username (if required)")
    ap.add_argument("--password", default=None, help="Password (if required)")

    # Modes
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="Fetch ALL entries 1..get_last_message_id()")
    mode.add_argument("--queries", nargs="*", default=None, help="Search queries (BM25/regex via server)")
    ap.add_argument("--scope", default="subtext",
                    choices=["title", "subtext", "author", "tags"], help="Search scope (when using --queries)")
    ap.add_argument("--per-query", type=int, default=10, help="Max results per query (when using --queries)")

    # Ranging / limits
    ap.add_argument("--ids-from", type=int, default=None, help="Start ID (inclusive) when iterating IDs")
    ap.add_argument("--ids-to", type=int, default=None, help="End ID (inclusive) when iterating IDs")
    ap.add_argument("--limit", type=int, default=None, help="Stop after N collected entries (default: no limit)")
    ap.add_argument("--sleep", type=float, default=0.05, help="Sleep seconds between reads to be polite")
    ap.add_argument("--retries", type=int, default=3, help="Retries per entry on read failure")
    ap.add_argument("--backoff", type=float, default=0.4, help="Exponential backoff base (seconds)")

    # Time filter (still supported)
    ap.add_argument("--since", default=None, help="ISO datetime lower bound (e.g. 2025-05-01T00:00:00+02:00)")
    ap.add_argument("--until", default=None, help="ISO datetime upper bound")

    # Attachments
    ap.add_argument("--download-attachments", action="store_true", help="Attempt to download image attachments")
    ap.add_argument("--attachments-dir", default="mock_attachments", help="Directory to save attachments")

    # Outputs
    ap.add_argument("--out-py", default="mock_elog_data.py", help="Output Python module")
    ap.add_argument("--out-json", default="mock_elog_data.json", help="Output JSON file")
    ap.add_argument("--out-jsonl", default=None, help="Stream output JSONL file (recommended for large crawls)")
    ap.add_argument("--flush-every", type=int, default=500,
                    help="When not streaming, flush to disk every N entries (to keep memory bounded)")

    # Resume
    ap.add_argument("--checkpoint", default=None, help="Path to a checkpoint file for last processed ID")
    ap.add_argument("--resume", action="store_true", help="Resume from checkpoint if present")

    # Materialization mode (no crawling)
    ap.add_argument("--materialize-json", action="store_true",
                    help="Build --out-json/--out-py from an existing --in-jsonl (no crawling)")
    ap.add_argument("--in-jsonl", default=None, help="Source JSONL for --materialize-json")

    args = ap.parse_args()

    # -------- Materialization path (no crawling) --------
    if args.materialize_json:
        if not args.in_jsonl:
            print("ERROR: --materialize-json requires --in-jsonl")
            sys.exit(2)
        in_jsonl = Path(args.in_jsonl)
        if not in_jsonl.exists():
            print(f"ERROR: --in-jsonl not found: {in_jsonl}")
            sys.exit(2)
        entries = materialize_from_jsonl(in_jsonl)
        # Write JSON
        if args.out_json:
            Path(args.out_json).write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✔ Wrote JSON: {args.out_json} ({len(entries)} entries)")
        # Write Python module
        if args.out_py:
            write_python_module(entries, Path(args.out_py))
            print(f"✔ Wrote Python module: {args.out_py}")
        return

    # -------- Normal crawling --------
    # Parse time window (optional)
    since_ts = None
    until_ts = None
    if args.since:
        try:
            since_ts = parsedate_to_datetime(args.since).timestamp() if "," in args.since else \
                       __import__("dateutil.parser").parser.parse(args.since).timestamp()
        except Exception:
            print(f"WARNING: could not parse --since={args.since}; ignoring")
    if args.until:
        try:
            until_ts = parsedate_to_datetime(args.until).timestamp() if "," in args.until else \
                       __import__("dateutil.parser").parser.parse(args.until).timestamp()
        except Exception:
            print(f"WARNING: could not parse --until={args.until}; ignoring")

    # Open client
    client = open_elog(args.url, args.user, args.password)

    # Decide ID range
    start_id = args.ids_from
    end_id = args.ids_to
    if args.all or (start_id is None and end_id is None and not args.queries):
        # Use server's last ID
        try:
            last_id = int(client.get_last_message_id())
        except Exception:
            # Fallback: try get_message_ids if available
            try:
                ids = list(client.get_message_ids())
                last_id = max(int(x) for x in ids)
            except Exception as e:
                print("ERROR: Could not determine last message id; provide --ids-from/--ids-to.")
                raise
        start_id = 1 if start_id is None else start_id
        end_id = last_id if end_id is None else end_id

    # Checkpoint
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if args.resume and ckpt_path and ckpt_path.exists():
        ckpt_value = load_checkpoint(ckpt_path)
        if ckpt_value is not None:
            # resume from next after checkpoint
            if start_id is None or ckpt_value + 1 > start_id:
                start_id = ckpt_value + 1
            print(f"↻ Resuming from checkpoint id {ckpt_value} -> start at {start_id}")

    # Prepare outputs
    jsonl_path = Path(args.out_jsonl) if args.out_jsonl else None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        # If not resuming, truncate existing JSONL
        if not args.resume and jsonl_path.exists():
            jsonl_path.unlink()

    py_path = Path(args.out_py) if args.out_py else None
    json_path = Path(args.out_json) if args.out_json else None

    # In-memory buffer (only used if not streaming)
    buffer: List[Tuple[str, Dict[str, Any], List[Dict[str, Any]]]] = []
    total_written = 0

    def maybe_flush():
        nonlocal buffer, total_written
        if not buffer:
            return
        # Append to JSONL if requested
        if jsonl_path:
            for (html, meta, atts) in buffer:
                stream_jsonl_write(jsonl_path, {"html": html, "meta": meta, "attachments": atts})
        # Append/maintain rolling JSON file (expensive for huge sets, optional)
        if json_path:
            # read-append-write pattern would be costly; instead write at the very end unless small
            pass
        total_written += len(buffer)
        buffer = []

    collected = 0
    limit = args.limit if args.limit and args.limit > 0 else None

    # -------- Crawl by queries (legacy mode) --------
    if args.queries:
        seen: set[str] = set()
        for q in (args.queries or DEFAULT_QUERIES):
            ids = fetch_ids_for_query(client, q, scope=args.scope, n_results=args.per_query)
            for eid in ids:
                if eid in seen:
                    continue
                try:
                    html, meta, atts = resilient_read(client, int(eid), retries=args.retries, backoff=args.backoff)
                except Exception as e:
                    print(f"READ ERROR id={eid}: {e}")
                    continue

                if not is_recent_enough(meta, since_ts, until_ts):
                    continue

                if args.download_attachments and atts:
                    try:
                        atts = maybe_download_attachments(client, eid, atts,
                                                          Path(args.attachments_dir) / str(eid))  # type: ignore
                    except Exception as e:
                        print(f"ATTACH WARN id={eid}: {e}")

                # Stream or buffer
                if jsonl_path:
                    stream_jsonl_write(jsonl_path, {"html": html, "meta": meta, "attachments": atts})
                    total_written += 1
                else:
                    buffer.append((html, meta, atts))
                    if len(buffer) >= args.flush_every:
                        maybe_flush()

                seen.add(eid)
                collected += 1
                if limit and collected >= limit:
                    break
                save_checkpoint(ckpt_path, int(eid))
                time.sleep(args.sleep)
            if limit and collected >= limit:
                break

    # -------- Crawl by ID range (recommended for all) --------
    else:
        if start_id is None or end_id is None:
            print("ERROR: No crawl mode selected. Use --all or provide --ids-from/--ids-to or --queries.")
            sys.exit(2)

        print(f"→ Crawling IDs {start_id}..{end_id} (inclusive)")
        for eid in range(int(start_id), int(end_id) + 1):
            try:
                html, meta, atts = resilient_read(client, eid, retries=args.retries, backoff=args.backoff)
            except Exception as e:
                # Some IDs may be missing or private; just skip
                if (eid % 1000) == 0:
                    print(f"… up to {eid}, skipped recent due to error: {e}")
                save_checkpoint(ckpt_path, eid)
                time.sleep(args.sleep)
                continue

            if not is_recent_enough(meta, since_ts, until_ts):
                save_checkpoint(ckpt_path, eid)
                time.sleep(args.sleep)
                continue

            if args.download_attachments and atts:
                try:
                    atts = maybe_download_attachments(client, eid, atts,
                                                      Path(args.attachments_dir) / str(eid))
                except Exception as e:
                    print(f"ATTACH WARN id={eid}: {e}")

            # Stream or buffer
            if jsonl_path:
                stream_jsonl_write(jsonl_path, {"html": html, "meta": meta, "attachments": atts})
                total_written += 1
            else:
                buffer.append((html, meta, atts))
                if len(buffer) >= args.flush_every:
                    maybe_flush()

            collected += 1
            save_checkpoint(ckpt_path, eid)

            if limit and collected >= limit:
                break

            # lightweight progress pulse
            if (eid % 1000) == 0:
                print(f"… reached id {eid}, written so far: {total_written if jsonl_path else (total_written + len(buffer))}")

            time.sleep(args.sleep)

    # Final flush
    maybe_flush()

    # Write final JSON / Python module if requested and not streaming
    if not jsonl_path:
        if json_path:
            json_path.write_text(json.dumps(buffer, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"✔ Wrote JSON: {json_path} ({len(buffer)} entries)")
        if py_path:
            write_python_module(buffer, py_path)
            print(f"✔ Wrote Python module: {py_path}")

    # If streaming JSONL and user also wants a consolidated JSON/PY, suggest materialization step
    print(f"✔ Done. Entries written: {total_written if jsonl_path else (total_written + len(buffer))}")
    if jsonl_path and (args.out_json or args.out_py):
        print("ℹ To build a single JSON/PY from JSONL, run:")
        print(f"   python {Path(sys.argv[0]).name} --materialize-json --in-jsonl {jsonl_path}"
              f" --out-json {args.out_json} --out-py {args.out_py}")


if __name__ == "__main__":
    main()


# if script gets stuck:
# python fetch_elog_entries.py --all --out-jsonl mock_elog_data.jsonl --checkpoint crawl.ckpt --resume