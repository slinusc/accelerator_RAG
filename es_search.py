"""
1. **`recent="sort"`**

* Explicit sort by `date desc` (newest → oldest).
* Deterministic: perfect for *“last/latest”* queries.
* Downside: may push highly relevant but slightly older hits below newer but less relevant ones.

2. **`recent="score"`** *(default)*

* Uses a **function_score** with time-decay (e.g. half-life 14 days).
* Balances **keyword match (BM25)** with **recency**.
* Good for most “normal” queries (find relevant evidence but weight recent more).

3. **`recent="none"`**

* Pure BM25 keyword ranking.
* No recency bias, no timestamp sorting.
* Useful if you want “most relevant ever”, regardless of time.
"""

from elasticsearch import Elasticsearch
from datetime import datetime, timedelta, timezone
from dateutil import parser as dtp

es = Elasticsearch("http://localhost:9200", request_timeout=30)

def _clean_filters(filters):
    filters = filters or {}
    out = {}
    for k in ("category","domain","system","section"):
        v = filters.get(k)
        if v is None: 
            continue
        if isinstance(v, (list, tuple)):
            v = [x for x in v if x]
            if v: out[k] = v
        else:
            if str(v).strip():
                out[k] = [v]
    return out

def _range_clause(since, until):
    if not since and not until:
        return None
    rng = {"range": {"date": {}}}
    if since: rng["range"]["date"]["gte"] = since
    if until: rng["range"]["date"]["lte"] = until
    return rng

def build_query(
    user_q: str,
    k: int = 10,
    filters=None,
    since: str | None = None,
    until: str | None = None,
    recent: str = "score",          # "sort" | "score" | "none"
    track_scores: bool = True
):
    """
    recent="sort": sort by date desc (then by _score)
    recent="score": use function_score with time-decay
    recent="none": pure BM25 (no recency bias)
    """
    filters = _clean_filters(filters)
    
    # Use match_all for wildcard queries, multi_match for specific terms
    if user_q == "*" or not user_q.strip():
        must = [{"match_all": {}}]
    else:
        must = [{
            "multi_match": {
                "query": user_q,
                "fields": ["title^3","body^1.5","attachments.caption"],
                "type": "best_fields",
                "operator": "and"
            }
        }]

    fil = []
    for key, vals in filters.items():
        fil.append({"terms": {key: vals}})
    rng = _range_clause(since, until)
    if rng:
        fil.append(rng)

    base_bool = {"must": must}
    if fil:
        base_bool["filter"] = fil

    body = {
        "size": k,
        "track_total_hits": True,
        "highlight": {
            "fields": {"title": {}, "body": {}, "attachments.caption": {}},
            "fragment_size": 180,
            "number_of_fragments": 1,
            "pre_tags": ["<mark>"], "post_tags": ["</mark>"]
        }
    }

    if recent == "sort":
        body["query"] = {"bool": base_bool}
        body["track_scores"] = track_scores
        body["sort"] = [
            {"date": {"order": "desc"}},
            {"_score": {"order": "desc"}}
        ]
    elif recent == "score":
        # in build_query(...), for recent == "score":
        body["query"] = {
            "function_score": {
                "query": {"bool": base_bool},
                "boost_mode": "multiply",   # <<< was "sum" in your snippet
                "score_mode": "multiply",   # <<< multiplicative effect
                "functions": [
                    {
                      "gauss": {
                        "date": {"origin": "now", "scale": "7d", "offset": "0d", "decay": 0.5}
                      }
                    }
                  # you can also add small boosts, e.g. has_attachments:
                  # {"field_value_factor": {"field": "has_attachments", "factor": 1.1, "missing": 1.0}}
                ]
            }
        }
    else:  # "none"
        body["query"] = {"bool": base_bool}

    return body

def search(
    user_q,
    k: int = 10,
    filters=None,
    since=None,
    until=None,
    index: str = "swissfel_elog",
    recent: str = "score",           # "sort" | "score" | "none"
    strategy: str = "auto",          # "auto" | "strict" | "latest"
    too_many_threshold: int = 400
):
    """
    strategy:
      - "strict": single-shot query (no retries)
      - "latest": force recent-first newest results (sort) and return top-k
      - "auto": try once; if too many → add time window; if empty with time → relax
    """
    filters = _clean_filters(filters)

    # Strategy normalization
    if strategy == "latest":
        recent = "sort"

    def _run(qbody, size_override=None):
        b = dict(qbody)
        if size_override:
            b["size"] = size_override
        return es.search(index=index, body=b)

    def _to_hits(res):
        hits = []
        for h in res["hits"]["hits"]:
            s = h["_source"]
            hl = h.get("highlight", {}) or {}
            snippet = (
                hl.get("title", [None])[0]
                or hl.get("body", [None])[0]
                or hl.get("attachments.caption", [None])[0]
                or (s.get("body","")[:240])
            )
            hits.append({
                "elog_id": s.get("doc_id") or h["_id"],
                "title": s.get("title",""),
                "timestamp": s.get("date"),
                "author": s.get("author",""),
                "category": s.get("category",""),
                "domain": s.get("domain",""),
                "system": s.get("system",""),
                "section": s.get("section",""),
                "snippet": snippet,
                "attachments": s.get("attachments", []),
                "score": h.get("_score", 0.0)
            })
        return hits, res["hits"]["total"]["value"]

    # Pass 1
    body = build_query(user_q, max(k, 50 if strategy != "strict" else k), filters, since, until, recent)
    res = _run(body)
    hits, total = _to_hits(res)

    # If no hits, try OR operator fallback (keep same recency policy)
    if not hits:
        try:
            q = body["query"]
            # reach into the bool/multi_match operator:
            mm = q["function_score"]["query"]["bool"]["must"][0]["multi_match"] if "function_score" in q \
                 else q["bool"]["must"][0]["multi_match"]
            mm["operator"] = "or"
            res = _run(body)
            hits, total = _to_hits(res)
        except Exception:
            pass

    if strategy == "strict":
        return hits[:k]

    # Too many results → add a coarse time window
    if total > too_many_threshold and not since and not until and not filters:
        # try last 90 days
        now = datetime.now(timezone.utc)
        since90 = (now - timedelta(days=90)).isoformat()
        body2 = build_query(user_q, max(k, 50), filters, since90, None, recent)
        hits2, total2 = _to_hits(_run(body2))
        if total2 > too_many_threshold:
            # try last 30 days
            since30 = (now - timedelta(days=30)).isoformat()
            body3 = build_query(user_q, max(k, 50), filters, since30, None, recent)
            hits3, total3 = _to_hits(_run(body3))
            return (hits3 or hits2 or hits)[:k]
        return (hits2 or hits)[:k]

    # Empty with time filter → widen then drop
    if not hits and (since or until):
        # widen to 365d
        now = datetime.now(timezone.utc)
        since365 = (now - timedelta(days=365)).isoformat()
        body_wide = build_query(user_q, max(k, 50), filters, since365, None, recent)
        hits_w, total_w = _to_hits(_run(body_wide))
        if hits_w:
            return hits_w[:k]
        # drop date filter entirely, keep recent-first via `recent`
        body_drop = build_query(user_q, max(k, 50), filters=None, since=None, until=None, recent=recent)
        hits_d, _ = _to_hits(_run(body_drop))
        return hits_d[:k]

    # "latest" → take freshest by date desc regardless of score
    if strategy == "latest":
        # Ensure sorted by date desc by rebuilding with sort
        body_latest = build_query(user_q, max(k, 50), filters, since, until, recent="sort")
        hits_latest, _ = _to_hits(_run(body_latest))
        return hits_latest[:k]

    return hits[:k]


if __name__ == "__main__":
    """
    Test the search function with various queries and parameters.
    """
    print(search("Orbit response", k=3, strategy="auto", recent="sort"))
    print("\n")
    print(search("Orbit response", k=3, strategy="auto", recent="score"))
    print("\n")
    print(search("Orbit response", k=3, strategy="latest", recent="sort"))
    

