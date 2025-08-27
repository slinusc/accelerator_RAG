# langgraph_mock.py
from typing import TypedDict, List, Dict, Any, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, END
from datetime import datetime
import pytz  # or zoneinfo if Python 3.9+: from zoneinfo import ZoneInfo
from es_search import search as es_search
from action_executor import ActionExecutor
from elog_schema import (
    create_temporal_analysis_plan, create_device_status_plan, create_system_health_plan,
    detect_query_type
)

# ---- Fake "API" (toy KB) -----------------------------------------------------
import re
from datetime import datetime
from dateutil import parser as dtparser

# ---- Robust JSON coercion ----------------------------------------------------
import json, re
import logging
from colorama import Fore, Style, init as colorama_init

import json
from datetime import timezone
from dateutil import parser as dtp
from datetime import datetime, timedelta
import pytz

# ---- Helper functions ---------------------------------------------------------

def _ensure_k(x, default=5):
    try:
        k = int(x)
        return max(1, min(10, k))
    except Exception:
        return default

def _to_iso_z(s: str) -> str:
    try:
        dt = dtp.parse(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return s

def coerce_json(text: str) -> dict:
    """
    Coerce LLM output into a JSON object:
    - Strip code fences
    - Try direct json.loads
    - Fallback to extracting the first {...} block
    """
    if text is None:
        raise ValueError("LLM returned None")
    t = text.strip()

    # Strip ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = t.strip("`")
        lines = [ln for ln in t.splitlines() if ln.strip().lower() != "json"]
        t = "\n".join(lines)

    # Try direct
    try:
        return json.loads(t)
    except Exception:
        pass

    # Fallback: take first {...}
    start = t.find("{")
    end = t.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = t[start:end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            candidate_min = re.sub(r"\s+", " ", candidate)
            return json.loads(candidate_min)

    snippet = t[:300].replace("\n", "\\n")
    raise ValueError(f"LLM did not return valid JSON. Snippet: {snippet}")

# ---- Real Ollama interface (Qwen2.5-VL) -------------------------------------
import os, httpx

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5vl:32b-q4_K_M")

def call_llm_prompt_ollama(system: str, user_text: str, images: Optional[List[str]] = None,
                           temperature: float = 0.2, max_tokens: int = 1000) -> str:
    """
    Calls Ollama's OpenAI-compatible /v1/chat/completions endpoint.

    Args:
      system: system prompt (string)
      user_text: user content as text
      images: optional list of local image paths or file:// URLs (Qwen2.5-VL can use them)
      temperature, max_tokens: generation controls

    Returns:
      The assistant's message content (string).
    """
    # Build content as OpenAI-style message; include images if provided
    user_content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
    if images:
        for p in images:
            # Support absolute paths -> convert to file:// URL
            if os.path.exists(p):
                url = f"file://{os.path.abspath(p)}"
            else:
                url = p  # assume already a URL
            user_content.append({"type": "image_url", "image_url": {"url": url}})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    url = f"{OLLAMA_HOST}/v1/chat/completions"
    try:
        with httpx.Client(timeout=120) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            msg = data["choices"][0]["message"]["content"]
            if not isinstance(msg, str):
                # Some models can return array-like content; normalize
                if isinstance(msg, list):
                    # concatenate text parts
                    texts = []
                    for part in msg:
                        if isinstance(part, dict) and part.get("type") == "text":
                            texts.append(part.get("text", ""))
                    msg = "\n".join(texts)
                else:
                    msg = str(msg)
            return msg
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}") from e
    except KeyError as e:
        raise RuntimeError(f"Unexpected Ollama response shape: {e}") from e

# ---- LLM system prompts (strict JSON) ---------------------------------------
PLANNER_SYS = (
    "You are Planner for a retrieval system.\n"
    "Return STRICT JSON with EXACT keys:\n"
    "{\n"
    '  "plan": "search|respond|analyze",\n'
    '  "query_type": "temporal_analysis|device_status|system_health|general_search|greeting",\n'
    '  "query": "plain keywords (NO dates/times)",\n'
    '  "k": <int 1-50>,\n'
    '  "strategy": "auto | latest | strict",\n'
    '  "recent": "score | sort | none",\n'
    '  "filters": {\n'
    '    "date_from": "ISO8601 with timezone or null",\n'
    '    "date_to":   "ISO8601 with timezone or null",\n'
    '    "category":  ["..."] (optional),\n'
    '    "domain":    ["..."] (optional),\n'
    '    "section":   ["..."] (optional)\n'
    "  }\n"
    "}\n"
    "\n"
    "You will be given a 'now' field (ISO8601, Europe/Zurich).\n"
    "Rules:\n"
    "- If the user query is NOT a specific technical question (e.g., greetings), use plan=\"respond\", query_type=\"greeting\".\n"
    "- For complex analytical queries (what happened, incidents, problems, analysis), use plan=\"analyze\" with:\n"
    "  * query_type=\"temporal_analysis\": queries about time periods (last week, yesterday, recent incidents)\n"
    "  * query_type=\"device_status\": queries about specific devices (SATUN18, status of X)\n" 
    "  * query_type=\"system_health\": queries about system performance (RF system, how is X performing)\n"
    "- For simple searches, use plan=\"search\", query_type=\"general_search\".\n"
    "- If the question asks for the **last/latest/most recent** item and does NOT specify a period, set strategy=\"latest\", recent=\"sort\", and leave date_from/date_to null.\n"
    "- If the user specifies a period (yesterday, last week, a date range), compute date_from/date_to using 'now'; use strategy=\"auto\", recent=\"score\".\n"
    "- Otherwise: strategy=\"auto\", recent=\"score\", date filters null.\n"
    "- NEVER put dates/relative time words in 'query'.\n"
    "- Use Europe/Zurich offsets (+01:00 or +02:00).\n"
    "- Return ONLY the JSON object. No prose. No code fences.\n"
    "\n"
    "Examples:\n"
    "USER: When was the last shift summary?\n"
    'OUTPUT: {"plan":"search","query_type":"general_search","query":"shift summary","k":3,"strategy":"latest","recent":"sort","filters":{"date_from":null,"date_to":null}}\n'
    "\n"
    "USER: What important incidents happened last week?\n"
    'OUTPUT: {"plan":"analyze","query_type":"temporal_analysis","query":"incidents problems","k":20,"strategy":"auto","recent":"score","filters":{"date_from":"2025-08-18T00:00:00+02:00","date_to":"2025-08-24T23:59:59+02:00"}}\n'
    "\n"
    "USER: What's the status of SATUN18?\n"
    'OUTPUT: {"plan":"analyze","query_type":"device_status","query":"SATUN18","k":15,"strategy":"latest","recent":"sort","filters":{"date_from":null,"date_to":null}}\n'
    "\n"
    "USER: How is the RF system performing?\n"
    'OUTPUT: {"plan":"analyze","query_type":"system_health","query":"RF","k":20,"strategy":"auto","recent":"score","filters":{"system":["RF"],"date_from":null,"date_to":null}}\n'
    "\n"
    "USER: hello\n"
    'OUTPUT: {"plan":"respond","query_type":"greeting"}\n'
)


    

EVALUATOR_SYS = (
  "You are the ELOG Result Evaluator for accelerator operations.\n"
  "Goal: decide if the current ELOG hits likely contain evidence to answer the operator’s intent.\n"
  "Return ONLY strict JSON with keys: decision (accept|refine), reason, refinement?, focus_ids? (array of hit ids, most relevant first).\n"
  "\n"
  "RECENCY POLICY:\n"
  "- Prefer more recent occurrences. Compute age_days for each hit from 'now' to hit.timestamp (ISO 8601).\n"
  "- Base recency score := exp(-age_days / H). Use half-life H=30 days by default; if query implies recency "
  "(e.g., contains: last, latest, this week, yesterday, today, recent, last reset), use H=7.\n"
  "\n"
  "RELEVANCE SIGNALS (combine qualitatively):\n"
  "- Content match: device/event keywords present in title/body/snippet.\n"
  "- Recency score (as above). Newer = better.\n"
  "- Evidence strength: presence of attachments (plots/screenshots), operator/shift summary authorship.\n"
  "\n"
  "DECISION RULE:\n"
  "- If ≥1 high-relevance hit exists (good content match AND decent recency score), set decision=\"accept\" and list focus_ids "
  "in descending relevance. Otherwise decision=\"refine\" and provide a concrete refinement hint (e.g., add device alias, "
  "narrow/expand date window, pick Section/Domain).\n"
  "\n"
  "STRICT JSON ONLY. No prose, no code fences."
)

REPORTER_SYS = """You are an expert SwissFEL accelerator system analyst.

CRITICAL INSTRUCTION: First check if the provided ELOG data directly answers the user's specific query.

For device-specific queries (e.g., "What's the status of SATUN18?"):
- If NO entries specifically mention the requested device by name, start your response with: 
  "### Query Result: No Specific Information Found\n\nNo recent ELOG entries were found that specifically mention [DEVICE_NAME] or its current status/reset activity."
- Then provide any contextually relevant information from related systems if available.
- Be honest about data limitations.

For general queries with relevant data, write a comprehensive technical summary in markdown format.

Focus on:
- Critical incidents and their impact on operations
- System-specific issues (RF, Controls, Diagnostics, etc.) 
- Temporal patterns and recurring problems
- Device-specific failures and resets
- Operational significance and severity assessment

Use clear headings, bullet points, and technical terminology. 
Highlight the most important incidents first.
Reference specific elog_ids when discussing incidents.
Be direct and honest about what information was actually found vs. what was requested.
"""


# ---- State & Report schema ---------------------------------------------------
# --- Models ---
from typing import List, Literal, Optional
from pydantic import BaseModel

Status = Literal["ok", "partial", "no_hit", "error"]

class Table(BaseModel):
    title: str
    columns: List[str]
    rows: List[List[str]]

class Figure(BaseModel):
    caption: str
    source_elog_id: str
    path_or_url: str

class Citation(BaseModel):
    elog_id: str
    url: str

class Report(BaseModel):
    status: Status
    query: str
    iterations: int = 1
    text: List[str]
    citations: List[Citation]
    notes: str = ""


class GraphState(TypedDict, total=False):
    user_query: str
    plan: Dict[str, Any]              # from Planner
    hits: List[Dict[str, Any]]        # from Searcher
    evaluation: Dict[str, Any]        # from Evaluator
    report: Report
    iterations: int
    error: Optional[str]

colorama_init(autoreset=True)
logging.basicConfig(
    level=logging.INFO,
    format=f'{Fore.CYAN}%(asctime)s{Style.RESET_ALL} %(levelname)s {Fore.YELLOW}%(name)s{Style.RESET_ALL}: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("LangGraph")
# ---- Nodes -------------------------------------------------------------------

import pytz
from datetime import datetime
import json
import time

def planner_node(state: GraphState) -> GraphState:
    start_time = time.time()
    
    if state.get("iterations") is None:
        state["iterations"] = 0
    if state.get("timing") is None:
        state["timing"] = {}

    refinement = state.get("evaluation", {}).get("refinement")
    user_prompt = state["user_query"]
    if refinement:
        user_prompt += f"\nRefinement hint: {refinement}"

    # Provide current time to allow resolving relative dates
    tz = pytz.timezone("Europe/Zurich")
    now_iso = datetime.now(tz).isoformat(timespec="seconds")
    payload = json.dumps({"now": now_iso, "user_query": user_prompt}, ensure_ascii=False)

    llm_start = time.time()
    out = call_llm_prompt_ollama(PLANNER_SYS, payload, images=None, temperature=0.0, max_tokens=300)
    llm_time = time.time() - llm_start
    
    logger.info(f"{Fore.BLUE}[Planner]{Style.RESET_ALL} user_prompt: {user_prompt}")
    logger.info(f"{Fore.BLUE}[Planner]{Style.RESET_ALL} LLM output: {out}")

    plan = coerce_json(out)  # tolerant JSON coercion

    # Hard defaults/guards
    plan["plan"] = plan.get("plan", "search")
    plan["k"] = _ensure_k(plan.get("k", 5))
    plan["strategy"] = plan.get("strategy") or "auto"
    plan["recent"] = plan.get("recent") or ("sort" if plan["strategy"] == "latest" else "score")

    # Keep query clean of dates (defensive)
    q = (plan.get("query") or "").strip()
    q_low = q.lower()
    if re.search(r"\d{4}-\d{2}-\d{2}", q_low) or any(t in q_low for t in ["yesterday","today","tomorrow","last","week","month","year","recent"]):
        q = re.sub(r"\d{4}-\d{2}-\d{2}", " ", q_low)
        q = re.sub(r"\b(yesterday|today|tomorrow|last|recent|week|month|year|days?)\b", " ", q, flags=re.I)
        q = re.sub(r"\s+", " ", q).strip()
    plan["query"] = q or state["user_query"]

    # Normalize filters block
    filters = plan.get("filters") or {}
    plan["filters"] = {
        "date_from": filters.get("date_from"),
        "date_to": filters.get("date_to"),
        **({} if not filters.get("category") else {"category": filters["category"]}),
        **({} if not filters.get("domain")   else {"domain":   filters["domain"]}),
        **({} if not filters.get("section")  else {"section":  filters["section"]}),
    }

    state["plan"] = plan
    
    # Record timing
    total_time = time.time() - start_time
    state["timing"]["planner"] = {
        "total": round(total_time, 3),
        "llm": round(llm_time, 3)
    }
    
    logger.info(f"{Fore.BLUE}[Planner]{Style.RESET_ALL} Parsed plan: {state['plan']}")
    logger.info(f"{Fore.BLUE}[Planner]{Style.RESET_ALL} Timing: {total_time:.3f}s (LLM: {llm_time:.3f}s)")
    return state


def searcher_node(state: GraphState) -> GraphState:
    plan = state.get("plan", {})
    if plan.get("plan") != "search":
        state["error"] = f"Unknown plan: {plan}"
        return state

    q         = plan.get("query", state["user_query"])
    k         = int(plan.get("k", 5))
    plan_filters = plan.get("filters", {}) or {}
    
    # Extract date filters for since/until params
    since     = plan_filters.pop("date_from", None)
    until     = plan_filters.pop("date_to", None)
    
    # Clean up remaining filters for es_search (remove date filters to avoid duplication)
    es_filters = {k: v for k, v in plan_filters.items() if k not in ["date_from", "date_to"] and v}
    
    strategy  = plan.get("strategy", "auto")
    recent    = plan.get("recent", "score")

    logger.info(f"{Fore.GREEN}[Searcher]{Style.RESET_ALL} Plan: {plan}")
    logger.info(f"{Fore.GREEN}[Searcher]{Style.RESET_ALL} Executing search: {q} (k={k}) strategy={strategy} recent={recent} since={since} until={until}")
    if es_filters:
        logger.info(f"{Fore.GREEN}[Searcher]{Style.RESET_ALL} Metadata filters: {es_filters}")

    hits = es_search(
        q, k=k, filters=es_filters, since=since, until=until,
        index="swissfel_elog", recent=recent, strategy=strategy
    )

    state["hits"] = hits
    logger.info(f"{Fore.GREEN}[Searcher]{Style.RESET_ALL} Retrieved {len(hits)} hits")
    return state



MAX_ITERS = 5

def evaluator_node(state: GraphState) -> GraphState:
    state["iterations"] = (state.get("iterations") or 0) + 1
    logger.info(f"{Fore.MAGENTA}[Evaluator]{Style.RESET_ALL} Iteration: {state['iterations']}")
    compact_hits = []
    for h in state.get("hits", []):
        compact_hits.append({
            "id": h.get("id") or h.get("elog_id"),
            "title": h.get("title", ""),
            "snippet": (h.get("body") or h.get("snippet") or "")[:600],
            "timestamp": h.get("timestamp"),
            "has_attachments": bool(h.get("attachments"))
        })
    tz = pytz.timezone("Europe/Zurich")
    now_iso = datetime.now(tz).isoformat(timespec="seconds")
    payload = json.dumps({
        "query": state["user_query"],
        "now": now_iso,
        "hits": compact_hits
    }, ensure_ascii=False)
    logger.info(f"{Fore.MAGENTA}[Evaluator]{Style.RESET_ALL} Payload: {payload}")
    out = call_llm_prompt_ollama(EVALUATOR_SYS, payload, images=None, temperature=0.1, max_tokens=300)
    logger.info(f"{Fore.MAGENTA}[Evaluator]{Style.RESET_ALL} LLM output: {out}")
    ev = coerce_json(out)
    logger.info(f"{Fore.MAGENTA}[Evaluator]{Style.RESET_ALL} Parsed evaluation: {ev}")
    if ev.get("decision") == "refine" and state["iterations"] > MAX_ITERS:
        ev = {"decision": "accept", "reason": "max iterations reached"}
        logger.warning(f"{Fore.MAGENTA}[Evaluator]{Style.RESET_ALL} Max iterations reached, forcing accept.")
    state["evaluation"] = ev
    focus = ev.get("focus_ids")
    if ev.get("decision") == "accept" and focus:
        id_to_hit = { (h.get("id") or h.get("elog_id")): h for h in state.get("hits", []) }
        state["hits"] = [id_to_hit[i] for i in focus if i in id_to_hit]
    return state

def reporter_node(state):
    """
    state has:
      - user_query: str
      - hits: List[dict]  (from Searcher; we pass these through to raw_hits)
      - plan: dict (to check if this is a respond-only case)
    """
    start_time = time.time()
    
    user_query = state["user_query"]
    hits = state.get("hits", []) or []
    plan = state.get("plan", {})
    
    # Handle direct respond cases (no search performed)
    if plan.get("plan") == "respond":
        report = Report(
            status="ok",
            query=user_query,
            iterations=1,
            text=["Hello! I'm here to help you search through ELOGs and accelerator system information. Do you have a specific question about the Elog system?"],
            citations=[],
            notes="Generic greeting - no search performed.",
        )
        return {"report": report}

    # Prepare a compact evidence summary table directly from hits
    # (LLM can augment/condense, but we give it good structure up-front)
    table_rows = []
    for h in hits:
        row = [
            h.get("elog_id", ""),
            h.get("section", "") or "",
            "reset" if "reset" in (h.get("title","").lower()+h.get("snippet","").lower()) else "",
            "",  # value
            "",  # unit
            _to_iso_z(h.get("timestamp","")),
            h.get("author",""),
        ]
        table_rows.append(row)

    # Compose the assistant call
    sys_prompt = REPORTER_SYS
    user_payload = {
        "query": user_query,
        "hits": hits,  # will be copied into raw_hits
        "draft_table": {
            "title": "DRM Resets / Related Events",
            "columns": ["elog_id","device","event","value","unit","timestamp","author"],
            "rows": table_rows
        }
    }

    logger.info(f"{Fore.CYAN}[Reporter]{Style.RESET_ALL} User query: {user_query}")
    logger.info(f"{Fore.CYAN}[Reporter]{Style.RESET_ALL} Payload: {json.dumps(user_payload, ensure_ascii=False)}")
    
    llm_start = time.time()
    llm_resp = call_llm_prompt_ollama(
        system=sys_prompt,
        user_text=json.dumps(user_payload, ensure_ascii=False),
        temperature=0.0,
        max_tokens=1200,
    )
    llm_time = time.time() - llm_start
    
    logger.info(f"{Fore.CYAN}[Reporter]{Style.RESET_ALL} LLM output: {llm_resp}")
    
    # Record timing
    total_time = time.time() - start_time
    if "timing" not in state:
        state["timing"] = {}
    state["timing"]["reporter"] = {
        "total": round(total_time, 3),
        "llm": round(llm_time, 3)
    }
    
    # Build structured JSON status directly from LangGraph state (no LLM parsing)
    data = {
        "status": "ok" if hits else "no_hit",
        "query": user_query,
        "iterations": state.get("iterations", 1),
        "text": [llm_resp],  # Store the human-readable LLM analysis
        "citations": [],
        "notes": f"Analysis completed - {len(hits)} entries processed",
        "timing": state.get("timing", {})  # Include timing in final output
    }
    allowed = {"ok","partial","no_hit","error"}
    if data.get("status") not in allowed:
        data["status"] = "ok"
    report = Report(
        status=data["status"],
        query=data.get("query", user_query),
        iterations=data.get("iterations", 1),
        text=data.get("text", data.get("text", [])),
        tables=[Table(**t) for t in data.get("tables", [])],
        figures=[Figure(**f) for f in data.get("figures", [])],
        citations=[Citation(elog_id="", url=c) if isinstance(c, str) else Citation(**c) for c in data.get("citations", [])],
        raw_hits=data.get("raw_hits", hits),
        notes=data.get("notes", ""),
    )
    
    logger.info(f"{Fore.CYAN}[Reporter]{Style.RESET_ALL} Timing: {total_time:.3f}s (LLM: {llm_time:.3f}s)")
    return {"report": report}

def analyzer_node(state: GraphState) -> GraphState:
    """
    Execute agentic analysis plans using ActionExecutor
    """
    start_time = time.time()
    
    user_query = state["user_query"]
    plan = state.get("plan", {})
    query_type = plan.get("query_type", "general_search")
    
    logger.info(f"{Fore.GREEN}[Analyzer]{Style.RESET_ALL} Query type: {query_type}")
    
    try:
        executor = ActionExecutor()
        
        # Create appropriate action plan based on query type
        if query_type == "temporal_analysis":
            # Extract date filters for temporal analysis
            filters = plan.get("filters", {})
            since = filters.get("date_from")
            until = filters.get("date_to")
            action_plan = create_temporal_analysis_plan(since, until, focus="incidents")
            
        elif query_type == "device_status":
            device_pattern = plan.get("query", "")
            action_plan = create_device_status_plan(device_pattern)
            
        elif query_type == "system_health":
            system = plan.get("query", "")
            if not system:
                # Try to extract from filters
                system_filters = plan.get("filters", {}).get("system", [])
                system = system_filters[0] if system_filters else "RF"
            action_plan = create_system_health_plan(system, "recent")
            
        else:
            # Fallback to simple search for general queries
            logger.warning(f"Unknown query type: {query_type}, falling back to search")
            return searcher_node(state)
        
        # Execute the action plan
        logger.info(f"{Fore.GREEN}[Analyzer]{Style.RESET_ALL} Executing {action_plan.plan_type} plan with {len(action_plan.steps)} steps")
        execution_start = time.time()
        results = executor.execute_plan(action_plan)
        execution_time = time.time() - execution_start
        
        # Convert results to expected format
        hits = results.get("hits", [])
        logger.info(f"{Fore.GREEN}[Analyzer]{Style.RESET_ALL} Analysis completed: {len(hits)} hits, {len(results.get('steps', []))} steps executed")
        
        # Record timing
        total_time = time.time() - start_time
        if "timing" not in state:
            state["timing"] = {}
        state["timing"]["analyzer"] = {
            "total": round(total_time, 3),
            "execution": round(execution_time, 3),
            "step_details": results.get("timing", {})
        }
        
        # Add analysis metadata to state
        state["hits"] = hits
        state["analysis_results"] = results
        # Let evaluator assess the quality of analysis results
        
        logger.info(f"{Fore.GREEN}[Analyzer]{Style.RESET_ALL} Timing: {total_time:.3f}s (execution: {execution_time:.3f}s)")
        return state
        
    except Exception as e:
        logger.error(f"{Fore.RED}[Analyzer]{Style.RESET_ALL} Error in analysis: {e}")
        # Fallback to regular search on error
        return searcher_node(state)


# ---- Build the graph ---------------------------------------------------------
def build_graph():
    g = StateGraph(GraphState)
    g.add_node("planner", planner_node)
    g.add_node("searcher", searcher_node)
    g.add_node("analyzer", analyzer_node)
    g.add_node("evaluator", evaluator_node)
    g.add_node("reporter", reporter_node)

    g.set_entry_point("planner")
    
    def on_plan(state: GraphState):
        plan_type = state.get("plan", {}).get("plan", "search")
        if plan_type == "respond":
            return "reporter"
        elif plan_type == "analyze":
            return "analyzer"
        else:  # "search"
            return "searcher"
    
    g.add_conditional_edges("planner", on_plan, {
        "searcher": "searcher", 
        "analyzer": "analyzer",
        "reporter": "reporter"
    })
    
    # Traditional search path
    g.add_edge("searcher", "evaluator")

    def on_eval(state: GraphState):
        decision = state.get("evaluation", {}).get("decision", "accept")
        return "planner" if decision == "refine" else "reporter"

    g.add_conditional_edges("evaluator", on_eval, {"planner": "planner", "reporter": "reporter"})
    
    # Analyzer path goes through evaluator for quality assessment
    g.add_edge("analyzer", "evaluator")
    
    g.add_edge("reporter", END)
    return g.compile()

# ---- Run from CLI / Notebook -------------------------------------------------
if __name__ == "__main__":
    import sys, argparse, json as pyjson, re

    def sanitize_query(q: str) -> str:
        if not q:
            return ""
        # Ignore Jupyter/IPython kernel files or anything that looks like a path to a JSON in runtime
        if re.search(r"kernel-.*\.json", q) or q.startswith("--f=") or "/jupyter/runtime/" in q or q.startswith("/"):
            return ""
        return q.strip()

    parser = argparse.ArgumentParser(description="LangGraph hello demo", add_help=True)
    parser.add_argument("query", nargs="*", help="Natural-language query (e.g., 'When was the last reset of SATUN18?')")
    # parse_known_args will ignore Jupyter's --f=... noise
    args, _ = parser.parse_known_args()

    # Join positional words, sanitize, then fallback
    raw_q = " ".join(args.query)
    user_q = sanitize_query(raw_q) or "When was the last reset of SATUN18?"

    graph = build_graph()
    logger.info(f"{Fore.CYAN}[Main]{Style.RESET_ALL} Invoking graph with query: {user_q}")
    final_state = graph.invoke({"user_query": user_q})
    logger.info(f"{Fore.CYAN}[Main]{Style.RESET_ALL} Final state: {final_state}")
    print(pyjson.dumps(final_state["report"].model_dump(), indent=2))