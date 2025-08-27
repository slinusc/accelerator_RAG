"""
ElasticSearch Schema and Primitive Actions for SwissFEL ELOG System
================================================================

This module defines:
1. Complete data schema for the SwissFEL ELOG ElasticSearch index
2. Available filter values (categories, systems, domains, sections)
3. Primitive search actions that can be composed into complex queries
4. Data structures for building agentic search plans
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

@dataclass
class ElogDocument:
    """Structure of a single ELOG document in ElasticSearch"""
    doc_id: str
    title: str
    author: str
    date: datetime
    category: str
    domain: str
    system: str
    section: str
    body: str
    html: str
    attachments: List[Dict[str, str]]  # filename, mime, path, caption, url
    _raw_meta: Dict[str, Any]  # Original metadata fields

@dataclass 
class SearchHit:
    """Structure of search result hit"""
    elog_id: str
    title: str
    timestamp: str
    author: str
    category: str
    domain: str
    system: str
    section: str
    snippet: str
    attachments: List[Dict]
    score: float

# ============================================================================
# FILTER VALUES (from actual ElasticSearch aggregations)
# ============================================================================

CATEGORIES = [
    "Info",                              # 12,985 docs
    "Problem",                           # 7,975 docs
    "Schicht-Übergabe",                 # 6,024 docs
    "Pikett",                           # 2,248 docs
    "Shift summary",                    # 1,459 docs
    "Measurement summary",              # 1,187 docs
    "Pre-beam Check",                   # 269 docs
    "DCM minutes",                      # 231 docs
    "Tipps & Tricks",                   # 205 docs
    "Überbrückung",                     # 169 docs
    "RC exchange minutes",              # 134 docs
    "Laser- & Gun-Performance Routine", # 126 docs
    "Access",                           # 123 docs
    "Schicht-Auftrag",                  # 82 docs
    "Procedures or Work-Arounds",       # 79 docs
    "Weekly reference settings",        # 68 docs
    "Seed laser operation",             # 16 docs
    "Measurement"                       # 7 docs
]

SYSTEMS = [
    "RF",                               # 4,275 docs
    "Operation",                        # 3,857 docs
    "Diagnostics",                      # 1,178 docs
    "Safety",                           # 1,040 docs
    "Controls",                         # 999 docs
    "Laser",                            # 741 docs
    "Other",                            # 660 docs
    "SwissFEL RF",                      # 507 docs
    "Photonics",                        # 419 docs
    "Run coordinator",                  # 417 docs
    "Magnet Power Supplies",            # 402 docs
    "Feedbacks",                        # 384 docs
    "Vacuum",                           # 367 docs
    "SwissFEL Controls",                # 347 docs
    "Insertion-devices",                # 308 docs
    "Beamdynamics",                     # 263 docs
    "Unknown",                          # 221 docs
    "Vakuum",                           # 203 docs (German)
    "SwissFEL Lasers",                  # 155 docs
    "SU-Ost",                          # 142 docs
    "Timing & Sync",                    # 110 docs
    "SPS/PSYS",                         # 79 docs
    "SwissFEL LLRF",                    # 62 docs
    "Water cooling",                    # 42 docs
    "Electric supply",                  # 31 docs
    "PLC",                              # 29 docs
    "Water cooling & Ventilation"       # 3 docs
]

DOMAINS = [
    "Injector",                         # 3,147 docs
    "Global",                           # 2,726 docs
    "Athos",                            # 1,696 docs
    "Aramis",                           # 1,446 docs
    "Linac3",                           # 1,325 docs
    "Linac1",                           # 1,202 docs
    "Aramis Beamlines",                 # 492 docs
    "Athos Beamlines",                  # 368 docs
    "Linac2"                            # 356 docs
]

# Top sections (100+ entries from 31k+ total, many are device-specific)
MAJOR_SECTIONS = [
    "SINEG01",      # 769 docs - Electron gun
    "SINSB02",      # 251 docs - S-band structure 
    "SINSB03",      # 229 docs - S-band structure
    "SINSB01",      # 187 docs - S-band structure
    "SINXB01",      # 175 docs - X-band structure
    "SINSB04",      # 171 docs - S-band structure
    "S10CB07",      # 139 docs - C-band cavity
    "SARFE10",      # 139 docs - Gas detector
    "S30CB12",      # 126 docs - C-band cavity
    "S10CB01",      # 124 docs - C-band cavity
    "S10CB04",      # 119 docs - C-band cavity
    "S30CB13",      # 111 docs - C-band cavity
    "S30CB02",      # 104 docs - C-band cavity
    "SINBC02",      # 104 docs - Bunch compressor
    "SATCB01",      # 103 docs - Athos C-band
]

# ============================================================================
# SEARCH STRATEGIES AND RECENCY OPTIONS
# ============================================================================

class SearchStrategy(Enum):
    AUTO = "auto"        # Adaptive search with fallbacks
    STRICT = "strict"    # Single-shot, no retries
    LATEST = "latest"    # Force newest results first

class RecencyMode(Enum):
    SORT = "sort"        # Explicit date sorting (newest first)
    SCORE = "score"      # Function score with time decay
    NONE = "none"        # Pure BM25, no recency bias

# ============================================================================
# PRIMITIVE SEARCH ACTIONS
# ============================================================================

@dataclass
class SearchAction:
    """Base class for all search actions"""
    action_type: str = "base_action"

@dataclass
class TextSearchAction(SearchAction):
    """Keyword-based search with relevance scoring"""
    query: str = ""
    action_type: str = "text_search"
    k: int = 10
    recent: RecencyMode = RecencyMode.SCORE
    strategy: SearchStrategy = SearchStrategy.AUTO

@dataclass
class TemporalSearchAction(SearchAction):
    """Retrieve entries within time range"""
    action_type: str = "temporal_search"
    query: str = ""  # Optional text filter
    since: Optional[str] = None  # ISO datetime
    until: Optional[str] = None  # ISO datetime
    k: int = 100

@dataclass
class FilteredSearchAction(SearchAction):
    """Search with metadata filters"""
    query: str = ""
    action_type: str = "filtered_search"
    categories: Optional[List[str]] = None
    systems: Optional[List[str]] = None
    domains: Optional[List[str]] = None
    sections: Optional[List[str]] = None
    k: int = 10

@dataclass
class BulkRetrieveAction(SearchAction):
    """Retrieve large batches for analysis"""
    filters: Optional[Dict[str, List[str]]] = None
    action_type: str = "bulk_retrieve"
    k: int = 100
    since: Optional[str] = None
    until: Optional[str] = None

@dataclass
class LatestEntriesAction(SearchAction):
    """Get most recent entries (chronologically)"""
    query: str = ""
    action_type: str = "latest_entries"
    k: int = 20
    filters: Optional[Dict[str, List[str]]] = None

# ============================================================================
# ANALYSIS ACTIONS (post-search processing)
# ============================================================================

@dataclass
class ExtractIncidentsAction:
    """Identify incident/problem entries from search results"""
    action_type: str = "extract_incidents"
    keywords: Optional[List[str]] = None  # Default: ["error", "failure", "problem", "alert", "fault", "down", "trip"]
    categories: Optional[List[str]] = None  # Default: ["Problem", "Pikett"]
    min_severity: int = 1  # 1-5 scale

@dataclass
class RankByImportanceAction:
    """Score entries by importance/criticality"""
    action_type: str = "rank_importance"
    criteria: Optional[List[str]] = None  # ["emergency", "critical", "urgent", "safety"]
    boost_categories: Optional[List[str]] = None  # ["Problem", "Safety", "Pikett"]
    boost_systems: Optional[List[str]] = None  # ["Safety", "RF", "Controls"]

@dataclass
class AggregateByMetadataAction:
    """Group and count entries by metadata fields"""
    group_by: Optional[List[str]] = None  # ["category", "system", "domain", "author"]
    action_type: str = "aggregate_metadata" 
    time_bucket: Optional[str] = None  # "day", "week", "month"

@dataclass
class TimeSeriesAnalysisAction:
    """Analyze temporal patterns in the data"""
    bucket_size: str = "1d"  # "1h", "1d", "1w", "1M"
    action_type: str = "timeseries_analysis"
    metrics: Optional[List[str]] = None  # ["count", "incidents", "systems_affected"]

@dataclass
class SmartRerankAction:
    """Smart reranking using semantic similarity and/or LLM"""
    action_type: str = "smart_rerank"
    method: str = "hybrid"  # "semantic", "llm", "hybrid"
    target_k: int = 10      # Final number of results
    max_per_category: Optional[int] = 3  # Diversity constraint

# ============================================================================
# MULTI-STEP SEARCH PLANS
# ============================================================================

@dataclass
class SearchPlan:
    """Complete search strategy with multiple steps"""
    plan_type: str
    description: str
    steps: List[Union[SearchAction, Any]]  # Mix of search and analysis actions
    synthesis_prompt: Optional[str] = None  # How to combine results

# ============================================================================
# PLAN TEMPLATES FOR COMMON QUERY TYPES
# ============================================================================

def create_temporal_analysis_plan(since: str, until: str, focus: str = "incidents") -> SearchPlan:
    """Plan for 'what happened last week' type queries"""
    return SearchPlan(
        plan_type="temporal_analysis",
        description=f"Analyze events from {since} to {until} focusing on {focus}",
        steps=[
            BulkRetrieveAction(
                filters={},  # No initial filters - get everything in time range
                k=500,       # Retrieve many more entries
                since=since,
                until=until
            ),
            ExtractIncidentsAction(
                keywords=["problem", "error", "failure", "trip", "fault", "down", "issue", 
                         "beam", "dump", "abort", "interlock", "alarm", "warning", "reset"],
                categories=["Problem", "Pikett", "Safety"]  # Include Safety category
            ),
            RankByImportanceAction(
                criteria=["critical", "urgent", "safety", "beam", "down", "fault", "trip"],
                boost_categories=["Problem", "Pikett", "Safety"]
            ),
            SmartRerankAction(
                method="hybrid",
                target_k=25,  # Allow more incidents for temporal analysis
                max_per_category=5  # More per category for comprehensive coverage
            ),
            AggregateByMetadataAction(
                group_by=["category", "system", "domain"],
                time_bucket="day"
            )
        ],
        synthesis_prompt="Summarize the most important incidents and events, grouping by system and severity. Highlight patterns and trends."
    )

def create_system_health_plan(system: str, timeframe: str) -> SearchPlan:
    """Plan for 'how is RF system doing' type queries"""
    return SearchPlan(
        plan_type="system_health",
        description=f"Analyze health and status of {system} system over {timeframe}",
        steps=[
            FilteredSearchAction(
                query="",
                systems=[system],
                k=100
            ),
            ExtractIncidentsAction(
                categories=["Problem", "Info"],
                keywords=["error", "fault", "maintenance", "repair", "calibration"]
            ),
            TimeSeriesAnalysisAction(
                bucket_size="1d",
                metrics=["count", "incidents"]
            )
        ],
        synthesis_prompt=f"Provide a health assessment of the {system} system, including recent issues, maintenance activities, and overall trend."
    )

def create_device_status_plan(device_pattern: str) -> SearchPlan:
    """Plan for 'status of SATUN18' type queries"""
    return SearchPlan(
        plan_type="device_status", 
        description=f"Get current status and recent activity for devices matching {device_pattern}",
        steps=[
            LatestEntriesAction(
                query=device_pattern,
                k=20
            ),
            TextSearchAction(
                query=f"{device_pattern} status reset calibration maintenance",
                k=10,
                recent=RecencyMode.SORT
            )
        ],
        synthesis_prompt=f"Summarize the current status of {device_pattern}, including recent resets, maintenance, or issues."
    )

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_filter_suggestions(field: str) -> List[str]:
    """Get available filter values for a field"""
    mapping = {
        "category": CATEGORIES,
        "system": SYSTEMS, 
        "domain": DOMAINS,
        "section": MAJOR_SECTIONS
    }
    return mapping.get(field, [])

def validate_filters(filters: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Validate that filter values are in allowed lists"""
    valid_filters = {}
    for field, values in filters.items():
        if field in ["category", "system", "domain", "section"]:
            allowed = get_filter_suggestions(field)
            valid_values = [v for v in values if v in allowed]
            if valid_values:
                valid_filters[field] = valid_values
    return valid_filters

def detect_query_type(user_query: str) -> str:
    """Classify user query to suggest appropriate plan type"""
    query_lower = user_query.lower()
    
    if any(phrase in query_lower for phrase in ["last week", "yesterday", "recent", "what happened"]):
        return "temporal_analysis"
    elif any(phrase in query_lower for phrase in ["status of", "how is", "current state"]):
        return "device_status" 
    elif any(system.lower() in query_lower for system in SYSTEMS[:10]):  # Top systems
        return "system_health"
    elif any(phrase in query_lower for phrase in ["compare", "difference", "vs", "versus"]):
        return "comparison"
    elif any(phrase in query_lower for phrase in ["trend", "over time", "pattern"]):
        return "trend_analysis"
    else:
        return "general_search"