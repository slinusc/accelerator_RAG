# SwissFEL Agentic RAG System

An intelligent, multi-step analysis system for querying SwissFEL accelerator ELOG data using LangGraph workflows, semantic search, and adaptive reranking.

## Overview

This system transforms simple keyword-based searches into truly **agentic** multi-step analysis workflows. Instead of just matching keywords, it:

- **Understands intent** through query classification
- **Plans analysis strategies** with multi-step workflows  
- **Executes systematically** using primitive search actions
- **Iteratively refines** searches through quality evaluation
- **Synthesizes insights** with comprehensive reporting

## Architecture

### Core Workflow (LangGraph)

```
┌─────────┐    ┌─────────┐    ┌───────────┐    ┌────────────┐
│ Planner │───▶│Searcher/│───▶│ Evaluator │───▶│ Reporter   │
│         │    │Analyzer │    │           │    │            │
└─────────┘    └─────────┘    └───────────┘    └────────────┘
                                      │              ▲
                                      ▼              │
                               ┌─────────────────────┘
                               │ Refine (max 6x)
```

### Agent Roles

#### 1. **Planner Agent**
- **Purpose**: Classifies user queries and selects appropriate search strategies
- **Input**: Natural language user query + refinement hints
- **Output**: Search plan with query type, parameters, and strategy
- **Query Types**:
  - `general_search`: Simple keyword-based search
  - `temporal_analysis`: "What happened last week" queries
  - `device_status`: "Status of SATUN18" queries  
  - `system_health`: "How is RF system doing" queries
  - `analyze`: Complex multi-step agentic analysis

#### 2. **Searcher Agent** 
- **Purpose**: Executes simple search plans
- **Capabilities**:
  - ElasticSearch integration with BM25 scoring
  - Recency bias (sort, score, none)
  - Metadata filtering (category, system, domain, section)
  - Timezone-aware date handling

#### 3. **Analyzer Agent**
- **Purpose**: Executes complex multi-step agentic analysis plans
- **Workflow**: Orchestrates primitive actions in sequence
- **Actions**: BulkRetrieve → ExtractIncidents → RankByImportance → SmartRerank → Aggregate
- **Adaptive Logic**: Adjusts result limits based on incident importance

#### 4. **Evaluator Agent**
- **Purpose**: Quality assessment and iterative refinement
- **Evaluation Criteria**:
  - Keyword matching between query and results
  - Recency relevance for time-sensitive queries
  - Device-specific matching for hardware queries
- **Actions**: `accept` (proceed to reporting) or `refine` (retry with hints)
- **Max Iterations**: 6 attempts before accepting best available results

#### 5. **Reporter Agent**
- **Purpose**: Synthesizes final comprehensive analysis
- **Key Features**:
  - **Explicit "no information found" handling** for device queries
  - Domain-aware analysis using SwissFEL knowledge
  - Structured tables and incident timelines
  - Comprehensive context even when specific data is missing

## Multi-Step Analysis Plans

### 1. Temporal Analysis Plan
**Trigger**: "What happened last week", "Recent incidents"

**Steps**:
1. **BulkRetrieve**: Get 500+ entries in time range
2. **ExtractIncidents**: Filter for problems/safety issues
3. **RankByImportance**: Score by criticality keywords
4. **SmartRerank**: Neural+LLM hybrid reranking (25 results)
5. **AggregateByMetadata**: Group by system/category/time

**Adaptive**: No hard result caps - captures all important incidents

### 2. Device Status Plan  
**Trigger**: "Status of SATUN18", "DEVICE999 reset"

**Steps**:
1. **LatestEntries**: Recent entries mentioning device (20 results)
2. **TextSearch**: Broader search with status keywords (10 results)

**Special Handling**: Reporter explicitly states when no device-specific information found

### 3. System Health Plan
**Trigger**: "How is RF system doing"

**Steps**:
1. **FilteredSearch**: System-specific entries (100 results) 
2. **ExtractIncidents**: Problems + maintenance activities
3. **TimeSeriesAnalysis**: Trend analysis over time

## Smart Reranking System

### Neural Cross-Encoder Approach
- **Model**: `cross-encoder/ms-marco-TinyBRT-L-2-v2`
- **Purpose**: True reranking (not just similarity) 
- **Input**: Query-passage pairs
- **Output**: Relevance scores for precise ranking

### Hybrid Scoring (50/50)
- **50% Neural**: Cross-encoder semantic relevance
- **50% LLM**: GPT-style relevance assessment (1-10 scale)
- **Efficiency**: Processes in batches of 20
- **Fallback**: Text-based scoring when models fail

### Adaptive Result Limits
```python
# Count high-importance incidents (score >= 2.0)  
high_importance_count = len([hit for hit in hits if hit.get('importance_score', 0) >= 2.0])

# Increase limit for many critical incidents
if high_importance_count > target_k:
    adaptive_target_k = min(high_importance_count + 5, len(hits))
```

### Time Decay & Diversity
- **Time Decay**: Exponential boost for recent entries (48h window)
- **Diversity**: Max 3-5 results per category to avoid dominance

## Search Actions & Data Schema

### Primitive Search Actions

#### Core Search Actions
```python
@dataclass
class TextSearchAction:
    query: str = ""
    k: int = 10
    recent: RecencyMode = RecencyMode.SCORE
    strategy: SearchStrategy = SearchStrategy.AUTO

@dataclass  
class BulkRetrieveAction:
    k: int = 100
    filters: Optional[Dict[str, List[str]]] = None
    since: Optional[str] = None
    until: Optional[str] = None
```

#### Analysis Actions
```python
@dataclass
class ExtractIncidentsAction:
    keywords: List[str]  # ["error", "fault", "trip", "down"]
    categories: List[str]  # ["Problem", "Pikett", "Safety"]
    min_severity: int = 1

@dataclass
class SmartRerankAction:
    method: str = "hybrid"  # "semantic", "llm", "hybrid"
    target_k: int = 10
    semantic_weight: float = 0.5
    llm_weight: float = 0.5
```

### SwissFEL Domain Knowledge

#### ELOG Schema
```python
@dataclass
class ElogDocument:
    doc_id: str
    title: str
    author: str  
    date: datetime
    category: str    # "Problem", "Info", "Schicht-Übergabe" (18 categories)
    domain: str      # "Injector", "Athos", "Aramis" (9 domains)  
    system: str      # "RF", "Diagnostics", "Controls" (25 systems)
    section: str     # "SATUN18", "SINSB02" (device-specific)
    body: str
    attachments: List[Dict]
```

#### Key Categories & Systems
- **Categories**: Problem (7,975), Info (12,985), Schicht-Übergabe (6,024)
- **Systems**: RF (4,275), Operation (3,857), Diagnostics (1,178) 
- **Domains**: Injector (3,147), Global (2,726), Athos (1,696)
- **Major Sections**: SINEG01 (769), SINSB02 (251), SATUN18 (device-specific)

## Performance & Timing

### Comprehensive Timing Tracking
Each workflow component tracks execution time:
```python
timing = {
    'planner': 2.5,      # LLM planning time
    'analyzer': 0.02,    # Multi-step execution  
    'evaluator': 3.1,    # Quality assessment
    'reporter': 29.4     # Synthesis & formatting
}
```

### Typical Performance
- **Simple Queries**: 5-10 seconds end-to-end
- **Complex Analysis**: 15-45 seconds (with reranking)
- **Bulk Processing**: 119 entries → 10 most relevant in ~2 seconds

### Efficiency Optimizations
- **Parallel Execution**: Multiple search actions run concurrently
- **Batch Processing**: LLM scoring in batches of 20
- **Lazy Loading**: Neural models loaded only when needed
- **Caching**: ElasticSearch query optimization

## Query Examples & Behavior

### 1. Device-Specific Query (Found)
**Input**: `"When was the last reset of SATUN18?"`

**Workflow**: 
- Planner: `general_search` → Searcher → Evaluator (accept) → Reporter
- **Result**: Comprehensive analysis with specific timestamps and context

### 2. Device-Specific Query (Not Found)  
**Input**: `"What's the status of DEVICE999?"`

**Workflow**:
- Planner: `analyze` → Analyzer (device_status plan) → Evaluator (6 refinements) → Reporter
- **Result**: Explicit "No Specific Information Found" + contextual analysis

### 3. Temporal Analysis
**Input**: `"What happened last week?"`

**Workflow**:
- Planner: `analyze` → Analyzer (temporal_analysis plan, 5 steps) → Reporter
- **Result**: Incident timeline, system impact analysis, trend identification

## Installation & Setup

### Prerequisites
```bash
# Core dependencies
pip install langgraph elasticsearch sentence-transformers
pip install numpy scikit-learn ollama

# Optional: GPU acceleration for neural models
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### ElasticSearch Setup
```bash
# Start ElasticSearch (Docker)
docker run -d --name elasticsearch \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  elasticsearch:8.11.0
```

### LLM Setup (Ollama)
```bash
# Install and run Ollama
ollama serve
ollama pull gemma2:27b  # or your preferred model
```

### Configuration
```python
# config/settings.py
ELASTICSEARCH_URL = "http://localhost:9200"
OLLAMA_BASE_URL = "http://127.0.0.1:11434"  
INDEX_NAME = "swissfel_elog"
```

## Usage

### Basic Usage
```python
from langgraph_main import build_graph

graph = build_graph()
result = graph.invoke({"user_query": "What happened to SATUN18?"})

print(result['report'].text[0])
```

### Advanced Configuration
```python
from smart_reranker import RerankConfig, SmartReranker

# Custom reranking
config = RerankConfig(
    method="hybrid",
    target_k=15,
    semantic_weight=0.6,  # Favor semantic over LLM
    llm_weight=0.4,
    max_per_category=4
)

reranker = SmartReranker(config)
reranked = reranker.rerank(hits, query, context)
```

### Plan Templates
```python
from elog_schema import create_temporal_analysis_plan

plan = create_temporal_analysis_plan(
    since="2025-08-01T00:00:00Z",
    until="2025-08-07T23:59:59Z", 
    focus="incidents"
)
```

## Key Files

| File | Purpose |
|------|---------|
| `langgraph_main.py` | Main LangGraph workflow orchestration |
| `elog_schema.py` | Domain schema, search actions, plan templates |
| `action_executor.py` | Multi-step plan execution engine |
| `smart_reranker.py` | Neural+LLM hybrid reranking system |
| `es_search.py` | ElasticSearch integration & query building |

## Troubleshooting

### Common Issues

#### 1. HuggingFace Authentication Errors
```bash
export HF_TOKEN="your_huggingface_token"
# Or disable neural reranking in config
```

#### 2. ElasticSearch Connection Failed
```bash
# Check ElasticSearch status
curl http://localhost:9200/_cluster/health
```

#### 3. Ollama Model Loading
```bash
# Verify model availability
ollama list
ollama pull gemma2:27b  # Download if missing
```

#### 4. Wildcard Query Returns 0 Results
**Fixed**: System now uses `match_all` instead of `multi_match` for "*" queries

#### 5. Neural Model Loading Failures
**Fallback**: System automatically falls back to text-based scoring when neural models fail

## Performance Tuning

### ElasticSearch Optimization
```json
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.max_result_window": 50000
  }
}
```

### Neural Model Optimization  
```python
# Use smaller models for faster inference
RerankConfig(
    model_name="cross-encoder/ms-marco-TinyBERT-L-2-v2",  # Faster
    batch_size=10  # Reduce for memory constraints
)
```

### LLM Optimization
```python
# Reduce context for faster processing
llm_config = {
    "temperature": 0.1,
    "max_tokens": 200,  # Shorter responses
    "timeout": 30
}
```

## Contributing

### Adding New Query Types
1. Add query type to `detect_query_type()` in `elog_schema.py`
2. Create plan template function (e.g., `create_comparison_plan()`)  
3. Update planner prompt in `langgraph_main.py`
4. Test with representative queries

### Adding New Search Actions
1. Define dataclass in `elog_schema.py`
2. Implement execution logic in `action_executor.py`
3. Add to plan templates as needed
4. Update documentation

### Adding New Analysis Methods
1. Create new analysis action class
2. Implement in action executor
3. Add to multi-step plans  
4. Test with real queries

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- **SwissFEL**: Accelerator facility data and domain expertise
- **LangGraph**: Workflow orchestration framework
- **sentence-transformers**: Neural reranking models
- **ElasticSearch**: Full-text search and indexing