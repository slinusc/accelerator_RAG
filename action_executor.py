"""
Action Executor for ElasticSearch Primitive Actions
==================================================

This module executes the primitive search and analysis actions defined in elog_schema.py.
It provides a unified interface for running complex multi-step search plans.
"""

import re
from typing import List, Dict, Any, Optional, Union
from datetime import datetime, timezone, timedelta
from dataclasses import asdict
import logging
from colorama import Fore, Style

# Import ElasticSearch functions
from es_search import search as es_search, build_query, es

# Import schema definitions  
from elog_schema import (
    SearchAction, TextSearchAction, TemporalSearchAction, FilteredSearchAction,
    BulkRetrieveAction, LatestEntriesAction, ExtractIncidentsAction,
    RankByImportanceAction, AggregateByMetadataAction, TimeSeriesAnalysisAction,
    SmartRerankAction, SearchPlan, SearchHit, RecencyMode, SearchStrategy
)

logger = logging.getLogger(__name__)

class ActionExecutor:
    """Executes search actions and analysis steps"""
    
    def __init__(self, es_client=None):
        self.es = es_client or es
        self.results_cache = {}  # Cache intermediate results
        
    def execute_action(self, action: Union[SearchAction, Any]) -> List[Dict[str, Any]]:
        """Execute a single action and return results"""
        logger.info(f"{Fore.BLUE}[ActionExecutor]{Style.RESET_ALL} Executing action: {type(action).__name__}")
        
        if isinstance(action, TextSearchAction):
            return self._execute_text_search(action)
        elif isinstance(action, TemporalSearchAction):
            return self._execute_temporal_search(action)
        elif isinstance(action, FilteredSearchAction):
            return self._execute_filtered_search(action)
        elif isinstance(action, BulkRetrieveAction):
            return self._execute_bulk_retrieve(action)
        elif isinstance(action, LatestEntriesAction):
            return self._execute_latest_entries(action)
        elif isinstance(action, ExtractIncidentsAction):
            return self._execute_extract_incidents(action)
        elif isinstance(action, RankByImportanceAction):
            return self._execute_rank_importance(action)
        elif isinstance(action, AggregateByMetadataAction):
            return self._execute_aggregate_metadata(action)
        elif isinstance(action, TimeSeriesAnalysisAction):
            return self._execute_timeseries_analysis(action)
        elif isinstance(action, SmartRerankAction):
            return self._execute_smart_rerank(action)
        else:
            raise ValueError(f"Unknown action type: {type(action)}")
    
    def execute_plan(self, plan: SearchPlan, original_query: str = "", context: Dict = None) -> Dict[str, Any]:
        """Execute a complete multi-step search plan"""
        logger.info(f"{Fore.GREEN}[ActionExecutor]{Style.RESET_ALL} Executing plan: {plan.plan_type}")
        logger.info(f"{Fore.GREEN}[ActionExecutor]{Style.RESET_ALL} Plan description: {plan.description}")
        
        results = {
            "plan_type": plan.plan_type,
            "description": plan.description,
            "steps": [],
            "hits": [],
            "analysis": {},
            "synthesis": plan.synthesis_prompt
        }
        
        # Cache context for reranking
        self.results_cache['original_query'] = original_query
        self.results_cache['plan_type'] = plan.plan_type
        if context:
            self.results_cache.update(context)
        
        current_hits = []
        
        for i, step in enumerate(plan.steps):
            logger.info(f"{Fore.YELLOW}[ActionExecutor]{Style.RESET_ALL} Executing step {i+1}/{len(plan.steps)}: {step.action_type}")
            
            try:
                if hasattr(step, 'action_type') and step.action_type.startswith('extract_') or \
                   step.action_type.startswith('rank_') or step.action_type.startswith('aggregate_') or \
                   step.action_type.startswith('timeseries_'):
                    # Analysis actions work on previous results
                    step_results = self.execute_action(step)
                    if step.action_type.startswith('extract_') or step.action_type.startswith('rank_'):
                        current_hits = step_results  # Update current hits
                    else:
                        results["analysis"][step.action_type] = step_results
                else:
                    # Search actions
                    step_results = self.execute_action(step)
                    current_hits.extend(step_results)
                    # Cache results for next step
                    self.results_cache['current_hits'] = current_hits
                
                results["steps"].append({
                    "step": i + 1,
                    "action": step.action_type,
                    "results_count": len(step_results) if isinstance(step_results, list) else 1,
                    "success": True
                })
                logger.info(f"{Fore.YELLOW}[ActionExecutor]{Style.RESET_ALL} Step {i+1} results count: {len(step_results) if isinstance(step_results, list) else 1}")
                
            except Exception as e:
                logger.error(f"Step {i+1} failed: {e}")
                results["steps"].append({
                    "step": i + 1,
                    "action": step.action_type,
                    "error": str(e),
                    "success": False
                })
        
        # Deduplicate hits by elog_id
        seen_ids = set()
        unique_hits = []
        for hit in current_hits:
            elog_id = hit.get('elog_id') or hit.get('id', '')
            if elog_id and elog_id not in seen_ids:
                seen_ids.add(elog_id)
                unique_hits.append(hit)
        
        results["hits"] = unique_hits
        logger.info(f"{Fore.GREEN}[ActionExecutor]{Style.RESET_ALL} Plan execution completed. Total unique hits: {len(unique_hits)}")
        return results
    
    # ========================================================================
    # SEARCH ACTION IMPLEMENTATIONS
    # ========================================================================
    
    def _execute_text_search(self, action: TextSearchAction) -> List[Dict[str, Any]]:
        """Execute keyword-based text search"""
        return es_search(
            user_q=action.query,
            k=action.k,
            recent=action.recent.value,
            strategy=action.strategy.value
        )
    
    def _execute_temporal_search(self, action: TemporalSearchAction) -> List[Dict[str, Any]]:
        """Execute time-range search"""
        return es_search(
            user_q=action.query or "*",
            k=action.k,
            since=action.since,
            until=action.until,
            recent="sort"  # Chronological for temporal searches
        )
    
    def _execute_filtered_search(self, action: FilteredSearchAction) -> List[Dict[str, Any]]:
        """Execute search with metadata filters"""
        filters = {}
        if action.categories:
            filters["category"] = action.categories
        if action.systems:
            filters["system"] = action.systems  
        if action.domains:
            filters["domain"] = action.domains
        if action.sections:
            filters["section"] = action.sections
            
        return es_search(
            user_q=action.query,
            k=action.k,
            filters=filters
        )
    
    def _execute_bulk_retrieve(self, action: BulkRetrieveAction) -> List[Dict[str, Any]]:
        """Execute bulk retrieval for analysis"""
        logger.info(f"{Fore.CYAN}[BulkRetrieve]{Style.RESET_ALL} Searching with filters: {action.filters}, since: {action.since}, until: {action.until}, k: {action.k}")
        
        results = es_search(
            user_q="*",  # Match all if no query
            k=action.k,
            filters=action.filters or {},
            since=action.since,
            until=action.until,
            recent="sort"
        )
        
        logger.info(f"{Fore.CYAN}[BulkRetrieve]{Style.RESET_ALL} Found {len(results)} total entries in time range")
        
        # If no results in time range, try broader search
        if not results and action.since and action.until:
            logger.info(f"[BulkRetrieve] No results in specified time range, trying broader search...")
            from datetime import datetime, timedelta
            
            # Try last 30 days instead
            now = datetime.now()
            broader_since = (now - timedelta(days=30)).isoformat()
            
            broader_results = es_search(
                user_q="*",
                k=min(action.k, 100),  # Limit broader search
                filters=action.filters or {},
                since=broader_since,
                until=None,
                recent="sort"
            )
            logger.info(f"[BulkRetrieve] Broader search found {len(broader_results)} entries (last 30 days)")
            results = broader_results
        
        if results:
            categories = {}
            for hit in results:
                cat = hit.get('category', 'Unknown')
                categories[cat] = categories.get(cat, 0) + 1
            logger.info(f"{Fore.CYAN}[BulkRetrieve]{Style.RESET_ALL} Categories found: {dict(sorted(categories.items(), key=lambda x: x[1], reverse=True))}")
        return results
    
    def _execute_latest_entries(self, action: LatestEntriesAction) -> List[Dict[str, Any]]:
        """Execute latest entries search"""
        return es_search(
            user_q=action.query or "*",
            k=action.k,
            filters=action.filters,
            recent="sort",
            strategy="latest"
        )
    
    # ========================================================================
    # ANALYSIS ACTION IMPLEMENTATIONS  
    # ========================================================================
    
    def _execute_extract_incidents(self, action: ExtractIncidentsAction) -> List[Dict[str, Any]]:
        """Filter results to extract incident-related entries"""
        # Use current results from cache or previous step
        current_hits = self.results_cache.get('current_hits', [])
        
        logger.info(f"[ExtractIncidents] Processing {len(current_hits)} entries")
        
        # Default incident keywords
        keywords = action.keywords or [
            "error", "failure", "problem", "alert", "fault", "down", "trip",
            "issue", "broken", "failed", "emergency", "critical", "urgent",
            "beam", "dump", "abort", "interlock", "alarm", "warning"
        ]
        
        # Default incident categories  
        categories = action.categories or ["Problem", "Pikett", "Safety"]
        
        logger.info(f"[ExtractIncidents] Looking for categories: {categories}")
        logger.info(f"[ExtractIncidents] Using keywords: {keywords[:5]}... ({len(keywords)} total)")
        
        incidents = []
        category_matches = 0
        keyword_matches = 0
        
        for hit in current_hits:
            # Check category first (higher priority)
            if hit.get('category') in categories:
                incidents.append(hit)
                category_matches += 1
                logger.debug(f"[ExtractIncidents] Category match: {hit.get('category')} - {hit.get('title', '')[:50]}")
                continue
                
            # Check keywords in title and snippet
            text_to_check = f"{hit.get('title', '')} {hit.get('snippet', '')}".lower()
            matched_keywords = [kw for kw in keywords if kw in text_to_check]
            if matched_keywords:
                incidents.append(hit)
                keyword_matches += 1
                logger.debug(f"[ExtractIncidents] Keyword match ({matched_keywords[0]}): {hit.get('title', '')[:50]}")
        
        # Sort by timestamp (newest first)
        incidents.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        logger.info(f"[ExtractIncidents] Found {len(incidents)} incidents ({category_matches} by category, {keyword_matches} by keywords)")
        
        self.results_cache['current_hits'] = incidents
        return incidents
    
    def _execute_rank_importance(self, action: RankByImportanceAction) -> List[Dict[str, Any]]:
        """Rank results by importance/criticality"""
        current_hits = self.results_cache.get('current_hits', [])
        
        # Importance keywords with weights
        criteria = action.criteria or ["emergency", "critical", "urgent", "safety", "beam", "down"]
        boost_categories = action.boost_categories or ["Problem", "Safety", "Pikett"] 
        boost_systems = action.boost_systems or ["Safety", "RF", "Controls", "Operation"]
        
        def calculate_importance_score(hit: Dict[str, Any]) -> float:
            score = hit.get('score', 0.0)
            
            # Category boost
            if hit.get('category') in boost_categories:
                score *= 1.5
                
            # System boost  
            if hit.get('system') in boost_systems:
                score *= 1.3
            
            # Keyword boost
            text_to_check = f"{hit.get('title', '')} {hit.get('snippet', '')}".lower()
            for criterion in criteria:
                if criterion in text_to_check:
                    score *= 1.2
            
            return score
        
        # Add importance scores and sort
        for hit in current_hits:
            hit['importance_score'] = calculate_importance_score(hit)
            
        ranked_hits = sorted(current_hits, key=lambda x: x.get('importance_score', 0), reverse=True)
        
        self.results_cache['current_hits'] = ranked_hits
        return ranked_hits
    
    def _execute_aggregate_metadata(self, action: AggregateByMetadataAction) -> Dict[str, Any]:
        """Aggregate results by metadata fields"""
        current_hits = self.results_cache.get('current_hits', [])
        
        aggregations = {}
        
        for field in action.group_by:
            if field in ['category', 'system', 'domain', 'section', 'author']:
                field_counts = {}
                for hit in current_hits:
                    value = hit.get(field, 'Unknown')
                    field_counts[value] = field_counts.get(value, 0) + 1
                
                # Sort by count descending
                aggregations[field] = dict(sorted(field_counts.items(), key=lambda x: x[1], reverse=True))
        
        # Time bucketing if requested
        if action.time_bucket and current_hits:
            time_buckets = {}
            bucket_format = {
                "hour": "%Y-%m-%d %H:00",
                "day": "%Y-%m-%d", 
                "week": "%Y-W%U",
                "month": "%Y-%m"
            }
            
            fmt = bucket_format.get(action.time_bucket, "%Y-%m-%d")
            
            for hit in current_hits:
                timestamp = hit.get('timestamp', '')
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                        bucket = dt.strftime(fmt)
                        time_buckets[bucket] = time_buckets.get(bucket, 0) + 1
                    except:
                        continue
                        
            aggregations['time_buckets'] = dict(sorted(time_buckets.items()))
        
        return aggregations
    
    def _execute_timeseries_analysis(self, action: TimeSeriesAnalysisAction) -> Dict[str, Any]:
        """Analyze temporal patterns in the data"""
        current_hits = self.results_cache.get('current_hits', [])
        
        # Group by time buckets
        time_series = {}
        
        bucket_formats = {
            "1h": "%Y-%m-%d %H:00",
            "1d": "%Y-%m-%d",
            "1w": "%Y-W%U", 
            "1M": "%Y-%m"
        }
        
        fmt = bucket_formats.get(action.bucket_size, "%Y-%m-%d")
        
        for hit in current_hits:
            timestamp = hit.get('timestamp', '')
            if not timestamp:
                continue
                
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                bucket = dt.strftime(fmt)
                
                if bucket not in time_series:
                    time_series[bucket] = {
                        "count": 0,
                        "incidents": 0,
                        "systems_affected": set(),
                        "categories": {}
                    }
                
                time_series[bucket]["count"] += 1
                
                # Count incidents
                if hit.get('category') in ['Problem', 'Pikett']:
                    time_series[bucket]["incidents"] += 1
                
                # Track systems  
                system = hit.get('system', '')
                if system:
                    time_series[bucket]["systems_affected"].add(system)
                    
                # Track categories
                category = hit.get('category', 'Unknown')
                time_series[bucket]["categories"][category] = time_series[bucket]["categories"].get(category, 0) + 1
                
            except Exception as e:
                logger.warning(f"Could not parse timestamp {timestamp}: {e}")
                continue
        
        # Convert sets to counts and sort
        for bucket_data in time_series.values():
            bucket_data["systems_affected"] = len(bucket_data["systems_affected"])
            
        sorted_series = dict(sorted(time_series.items()))
        
        return {
            "bucket_size": action.bucket_size,
            "metrics": action.metrics or ["count", "incidents", "systems_affected"],
            "time_series": sorted_series,
            "summary": {
                "total_buckets": len(sorted_series),
                "total_entries": sum(b["count"] for b in sorted_series.values()),
                "total_incidents": sum(b["incidents"] for b in sorted_series.values()),
                "peak_activity": max(sorted_series.items(), key=lambda x: x[1]["count"]) if sorted_series else None
            }
        }
    
    def _execute_smart_rerank(self, action: SmartRerankAction) -> List[Dict[str, Any]]:
        """Execute smart reranking of current results"""
        current_hits = self.results_cache.get('current_hits', [])
        
        if not current_hits:
            logger.warning("[SmartRerank] No hits to rerank")
            return []
        
        logger.info(f"[SmartRerank] Reranking {len(current_hits)} hits using {action.method} method, target_k={action.target_k}")
        
        try:
            from smart_reranker import SmartReranker, RerankConfig
            
            config = RerankConfig(
                method=action.method,
                target_k=action.target_k,
                max_per_category=action.max_per_category
            )
            
            reranker = SmartReranker(config)
            
            # Get context from cache if available
            context = {
                "query_type": self.results_cache.get('query_type', 'general'),
                "plan_type": self.results_cache.get('plan_type', 'unknown')
            }
            
            # Use original query from cache
            original_query = self.results_cache.get('original_query', '')
            
            reranked_hits = reranker.rerank(current_hits, original_query, context)
            
            logger.info(f"[SmartRerank] Reranked from {len(current_hits)} to {len(reranked_hits)} hits")
            
            # Update cache with reranked results
            self.results_cache['current_hits'] = reranked_hits
            
            return reranked_hits
            
        except Exception as e:
            logger.error(f"[SmartRerank] Error during reranking: {e}")
            # Fallback: return top-k by current order
            fallback_hits = current_hits[:action.target_k]
            self.results_cache['current_hits'] = fallback_hits
            return fallback_hits


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_temporal_analysis(since: str, until: str, focus: str = "incidents") -> Dict[str, Any]:
    """Quick function for temporal analysis"""
    from elog_schema import create_temporal_analysis_plan
    
    executor = ActionExecutor()
    plan = create_temporal_analysis_plan(since, until, focus)
    return executor.execute_plan(plan)

def quick_device_status(device_pattern: str) -> Dict[str, Any]:
    """Quick function for device status"""
    from elog_schema import create_device_status_plan
    
    executor = ActionExecutor()
    plan = create_device_status_plan(device_pattern)
    return executor.execute_plan(plan)

def quick_system_health(system: str, days: int = 30) -> Dict[str, Any]:
    """Quick function for system health analysis"""  
    from elog_schema import create_system_health_plan
    
    since = (datetime.now() - timedelta(days=days)).isoformat()
    executor = ActionExecutor()
    plan = create_system_health_plan(system, f"last {days} days")
    
    # Modify plan to add time filter
    for step in plan.steps:
        if hasattr(step, 'since'):
            step.since = since
            
    return executor.execute_plan(plan)