"""
Smart Bulk Reranker for ELOG Search Results
==========================================

Efficiently reranks large sets of search results using semantic similarity,
lightweight LLM calls, or hybrid approaches.
"""

import logging
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

# Cross-encoder for neural reranking
try:
    from sentence_transformers import CrossEncoder
    CROSSENCODER_AVAILABLE = True
except ImportError:
    CrossEncoder = None
    CROSSENCODER_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class RerankConfig:
    """Configuration for reranking"""
    method: str = "hybrid"  # "semantic", "llm", "hybrid"
    target_k: int = 10      # Final number of results
    
    # Semantic config - using cross-encoder for proper reranking
    model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"  # Smaller neural reranker
    semantic_weight: float = 0.5
    
    # LLM config  
    llm_weight: float = 0.5
    batch_size: int = 20    # Process in batches for efficiency
    
    # Diversity
    max_per_category: Optional[int] = 3
    time_decay_hours: float = 48.0  # Boost recent entries

class SmartReranker:
    """Generic smart reranker for bulk search results"""
    
    def __init__(self, config: RerankConfig = None):
        self.config = config or RerankConfig()
        self.semantic_model = None
        
        # Lazy load semantic model only when needed
        if self.config.method in ["semantic", "hybrid"]:
            self._load_semantic_model()
    
    def _load_semantic_model(self):
        """Load cross-encoder model for neural reranking"""
        if self.semantic_model is None:
            try:
                if CROSSENCODER_AVAILABLE and "cross-encoder" in self.config.model_name:
                    logger.info(f"Loading cross-encoder reranker: {self.config.model_name}")
                    
                    # Try to load the cross-encoder model
                    logger.info("Attempting to load cross-encoder without authentication")
                    self.semantic_model = CrossEncoder(self.config.model_name)
                    
                    self.is_cross_encoder = True
                else:
                    # Fallback to simple text-based scoring
                    logger.info("Cross-encoder not available, falling back to text-based scoring")
                    self.semantic_model = "fallback"
                    self.is_cross_encoder = False
                    
            except Exception as e:
                logger.warning(f"Failed to load cross-encoder {self.config.model_name}: {e}")
                logger.info("Falling back to simple text-based scoring")
                self.semantic_model = "fallback"
                self.is_cross_encoder = False
    
    def rerank(self, 
               hits: List[Dict[str, Any]], 
               query: str, 
               context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Rerank hits based on relevance to query and context
        
        Args:
            hits: List of search result dictionaries
            query: Original user query
            context: Additional context (query_type, plan_type, etc.)
            
        Returns:
            Reranked list of top-k hits
        """
        if not hits:
            return []
            
        context = context or {}
        logger.info(f"Reranking {len(hits)} hits using {self.config.method} method")
        
        if self.config.method == "semantic":
            return self._semantic_rerank(hits, query, context)
        elif self.config.method == "llm":
            return self._llm_rerank(hits, query, context)
        elif self.config.method == "hybrid":
            return self._hybrid_rerank(hits, query, context)
        else:
            raise ValueError(f"Unknown reranking method: {self.config.method}")
    
    def _semantic_rerank(self, hits: List[Dict], query: str, context: Dict) -> List[Dict]:
        """Semantic similarity-based reranking"""
        
        if self.semantic_model == "fallback":
            # Simple text-based scoring fallback
            return self._fallback_text_rerank(hits, query, context)
        
        # Combine title + snippet for better semantic matching
        texts = []
        for hit in hits:
            text = f"{hit.get('title', '')} {hit.get('snippet', '')}"
            # Add category/system for domain context
            if hit.get('category'):
                text += f" {hit.get('category')}"
            if hit.get('system'):  
                text += f" {hit.get('system')}"
            texts.append(text)
        
        # Use CrossEncoder if available (best for reranking)
        if hasattr(self, 'is_cross_encoder') and self.is_cross_encoder:
            # CrossEncoder expects query-passage pairs
            query_passage_pairs = [[query, text] for text in texts]
            scores = self.semantic_model.predict(query_passage_pairs)
            
            # Ensure scores is a numpy array
            if not isinstance(scores, np.ndarray):
                scores = np.array(scores)
        else:
            # Standard SentenceTransformer approach
            query_embedding = self.semantic_model.encode([query])
            hit_embeddings = self.semantic_model.encode(texts)
            scores = cosine_similarity(query_embedding, hit_embeddings)[0]
        
        # Apply time decay if timestamps available
        scores = self._apply_time_decay(hits, np.array(scores))
        
        # Sort by score and return top-k
        scored_hits = list(zip(hits, scores))
        scored_hits.sort(key=lambda x: x[1], reverse=True)
        
        return self._apply_diversity([hit for hit, _ in scored_hits])
    
    def _fallback_text_rerank(self, hits: List[Dict], query: str, context: Dict) -> List[Dict]:
        """Simple text-based scoring when semantic models aren't available"""
        
        query_words = set(query.lower().split())
        scored_hits = []
        
        for hit in hits:
            # Combine text fields
            text = f"{hit.get('title', '')} {hit.get('snippet', '')}".lower()
            text_words = set(text.split())
            
            # Simple word overlap score
            overlap_score = len(query_words.intersection(text_words)) / max(len(query_words), 1)
            
            # Boost for problem/incident categories
            category_boost = 1.5 if hit.get('category') in ['Problem', 'Pikett'] else 1.0
            
            final_score = overlap_score * category_boost
            scored_hits.append((hit, final_score))
        
        # Apply time decay
        base_scores = np.array([score for _, score in scored_hits])
        time_adjusted_scores = self._apply_time_decay(hits, base_scores)
        
        # Re-sort with time-adjusted scores
        final_scored = [(hit, score) for (hit, _), score in zip(scored_hits, time_adjusted_scores)]
        final_scored.sort(key=lambda x: x[1], reverse=True)
        
        return self._apply_diversity([hit for hit, _ in final_scored])
    
    def _llm_rerank(self, hits: List[Dict], query: str, context: Dict) -> List[Dict]:
        """LLM-based lightweight reranking"""
        
        # Process in batches for efficiency
        all_scored = []
        
        for i in range(0, len(hits), self.config.batch_size):
            batch = hits[i:i + self.config.batch_size]
            batch_scores = self._llm_score_batch(batch, query, context)
            all_scored.extend(batch_scores)
        
        # Sort by LLM scores
        all_scored.sort(key=lambda x: x[1], reverse=True)
        
        return self._apply_diversity([hit for hit, _ in all_scored])
    
    def _llm_score_batch(self, batch: List[Dict], query: str, context: Dict) -> List[Tuple[Dict, float]]:
        """Score a batch of hits using LLM"""
        
        # Create lightweight prompt with just key fields
        items = []
        for i, hit in enumerate(batch):
            item = {
                "id": i,
                "title": hit.get('title', '')[:100],  # Truncate for efficiency
                "category": hit.get('category', ''),
                "system": hit.get('system', ''),
                "timestamp": hit.get('timestamp', ''),
                "snippet": hit.get('snippet', '')[:200]  # Truncate snippet
            }
            items.append(item)
        
        query_type = context.get('query_type', 'general')
        
        prompt = f"""Rate the relevance of each entry to the user query on a scale of 1-10.
        
User Query: "{query}"
Query Type: {query_type}

Entries to rate:
{json.dumps(items, indent=2)}

Return ONLY a JSON object mapping id to score:
{{"0": 8.5, "1": 6.2, "2": 9.1, ...}}"""

        try:
            from es_search import call_llm_prompt_ollama  # Import your LLM function
            response = call_llm_prompt_ollama(
                "Rate relevance 1-10 for each entry. Return only JSON.",
                prompt,
                temperature=0.1,
                max_tokens=200
            )
            
            scores_dict = json.loads(response)
            scored_batch = []
            
            for i, hit in enumerate(batch):
                score = float(scores_dict.get(str(i), 5.0))  # Default 5.0 if missing
                scored_batch.append((hit, score))
            
            return scored_batch
            
        except Exception as e:
            logger.warning(f"LLM scoring failed: {e}, using neutral scores")
            return [(hit, 5.0) for hit in batch]
    
    def _hybrid_rerank(self, hits: List[Dict], query: str, context: Dict) -> List[Dict]:
        """Hybrid semantic + LLM reranking"""
        
        # Get semantic scores
        semantic_hits = self._semantic_rerank(hits, query, context)
        semantic_scores = {id(hit): i for i, hit in enumerate(reversed(semantic_hits))}
        
        # Sample top semantic candidates for LLM scoring (efficiency)
        top_candidates = semantic_hits[:min(len(hits), 30)]  # Top 30 for LLM
        
        if len(top_candidates) <= self.config.target_k:
            return top_candidates[:self.config.target_k]
        
        # LLM rerank the top candidates
        llm_scores = self._llm_score_batch(top_candidates, query, context)
        
        # Combine scores
        final_scores = []
        for hit, llm_score in llm_scores:
            semantic_rank = semantic_scores.get(id(hit), 0)
            semantic_score = semantic_rank / len(hits)  # Normalize
            
            combined_score = (
                self.config.semantic_weight * semantic_score + 
                self.config.llm_weight * (llm_score / 10.0)  # Normalize LLM score
            )
            final_scores.append((hit, combined_score))
        
        # Sort by combined score
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return self._apply_diversity([hit for hit, _ in final_scores])
    
    def _apply_time_decay(self, hits: List[Dict], base_scores: np.ndarray) -> np.ndarray:
        """Apply time decay to boost recent entries"""
        
        from datetime import datetime, timezone
        
        try:
            now = datetime.now(timezone.utc)
            decay_scores = base_scores.copy()
            
            for i, hit in enumerate(hits):
                timestamp_str = hit.get('timestamp')
                if timestamp_str:
                    # Parse timestamp
                    timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                    hours_ago = (now - timestamp).total_seconds() / 3600
                    
                    # Exponential decay: boost recent entries
                    decay_factor = np.exp(-hours_ago / self.config.time_decay_hours)
                    decay_scores[i] *= (1.0 + 0.5 * decay_factor)  # Up to 50% boost for very recent
            
            return decay_scores
            
        except Exception as e:
            logger.warning(f"Time decay failed: {e}")
            return base_scores
    
    def _apply_diversity(self, hits: List[Dict]) -> List[Dict]:
        """Apply diversity constraints to avoid category/system dominance"""
        
        if not self.config.max_per_category:
            return hits[:self.config.target_k]
        
        diversified = []
        category_counts = {}
        
        for hit in hits:
            if len(diversified) >= self.config.target_k:
                break
                
            category = hit.get('category', 'Unknown')
            count = category_counts.get(category, 0)
            
            if count < self.config.max_per_category:
                diversified.append(hit)
                category_counts[category] = count + 1
            elif len(diversified) < self.config.target_k:
                # If we still need more results, relax constraints
                diversified.append(hit)
        
        return diversified[:self.config.target_k]

# Convenience functions for common use cases
def quick_semantic_rerank(hits: List[Dict], query: str, top_k: int = 10) -> List[Dict]:
    """Quick semantic reranking"""
    config = RerankConfig(method="semantic", target_k=top_k)
    reranker = SmartReranker(config)
    return reranker.rerank(hits, query)

def quick_hybrid_rerank(hits: List[Dict], query: str, context: Dict = None, top_k: int = 10) -> List[Dict]:
    """Quick hybrid reranking with context"""
    config = RerankConfig(method="hybrid", target_k=top_k)
    reranker = SmartReranker(config)
    return reranker.rerank(hits, query, context or {})


if __name__ == "__main__":
    # Example usage
    sample_hits = [{"id": "1", "text": "Sample document 1"}, {"id": "2", "text": "Sample document 2"}]
    sample_query = "Sample query"
    print(quick_semantic_rerank(sample_hits, sample_query))
