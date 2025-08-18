import asyncio
import structlog
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum

from src.data.elog_client import ElogClient, ElogEntry
from src.data.knowledge_graph import KnowledgeGraphClient
from src.models.ollama_client import OllamaRAGGenerator
from config.settings import settings

logger = structlog.get_logger(__name__)


class QueryType(Enum):
    """Types of queries the RAG system can handle."""
    GENERAL = "general"
    OPERATIONAL = "operational" 
    TROUBLESHOOTING = "troubleshooting"
    MACHINE_SPECIFIC = "machine_specific"
    PROCEDURE = "procedure"
    HISTORICAL = "historical"


@dataclass
class RAGContext:
    """Context information for RAG processing."""
    query: str
    query_type: QueryType
    elog_entries: List[ElogEntry]
    knowledge_chunks: List[Dict[str, Any]]
    total_sources: int
    confidence_score: float
    processing_time: float


@dataclass
class RAGResponse:
    """Complete response from RAG system."""
    answer: str
    context: RAGContext
    follow_up_questions: List[str]
    sources: List[Dict[str, str]]
    timestamp: datetime


class AgenticRAGOrchestrator:
    """
    Main orchestrator for the agentic RAG system that coordinates
    between ELOG, knowledge graph, and LLM components.
    """
    
    def __init__(
        self,
        elog_client: ElogClient = None,
        knowledge_client: KnowledgeGraphClient = None,
        llm_generator: OllamaRAGGenerator = None
    ):
        self.elog_client = elog_client or ElogClient()
        self.knowledge_client = knowledge_client or KnowledgeGraphClient()
        self.llm_generator = llm_generator or OllamaRAGGenerator()
        
        # Query classification keywords
        self.query_classifiers = {
            QueryType.OPERATIONAL: [
                "startup", "shutdown", "operation", "run", "beam", "current",
                "voltage", "settings", "configuration", "parameters"
            ],
            QueryType.TROUBLESHOOTING: [
                "problem", "issue", "error", "fault", "broken", "failure",
                "troubleshoot", "debug", "fix", "repair", "malfunction"
            ],
            QueryType.MACHINE_SPECIFIC: [
                "undulator", "linac", "injector", "laser", "rf", "magnet",
                "diagnostic", "bpm", "screen", "spectrometer", "cathode"
            ],
            QueryType.PROCEDURE: [
                "procedure", "protocol", "how to", "step", "manual",
                "instruction", "guideline", "process", "sop"
            ],
            QueryType.HISTORICAL: [
                "last", "previous", "yesterday", "week", "month", "history",
                "when", "recent", "latest", "past"
            ]
        }
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify the query type based on keywords and context.
        
        Args:
            query: User query string
            
        Returns:
            Detected query type
        """
        query_lower = query.lower()
        scores = {}
        
        for query_type, keywords in self.query_classifiers.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[query_type] = score
        
        if scores:
            # Return the type with highest score
            return max(scores, key=scores.get)
        else:
            return QueryType.GENERAL
    
    async def retrieve_elog_context(
        self,
        query: str,
        query_type: QueryType,
        max_entries: int = 5
    ) -> List[ElogEntry]:
        """
        Retrieve relevant ELOG entries based on query and type.
        
        Args:
            query: User query
            query_type: Classified query type
            max_entries: Maximum number of entries to retrieve
            
        Returns:
            List of relevant ELOG entries
        """
        try:
            if query_type == QueryType.HISTORICAL:
                # For historical queries, get recent entries
                entries = self.elog_client.get_recent_entries(days=30)
                return entries[:max_entries]
            elif query_type == QueryType.TROUBLESHOOTING:
                # Search for error-related entries
                error_terms = ["error", "fault", "problem", "issue", "failure"]
                all_entries = []
                for term in error_terms:
                    try:
                        entries = self.elog_client.search_entries(term, max_results=max_entries)
                        all_entries.extend(entries)
                    except Exception as e:
                        logger.warning("ELOG search failed", term=term, error=str(e))
                
                # Remove duplicates and limit
                seen_ids = set()
                unique_entries = []
                for entry in all_entries:
                    if entry.id not in seen_ids:
                        unique_entries.append(entry)
                        seen_ids.add(entry.id)
                
                return unique_entries[:max_entries]
            else:
                # General text search
                entries = self.elog_client.search_entries(query, max_results=max_entries)
                return entries
                
        except Exception as e:
            logger.error("Failed to retrieve ELOG context", error=str(e))
            return []
    
    async def retrieve_knowledge_context(
        self,
        query: str,
        query_type: QueryType,
        max_chunks: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant knowledge graph context.
        
        Args:
            query: User query
            query_type: Classified query type
            max_chunks: Maximum number of chunks to retrieve
            
        Returns:
            List of relevant knowledge chunks
        """
        try:
            if query_type == QueryType.MACHINE_SPECIFIC:
                # Extract machine keywords and search
                machine_keywords = []
                for keyword in self.query_classifiers[QueryType.MACHINE_SPECIFIC]:
                    if keyword in query.lower():
                        machine_keywords.append(keyword)
                
                if machine_keywords:
                    machine_results = self.knowledge_client.search_by_machine_type(
                        machine_keywords, limit=max_chunks//2
                    )
                    vector_results = self.knowledge_client.search_similar_content(
                        query, limit=max_chunks//2, score_threshold=0.6
                    )
                    
                    # Combine results
                    all_results = vector_results + machine_results
                    return all_results[:max_chunks]
            
            elif query_type == QueryType.PROCEDURE:
                # Search for procedures specifically
                procedure_results = self.knowledge_client.search_procedures_and_guidelines(
                    query, limit=max_chunks
                )
                return procedure_results
            
            else:
                # General vector search
                results = self.knowledge_client.search_similar_content(
                    query, limit=max_chunks, score_threshold=0.7
                )
                return results
                
        except Exception as e:
            logger.error("Failed to retrieve knowledge context", error=str(e))
            return []
    
    def combine_contexts(
        self,
        elog_entries: List[ElogEntry],
        knowledge_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Combine ELOG and knowledge graph contexts into unified format.
        
        Args:
            elog_entries: ELOG entries
            knowledge_chunks: Knowledge graph chunks
            
        Returns:
            Combined context chunks
        """
        combined = []
        
        # Add ELOG entries
        for entry in elog_entries:
            combined.append({
                "text": f"ELOG Entry #{entry.id} - {entry.subject}\n\n{entry.text}",
                "title": f"ELOG: {entry.subject}",
                "url": entry.url,
                "source_type": "elog",
                "date": entry.date.isoformat(),
                "author": entry.author,
                "chunk_id": f"elog_{entry.id}",
                "score": 1.0  # ELOG entries get full score
            })
        
        # Add knowledge graph chunks
        for chunk in knowledge_chunks:
            chunk_copy = chunk.copy()
            chunk_copy["source_type"] = "knowledge_graph"
            if "score" not in chunk_copy:
                chunk_copy["score"] = 0.8  # Default score for knowledge chunks
            combined.append(chunk_copy)
        
        # Sort by score descending
        combined.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        return combined
    
    def calculate_confidence(self, context_chunks: List[Dict[str, Any]]) -> float:
        """
        Calculate confidence score based on context quality.
        
        Args:
            context_chunks: Retrieved context chunks
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not context_chunks:
            return 0.0
        
        # Base score from number of sources
        source_score = min(len(context_chunks) / 10.0, 1.0)
        
        # Score from similarity scores
        similarity_scores = [chunk.get("score", 0) for chunk in context_chunks]
        avg_similarity = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        
        # Score from source diversity
        source_types = set(chunk.get("source_type", "unknown") for chunk in context_chunks)
        diversity_score = len(source_types) / 2.0  # Max 2 source types (elog, knowledge_graph)
        
        # Combined confidence
        confidence = (source_score * 0.4 + avg_similarity * 0.4 + diversity_score * 0.2)
        return min(confidence, 1.0)
    
    async def process_query(self, query: str) -> RAGResponse:
        """
        Process a user query through the complete RAG pipeline.
        
        Args:
            query: User query string
            
        Returns:
            Complete RAG response
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Classify query
            query_type = self.classify_query(query)
            logger.info("Query classified", query=query, type=query_type.value)
            
            # Step 2: Retrieve contexts in parallel
            elog_task = self.retrieve_elog_context(query, query_type)
            knowledge_task = self.retrieve_knowledge_context(query, query_type)
            
            elog_entries, knowledge_chunks = await asyncio.gather(
                elog_task, knowledge_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(elog_entries, Exception):
                logger.error("ELOG retrieval failed", error=str(elog_entries))
                elog_entries = []
            
            if isinstance(knowledge_chunks, Exception):
                logger.error("Knowledge retrieval failed", error=str(knowledge_chunks))
                knowledge_chunks = []
            
            # Step 3: Combine contexts
            combined_context = self.combine_contexts(elog_entries, knowledge_chunks)
            
            # Step 4: Calculate confidence
            confidence = self.calculate_confidence(combined_context)
            
            # Step 5: Generate answer
            if combined_context:
                answer = await self.llm_generator.generate_answer(
                    query=query,
                    context_chunks=combined_context,
                    max_context_length=settings.max_context_length
                )
            else:
                answer = "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic."
            
            # Step 6: Generate follow-up questions
            follow_up_questions = []
            if combined_context and confidence > 0.5:
                try:
                    follow_up_questions = await self.llm_generator.generate_follow_up_questions(
                        original_query=query,
                        answer=answer,
                        context_chunks=combined_context
                    )
                except Exception as e:
                    logger.warning("Failed to generate follow-up questions", error=str(e))
            
            # Step 7: Extract sources
            sources = []
            seen_sources = set()
            for chunk in combined_context:
                source_key = f"{chunk.get('title', 'Unknown')}_{chunk.get('url', '')}"
                if source_key not in seen_sources:
                    sources.append({
                        "title": chunk.get("title", "Unknown"),
                        "url": chunk.get("url", ""),
                        "type": chunk.get("source_type", "unknown")
                    })
                    seen_sources.add(source_key)
            
            # Step 8: Build response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            context = RAGContext(
                query=query,
                query_type=query_type,
                elog_entries=elog_entries,
                knowledge_chunks=knowledge_chunks,
                total_sources=len(combined_context),
                confidence_score=confidence,
                processing_time=processing_time
            )
            
            response = RAGResponse(
                answer=answer,
                context=context,
                follow_up_questions=follow_up_questions,
                sources=sources,
                timestamp=datetime.now()
            )
            
            logger.info(
                "RAG query processed",
                query=query,
                type=query_type.value,
                sources=len(combined_context),
                confidence=confidence,
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            logger.error("RAG processing failed", query=query, error=str(e))
            
            # Return error response
            processing_time = (datetime.now() - start_time).total_seconds()
            
            context = RAGContext(
                query=query,
                query_type=QueryType.GENERAL,
                elog_entries=[],
                knowledge_chunks=[],
                total_sources=0,
                confidence_score=0.0,
                processing_time=processing_time
            )
            
            return RAGResponse(
                answer="I apologize, but I encountered an error while processing your question. Please try again.",
                context=context,
                follow_up_questions=[],
                sources=[],
                timestamp=datetime.now()
            )
    
    async def close(self):
        """Close all clients and cleanup resources."""
        try:
            await self.llm_generator.close()
            self.elog_client.close()
            self.knowledge_client.close()
            logger.info("RAG orchestrator closed")
        except Exception as e:
            logger.error("Error closing RAG orchestrator", error=str(e))