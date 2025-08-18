#!/usr/bin/env python3
"""
Test script for the SwissFEL Agentic RAG system.

This script performs end-to-end testing of all components.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.rag_orchestrator import AgenticRAGOrchestrator
from src.data.elog_client import ElogClient
from src.data.knowledge_graph import KnowledgeGraphClient
from src.models.ollama_client import OllamaClient
import structlog

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def test_elog_client():
    """Test ELOG client functionality."""
    logger.info("Testing ELOG client...")
    
    try:
        client = ElogClient()
        
        # Test connection
        logger.info("Testing ELOG connection...")
        client.connect()
        
        # Test statistics
        logger.info("Getting ELOG statistics...")
        stats = client.get_entry_statistics()
        logger.info("ELOG statistics", **stats)
        
        # Test recent entries
        logger.info("Getting recent entries...")
        entries = client.get_recent_entries(days=7)
        logger.info(f"Retrieved {len(entries)} recent entries")
        
        if entries:
            logger.info("Sample entry", 
                       id=entries[0].id,
                       subject=entries[0].subject[:50] + "...",
                       date=entries[0].date)
        
        client.close()
        logger.info("✓ ELOG client test passed")
        return True
        
    except Exception as e:
        logger.error("✗ ELOG client test failed", error=str(e))
        return False


async def test_knowledge_graph():
    """Test knowledge graph functionality."""
    logger.info("Testing knowledge graph client...")
    
    try:
        client = KnowledgeGraphClient()
        
        # Test vector search
        logger.info("Testing vector search...")
        results = client.search_similar_content("undulator", limit=3)
        logger.info(f"Vector search returned {len(results)} results")
        
        # Test machine search
        logger.info("Testing machine search...")
        machine_results = client.search_by_machine_type(["linac", "rf"], limit=3)
        logger.info(f"Machine search returned {len(machine_results)} results")
        
        # Test hybrid search
        logger.info("Testing hybrid search...")
        hybrid_results = client.hybrid_search("beam diagnostics", vector_limit=3, graph_limit=3)
        logger.info(f"Hybrid search returned {len(hybrid_results['vector_results'])} vector + {len(hybrid_results['graph_results'])} graph results")
        
        client.close()
        logger.info("✓ Knowledge graph test passed")
        return True
        
    except Exception as e:
        logger.error("✗ Knowledge graph test failed", error=str(e))
        return False


async def test_ollama_client():
    """Test Ollama LLM client."""
    logger.info("Testing Ollama client...")
    
    try:
        client = OllamaClient()
        
        # Test health check
        logger.info("Testing Ollama health...")
        health = await client.check_health()
        if not health:
            logger.warning("Ollama server not healthy")
            return False
        
        # Test model list
        logger.info("Testing model list...")
        models = await client.list_models()
        logger.info(f"Available models: {len(models)}")
        
        # Test simple generation
        logger.info("Testing text generation...")
        response = await client.generate("What is SwissFEL?", temperature=0.3)
        logger.info(f"Generated response ({len(response)} chars): {response[:100]}...")
        
        await client.close()
        logger.info("✓ Ollama client test passed")
        return True
        
    except Exception as e:
        logger.error("✗ Ollama client test failed", error=str(e))
        return False


async def test_rag_orchestrator():
    """Test the complete RAG orchestrator."""
    logger.info("Testing RAG orchestrator...")
    
    try:
        orchestrator = AgenticRAGOrchestrator()
        
        # Test queries of different types
        test_queries = [
            "What are the startup procedures for the linac?",
            "How do I troubleshoot undulator problems?", 
            "What is the current beam energy?",
            "Show me recent commissioning logs",
            "What are the RF system specifications?"
        ]
        
        for query in test_queries:
            logger.info(f"Testing query: {query}")
            
            response = await orchestrator.process_query(query)
            
            logger.info("Query result",
                       query_type=response.context.query_type.value,
                       confidence=response.context.confidence_score,
                       sources=response.context.total_sources,
                       processing_time=response.context.processing_time,
                       answer_length=len(response.answer))
            
            # Show a snippet of the answer
            answer_snippet = response.answer[:200] + "..." if len(response.answer) > 200 else response.answer
            logger.info(f"Answer snippet: {answer_snippet}")
            
            if response.follow_up_questions:
                logger.info(f"Follow-up questions: {response.follow_up_questions}")
        
        await orchestrator.close()
        logger.info("✓ RAG orchestrator test passed")
        return True
        
    except Exception as e:
        logger.error("✗ RAG orchestrator test failed", error=str(e))
        return False


async def main():
    """Run all tests."""
    logger.info("Starting SwissFEL Agentic RAG system tests...")
    
    tests = [
        ("ELOG Client", test_elog_client),
        ("Knowledge Graph", test_knowledge_graph),
        ("Ollama LLM", test_ollama_client),
        ("RAG Orchestrator", test_rag_orchestrator),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running {test_name} test...")
        logger.info(f"{'='*50}")
        
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} crashed", error=str(e))
            results.append((test_name, False))
    
    # Summary
    logger.info(f"\n{'='*50}")
    logger.info("TEST SUMMARY")
    logger.info(f"{'='*50}")
    
    passed = 0
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        logger.info("🎉 All tests passed! System is ready.")
        return 0
    else:
        logger.error(f"❌ {len(results) - passed} tests failed. Check configuration.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())