# SwissFEL Agentic RAG System

An intelligent Retrieval-Augmented Generation (RAG) system designed for SwissFEL commissioning data, integrating ELOG operational logs with a comprehensive knowledge graph of machine specifications, procedures, and guidelines.

## Architecture Overview

This system implements a state-of-the-art agentic RAG workflow that combines:

- **ELOG Integration**: Real-time access to SwissFEL commissioning logs
- **Knowledge Graph**: Neo4j + Qdrant for structured and semantic knowledge retrieval  
- **Local LLM**: Ollama for privacy-preserving text generation
- **Agentic Orchestration**: Intelligent query routing and multi-source information synthesis

## Key Features

### Multi-Modal Information Retrieval
- Semantic search across ELOG entries and documentation
- Structured queries on machine relationships and dependencies
- Hybrid retrieval combining multiple knowledge sources

### Intelligent Query Processing
- Automatic query classification (operational, troubleshooting, machine-specific, etc.)
- Dynamic source selection based on query intent
- Context-aware information synthesis

### Real-time Integration
- Continuous ELOG monitoring for new entries
- Incremental knowledge graph updates
- Live operational data integration

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Access to SwissFEL ELOG system

### Installation

1. Clone and setup:
```bash
cd /home/linus/psirag/acc_RAG
cp .env.example .env
# Edit .env with your configurations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start infrastructure services:
```bash
# In the knowledge graph directory
cd /home/linus/psirag/acc_wiki_know_graph
docker-compose up -d
```

4. Verify Ollama is running:
```bash
ollama serve
# In another terminal:
ollama pull llama2:7b  # or your preferred model
```

### Basic Usage

1. Start the API server:
```bash
cd /home/linus/psirag/acc_RAG
python -m src.api.main
```

2. Test the system:
```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the startup procedures for the linac?"}'
```

3. Access the interactive docs at: `http://localhost:8000/docs`

## System Components

### Data Sources

#### ELOG Client (`src/data/elog_client.py`)
- Connects to SwissFEL commissioning logbook
- Supports real-time monitoring and historical searches
- Handles authentication and error recovery

#### Knowledge Graph Client (`src/data/knowledge_graph.py`)  
- Integrates existing Neo4j/Qdrant setup
- Provides semantic and structural search capabilities
- Handles machine dependencies and procedure lookups

### AI/ML Layer

#### Ollama LLM Integration (`src/models/ollama_client.py`)
- Local LLM deployment for privacy/security
- Streaming and batch text generation
- Specialized RAG prompt engineering

#### Embedding Models
- Reuses existing `thellert/accphysbert_cased` model
- Domain-specific embeddings for accelerator physics
- Optimized for technical documentation

### Orchestration

#### Agentic RAG Orchestrator (`src/core/rag_orchestrator.py`)
- Central coordination of all components
- Query classification and routing
- Multi-source information synthesis
- Confidence scoring and response generation

## Query Types

The system automatically classifies and routes queries:

- **Operational**: Startup, shutdown, beam operations
- **Troubleshooting**: Error diagnosis, fault analysis
- **Machine-specific**: Component specifications, dependencies
- **Procedural**: Step-by-step instructions, protocols
- **Historical**: Recent events, trend analysis

## API Endpoints

### Core Endpoints
- `POST /query` - Process RAG queries
- `GET /health` - System health check
- `GET /stats` - System statistics

### Data Access
- `GET /elog/recent` - Recent ELOG entries
- `GET /knowledge/search` - Knowledge graph search
- `GET /models/available` - Available LLM models

## Configuration

Key settings in `.env`:

```bash
# Core services
ELOG_URL=https://elog-gfa.psi.ch/SwissFEL+commissioning/
OLLAMA_HOST=http://localhost:11434
NEO4J_URI=bolt://localhost:7687
QDRANT_HOST=localhost

# AI Configuration  
OLLAMA_MODEL=llama2:7b
EMBEDDING_MODEL=thellert/accphysbert_cased
MAX_CONTEXT_LENGTH=4096
SIMILARITY_THRESHOLD=0.7
```

## Development

### Project Structure
```
acc_RAG/
├── src/
│   ├── api/          # FastAPI web interface
│   ├── agents/       # Specialized AI agents  
│   ├── core/         # RAG orchestration
│   ├── data/         # Data source clients
│   ├── models/       # LLM integrations
│   └── utils/        # Helper functions
├── config/           # Configuration management
├── tests/           # Unit and integration tests
└── docs/            # Documentation
```

### Running Tests
```bash
pytest tests/
```

### Adding New Data Sources
1. Implement client in `src/data/`
2. Add integration to orchestrator
3. Update query classification if needed

## Deployment

### Production Considerations

- Use authentication for ELOG access
- Configure proper CORS origins
- Set up monitoring and logging
- Use production-grade databases
- Implement rate limiting

### Docker Deployment
```bash
# Build container
docker build -t swissfel-rag .

# Run with docker-compose
docker-compose up -d
```

## Monitoring

The system provides comprehensive observability:

- Health checks for all components
- Performance metrics and timing
- Query classification statistics
- Confidence score tracking
- Source attribution and lineage

## Security

- Local LLM deployment (no external API calls)
- Encrypted communication with databases
- Access control and authentication support
- Audit logging for all queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Submit a pull request

## License

This project is part of the SwissFEL commissioning infrastructure at PSI.

## Support

For issues and questions:
- Check the system health endpoint: `/health`
- Review logs for error details
- Contact the SwissFEL commissioning team