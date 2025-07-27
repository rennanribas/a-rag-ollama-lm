# AI RAG Agent

A retrieval-augmented generation system that crawls documentation websites and provides intelligent, context-aware responses through a REST API. Built with FastAPI, ChromaDB, and support for multiple LLM providers including Gemini, OpenAI, and Ollama.

## Features

- Asynchronous web crawling with configurable depth and concurrency
- Vector-based document indexing using ChromaDB
- Multiple LLM provider support (Gemini, OpenAI, Ollama)
- RESTful API with automatic documentation
- MCP (Model Context Protocol) server integration
- Incremental updates with content change detection
- Domain-specific crawling configurations

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/ai-rag-agent.git
cd ai-rag-agent
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure your settings:

```bash
cp .env.example .env
```

Edit `.env` with your preferred LLM provider:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

### Usage

Start the API server:

```bash
python -m src serve --host localhost --port 8000
```

Crawl documentation:

```bash
python -m src crawl https://developer.apple.com/documentation/swiftui --max-depth 3
```

Query the system:

```bash
# Interactive mode
python -m src interactive

# Direct query
python -m src query "How to implement HealthKit in SwiftUI?"

# HTTP API
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "SwiftUI navigation best practices"}'
```

## Architecture

The system follows a pipeline architecture:

1. **Web Crawler**: Asynchronously fetches and processes documentation
2. **Document Indexer**: Chunks content and generates embeddings
3. **Vector Store**: ChromaDB for efficient similarity search
4. **LLM Integration**: Multiple provider support for response generation
5. **API Layer**: FastAPI service with automatic documentation

## API Reference

### Health Check
```http
GET /health
```

### Crawl Documentation
```http
POST /crawl
Content-Type: application/json

{
  "urls": ["https://developer.apple.com/documentation/swiftui"],
  "max_depth": 2
}
```

### Query System
```http
POST /query
Content-Type: application/json

{
  "query": "How to implement HealthKit workout sessions?",
  "domain": "apple_ios",
  "session_id": "optional-session-id"
}
```

## Configuration

### LLM Providers

The system supports multiple LLM providers. Choose based on your requirements:

#### Gemini (Recommended)
Google's Gemini API offers generous free tier limits:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
EMBEDDING_PROVIDER=huggingface
```

Get your API key at [Google AI Studio](https://ai.google.dev/)

#### Ollama (Local)
Run models locally without API costs:

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=huggingface
```

Requires Ollama installation and model download.

#### OpenAI
For production deployments requiring high reliability:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
EMBEDDING_PROVIDER=openai
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|----------|
| `LLM_PROVIDER` | LLM provider (gemini, ollama, openai) | gemini |
| `EMBEDDING_PROVIDER` | Embedding provider (huggingface, gemini, ollama, openai) | huggingface |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage directory | ./data/chroma_db |
| `CRAWL_DELAY` | Delay between requests (seconds) | 1.0 |
| `MAX_CONCURRENT_REQUESTS` | Concurrent crawling limit | 5 |
| `API_HOST` | API server host | localhost |
| `API_PORT` | API server port | 8000 |
| `LOG_LEVEL` | Logging level | INFO |

### Complete Configuration Example

```env
# LLM Configuration
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Storage
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Crawler Settings
CRAWL_DELAY=1.0
MAX_CONCURRENT_REQUESTS=5
USER_AGENT=AI-RAG-Agent/1.0

# API Server
API_HOST=localhost
API_PORT=8000

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agent.log
```

## MCP Integration

The system includes Model Context Protocol (MCP) server support for IDE integration:

```json
{
  "mcpServers": {
    "ai-rag-agent": {
      "command": "python3",
      "args": ["/path/to/ai-rag-agent/run_mcp.py"],
      "env": {
        "TRAE_AI": "true"
      }
    }
  }
}
```

Update the path to match your installation directory.

## Performance Features

- **Incremental Updates**: Content hashing detects changes, avoiding unnecessary re-indexing
- **Concurrent Processing**: Asynchronous crawling with configurable concurrency limits
- **Memory Optimization**: Chunked document processing and configurable batch sizes
- **Persistent State**: Crawler state and vector embeddings are preserved between runs

## Troubleshooting

### Common Issues

**API Key Problems**
```bash
# Verify your configuration
python -m src query "test" --log-level DEBUG
```

**Crawling Failures**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m src crawl https://example.com
```

**Storage Issues**
```bash
# Clear and rebuild index
rm -rf ./data/chroma_db
python -m src crawl https://your-docs-url.com
```

**Memory Constraints**
```bash
# Reduce concurrent requests
export MAX_CONCURRENT_REQUESTS=2
```

### Log Files
- Application: `./logs/agent.log`
- Crawler state: `./data/chroma_db/crawler_state.json`

## Development

### Project Structure
```
ai-rag-agent/
├── src/
│   ├── config.py          # Configuration management
│   ├── crawler.py         # Web crawling logic
│   ├── indexer.py         # Document indexing
│   ├── agent.py           # LLM integration
│   ├── api.py             # FastAPI service
│   ├── main.py            # CLI interface
│   └── mcp_server.py      # MCP protocol server
├── data/                  # Storage directory
├── requirements.txt       # Dependencies
└── .env.example          # Environment template
```

### Testing
```bash
pytest tests/ -v
```

### Code Quality
```bash
black src/
flake8 src/
mypy src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details.