# MCP Configuration for AI RAG Agent

Paste this JSON configuration into your Trae AI MCP settings:

```json
{
  "mcpServers": {
    "ai-rag-agent": {
      "command": "python3",
      "args": ["/Users/rennanribas/Projects/ai-rag-agent/run_mcp.py"],
      "env": {
        "TRAE_AI": "true"
      }
    }
  }
}
```

## Instructions:

1. Copy the `ai-rag-agent` object from the `mcpServers` section above
2. Paste it into your Trae AI MCP configuration file
3. **IMPORTANT**: Update the path in `args` to match your project location:
   - Replace `/Users/rennanribas/Projects/ai-rag-agent/run_mcp.py` with your actual project path
   - Example: `/Users/yourname/Projects/ai-rag-agent/run_mcp.py`
4. The server will automatically start in background mode for Trae AI compatibility

## Server Endpoints:

- **Health Check**: `http://localhost:8001/health`
- **API Documentation**: `http://localhost:8001/docs`
- **Query Endpoint**: `http://localhost:8001/query`

## Environment Setup:

Make sure you have a `.env` file with your API keys:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key
EMBEDDING_PROVIDER=huggingface
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
API_HOST=0.0.0.0
API_PORT=8001
```

## Quick Start

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ai-rag-agent

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your provider settings
```

### Start Service

```bash
# Start API server
python -m src serve --host localhost --port 8000
```

### Crawl Documentation

```bash
# Crawl Apple documentation
python -m src crawl https://developer.apple.com/documentation/swiftui --max-depth 3
```

### Query

```bash
# Interactive mode
python -m src interactive

# Single query
python -m src query "How to create a workout session in watchOS?"

# API endpoint
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How to implement HealthKit in SwiftUI?"}'
```

## Architecture

Web Crawler → LlamaIndex + ChromaDB → LLM Agent → FastAPI Service

## API Endpoints

```bash
# Health check
GET /health

# Crawl documentation
POST /crawl
{
  "urls": ["https://developer.apple.com/documentation/swiftui"],
  "max_depth": 2
}

# Query agent
POST /query
{
  "query": "How to implement HealthKit workout sessions?"
}
```

## Configuration

Edit `.env` file:

```bash
# LLM Provider (gemini, ollama, openai)
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_api_key

# Database
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Logging
LOG_LEVEL=INFO
```

**📖 See [TRAE_INTEGRATION.md](TRAE_INTEGRATION.md) for detailed integration guide**

### VS Code Extension Example
```javascript
// Query the RAG agent
const response = await fetch('http://localhost:8000/query', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    query: selectedText,
    domain: 'apple_ios',
    session_id: vscode.env.sessionId
  })
});

const result = await response.json();
// Display result.answer in IDE
```

## Configuration

### LLM Provider Options

#### 🆓 Free Options (Recommended)

**Gemini (Free API)**
```bash
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_free_api_key  # Get at https://ai.google.dev/
EMBEDDING_PROVIDER=huggingface
```

**Ollama (Completely Free)**
```bash
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1:8b
EMBEDDING_PROVIDER=huggingface
```

#### 💰 Paid Options

**OpenAI**
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key
EMBEDDING_PROVIDER=openai
```

### Key Environment Variables

- `LLM_PROVIDER`: Choose from `gemini`, `ollama`, or `openai`
- `EMBEDDING_PROVIDER`: Choose from `huggingface`, `gemini`, `ollama`, or `openai`
- `CHROMA_PERSIST_DIRECTORY`: Directory for ChromaDB storage
- `CRAWLER_DELAY`: Delay between requests (seconds)
- `API_HOST`: API server host
- `API_PORT`: API server port

### Additional Configuration
```bash
# Storage Configuration
CHROMA_PERSIST_DIRECTORY=./data/chroma_db

# Crawler Configuration
CRAWL_DELAY=1.0
MAX_CONCURRENT_REQUESTS=5
USER_AGENT=AI-RAG-Agent/1.0

# API Configuration
API_HOST=localhost
API_PORT=8000
API_RELOAD=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/agent.log
```

### Custom Domain Configuration
Add new domains by extending the domain configurations in `src/api.py`:

```python
"my_framework": {
    "name": "My Framework Documentation",
    "urls": [
        "https://docs.myframework.com/"
    ],
    "max_depth": 3
}
```

## Performance Optimization

### Incremental Updates
- Documents are hashed to detect changes
- Only modified content is re-indexed
- Crawler state is persisted between runs

### Concurrent Processing
- Asynchronous crawling with configurable concurrency
- Background indexing jobs
- Non-blocking API responses

### Memory Management
- Chunked document processing
- Configurable embedding batch sizes
- Automatic cleanup of old conversation history

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**
   ```bash
   # Verify your API key
   export OPENAI_API_KEY="your-key-here"
   ai-rag status
   ```

2. **Crawling Failures**
   ```bash
   # Check crawler logs
   ai-rag crawl https://example.com --log-level DEBUG
   ```

3. **Storage Issues**
   ```bash
   # Clear and rebuild index
   ai-rag clear
   ai-rag crawl-domain apple_ios
   ```

4. **Memory Issues**
   ```bash
   # Reduce concurrent requests
   export MAX_CONCURRENT_REQUESTS=2
   ```

### Logs
- Application logs: `./logs/agent.log`
- API access logs: Console output when running server
- Crawler state: `./data/chroma_db/crawler_state.json`

## Development

### Project Structure
```
ai-rag-agent/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── crawler.py         # Web crawling logic
│   ├── indexer.py         # LlamaIndex integration
│   ├── agent.py           # OpenAI agent logic
│   ├── api.py             # FastAPI service
│   └── main.py            # CLI interface
├── data/                  # Storage directory
├── logs/                  # Log files
├── requirements.txt       # Dependencies
├── setup.py              # Package setup
├── .env.example          # Environment template
└── README.md             # This file
```

### Running Tests
```bash
pytest tests/ -v
```

### Code Quality
```bash
# Format code
black src/

# Lint code
flake8 src/

# Type checking
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

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub

---

**Built for developers who want intelligent, contextual documentation assistance without leaving their IDE.**