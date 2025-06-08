# Technical Context: DeepWiki-Open

## Technology Stack

### Frontend
- **Framework**: Next.js (React)
- **Styling**: Tailwind CSS
- **State Management**: React Context API
- **Rendering**: Server-side rendering (SSR) and client-side rendering
- **Diagram Rendering**: Mermaid.js
- **Markdown Rendering**: Custom components with syntax highlighting
- **WebSocket Client**: Custom implementation for streaming responses
- **Internationalization**: Next.js i18n

### Backend
- **Framework**: FastAPI (Python)
- **API**: RESTful endpoints and WebSocket
- **API Documentation**: OpenAPI/Swagger
- **Repository Management**: Git operations via Python libraries
- **Vector Database**: FAISS for similarity search
- **Embedding Generation**: Various models via provider APIs
- **Text Processing**: Custom chunking and processing logic
- **Caching**: File-based caching system
- **Local Project Analysis**: File system operations and code analysis tools

### AI Integration
- **Model Providers**:
  - Google Gemini
  - OpenAI
  - OpenRouter
  - Ollama (local models)
- **RAG Implementation**: Custom retrieval and generation pipeline
- **Streaming**: Server-sent events via WebSockets

### Deployment
- **Containerization**: Docker
- **Orchestration**: Docker Compose
- **Environment Management**: Environment variables

## Development Setup

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- Git
- Docker (optional, for containerized deployment)

### Environment Configuration
- `.env` file for API keys and configuration
- Configuration files in `api/config/` directory

### Local Development
1. Clone repository
2. Install frontend dependencies (`npm install` or `yarn install`)
3. Install backend dependencies (`pip install -r api/requirements.txt`)
4. Set up environment variables
5. Start backend server (`python -m api.main`)
6. Start frontend development server (`npm run dev` or `yarn dev`)
   - Note: Turbopack is not used due to compatibility issues with Google Fonts

### Docker Development
1. Clone repository
2. Set up environment variables
3. Run with Docker Compose (`docker-compose up`)

## Technical Constraints

### API Limitations
- Rate limits from model providers
- Token limits for context windows
- Cost considerations for API usage

### Repository Processing
- Size limitations for repositories
- Processing time for large codebases
- Memory constraints for embedding storage

### Model Capabilities
- Varying quality across different providers
- Context window limitations
- Hallucination potential in generated content

### Performance Considerations
- Latency in API responses
- Memory usage for large repositories
- Disk space for caching

## Dependencies

### Frontend Dependencies
- **next**: React framework for web applications
- **react**: UI library
- **tailwindcss**: Utility-first CSS framework
- **mermaid**: Diagram generation library
- **react-markdown**: Markdown rendering
- **prism-react-renderer**: Syntax highlighting
- **next-themes**: Theme management
- **next-intl**: Internationalization

### Backend Dependencies
- **fastapi**: Web framework for building APIs
- **uvicorn**: ASGI server
- **pydantic**: Data validation
- **langchain**: Framework for LLM applications
- **faiss-cpu**: Vector similarity search
- **openai**: OpenAI API client
- **google-generativeai**: Google Gemini API client
- **websockets**: WebSocket implementation
- **gitpython**: Git operations
- **tiktoken**: Tokenization for OpenAI models
- **click**: Command-line interface creation
- **pickle**: Object serialization for report storage
- **glob**: File pattern matching for report listing

## Configuration System

### Environment Variables
- `GOOGLE_API_KEY`: Google Gemini API key
- `OPENAI_API_KEY`: OpenAI API key
- `OPENROUTER_API_KEY`: OpenRouter API key
- `OLLAMA_HOST`: Ollama host URL
- `PORT`: Backend server port
- `SERVER_BASE_URL`: Base URL for the API server
- `OPENAI_BASE_URL`: Custom OpenAI API endpoint (optional)
- `DEEPWIKI_CONFIG_DIR`: Custom configuration directory (optional)

### Configuration Files
- **generator.json**: Model configuration for text generation
  ```json
  {
    "providers": {
      "google": {
        "default_model": "gemini-2.0-flash",
        "available_models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.0-pro"]
      },
      "openai": {
        "default_model": "gpt-4o",
        "available_models": ["gpt-4o", "o4-mini"]
      },
      "openrouter": {
        "default_model": "openai/gpt-4o",
        "available_models": ["openai/gpt-4o", "anthropic/claude-3-5-sonnet", "google/gemini-2.0"]
      },
      "ollama": {
        "default_model": "llama3",
        "available_models": ["llama3", "mistral", "codellama"]
      }
    },
    "parameters": {
      "temperature": 0.7,
      "top_p": 0.9,
      "max_tokens": 4000
    }
  }
  ```

- **embedder.json**: Configuration for embedding models
  ```json
  {
    "embedding_model": "text-embedding-3-small",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "retriever": {
      "search_type": "similarity",
      "k": 5
    }
  }
  ```

- **repo.json**: Repository processing configuration
  ```json
  {
    "exclude_patterns": [
      "node_modules/**",
      ".git/**",
      "**/*.min.js",
      "**/*.map"
    ],
    "max_repo_size_mb": 500,
    "max_file_size_kb": 1000
  }
  ```

## Data Storage

### Repository Storage
- Cloned repositories stored in `~/.adalflow/repos/`
- Directory structure based on repository source and path

### Vector Database
- Embeddings stored in `~/.adalflow/databases/`
- FAISS indexes for efficient similarity search

### Wiki Cache
- Generated wiki content cached in `~/.adalflow/wikicache/`
- Organized by repository identifier

### Local Project Reports
- Project analysis reports stored in `~/.adalflow/databases/` with prefix `local_`
- Reports include project statistics, file information, and AI-generated summaries
- Serialized using pickle format with timestamp-based naming
- Accessible through API endpoints and command-line interface

## API Documentation

### OpenAPI Specification
- Comprehensive OpenAPI 3.0.3 specification in `api/openapi.yaml`
- Documents all API endpoints, request/response models, and parameters
- Organized into logical tags: Chat, Wiki, Local Project, Models, Health
- Includes detailed descriptions for all components

### Swagger UI
- Interactive API documentation via Swagger UI
- Accessible at `/api/docs` endpoint
- Allows testing API endpoints directly from the browser
- Customized styling for better readability
- Responsive design for various screen sizes

### API Endpoints
- **Documentation**: `/api/docs`, `/api/openapi.yaml`
- **Chat**: `/chat/completions/stream`, `/ws/chat`
- **Wiki**: `/export/wiki`, `/api/wiki_cache`, `/api/processed_projects`
- **Local Project**: `/api/local_project/analyze`, `/api/local_project/reports`, `/api/local_project/report/{id}`
- **Models**: `/models/config`
- **Health**: `/health`, `/`

## Tool Usage Patterns

### Local Project Analysis
- File system traversal for project structure analysis
- Language detection based on file extensions
- Line counting for code statistics
- Classification of files (code, test, config, documentation)
- Project summary generation using RAG
- Command-line interface for easy access

### Repository Management
- GitPython for cloning and analyzing repositories
- Custom filtering based on configuration

### Text Processing
- Custom chunking strategies for code files
- Special handling for different file types
- Tokenization for embedding generation

### Embedding Generation
- OpenAI embeddings for vector representation
- FAISS for similarity search and retrieval

### Content Generation
- Provider-specific clients for text generation
- Prompt engineering for context-aware responses
- Streaming for real-time interaction

### Diagram Generation
- Mermaid syntax generation for visual representations
- Client-side rendering of diagrams

### WebSocket Communication
- Custom connection management
- Streaming response handling
- Client reconnection logic

## Security Considerations

### API Key Management
- Environment variables for sensitive credentials
- No client-side exposure of API keys

### Private Repository Access
- Token-based authentication for private repositories
- Secure storage of access tokens

### Data Isolation
- Separation of data between different users
- Temporary storage with appropriate cleanup

## Performance Optimization

### Caching Strategy
- Repository caching to avoid repeated cloning
- Embedding caching for performance
- Wiki content caching for quick access

### Resource Management
- Efficient memory usage for large repositories
- Garbage collection for temporary files
- Streaming responses to reduce memory pressure

### Concurrency
- Parallel processing where appropriate
- Asynchronous API calls
- WebSocket for non-blocking communication
