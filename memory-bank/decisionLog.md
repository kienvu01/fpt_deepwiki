# Decision Log

This file records architectural and implementation decisions using a list format.
2025-06-13 09:13:26 - Initial decision log setup based on project documentation

## Decision: Provider-Based Model Selection System

### Rationale
* Need for flexibility in model selection
* Support for multiple AI providers
* Easy integration of new models
* Enterprise deployment considerations

### Implementation Details
* JSON configuration files for model settings
* Environment variable integration
* Support for custom API endpoints
* Standardized provider interfaces

## Decision: Client-Server Architecture

### Rationale
* Clear separation of concerns
* Scalability requirements
* Need for real-time features
* Cross-platform compatibility

### Implementation Details
* FastAPI backend for performance
* Next.js frontend for SSR and SEO
* RESTful API design
* Streaming support for chat completions

## Decision: Data Storage Strategy

### Rationale
* Need for persistent data storage
* Cache management requirements
* Docker deployment support
* Multi-user considerations

### Implementation Details
* Structured ~/.adalflow/ directory
* Separate spaces for different data types
* Docker volume mapping support
* Cache invalidation system

## Decision: Configuration Management

### Rationale
* Need for flexible configuration
* Support for multiple environments
* Enterprise deployment requirements
* Security considerations

### Implementation Details
* JSON-based configuration files
* Environment variable overrides
* Custom configuration directory support
* Secure API key handling
2025-06-13 09:25:07 - Integration of Wiki Generation into Local Project Analysis

### Rationale
* Consolidate wiki generation functionality in the backend
* Provide consistent wiki generation across different interfaces
* Enable better control over generation process
* Support multiple output formats and configurations

### Implementation Details
* Added WikiGenerationConfig for flexible configuration
* Integrated prompts directly in the backend
* Enhanced local project analysis endpoint
* Structured wiki content generation with multiple pages
* Support for diagrams and code snippets

### Impact
* Improved maintainability by centralizing wiki logic
* Better consistency in generated content
* More flexible configuration options
* Reduced frontend complexity
2025-06-13 09:34:24 - Unified RAG System Implementation

### Rationale
* Need for consistent RAG behavior between frontend and local project analysis
* Better code reuse and maintainability
* Improved document indexing and retrieval

### Implementation Details
* Modified store_report_in_rag to use the same RAG and DatabaseManager setup
* Added file content indexing for better context
* Reused frontend's document transformation pipeline
* Maintained consistent metadata structure

### Impact
* Consistent search and retrieval behavior
* Better integration between local and remote repository analysis
* Improved context for RAG responses
[2025-06-13 14:59:12] - Model Configuration Parameter Adjustment
- Decision: Removed unsupported parameters (num_ctx) from Ollama configuration in generator.json
- Rationale: Ollama client does not support the num_ctx parameter, causing generation failures
- Impact: Successfully enabled wiki generation with both Ollama and OpenAI providers
- Validation: Tested with OpenAI provider, generating comprehensive wiki content with proper formatting