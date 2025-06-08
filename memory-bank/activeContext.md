# Active Context: DeepWiki-Open

## Current Work Focus

The current focus is on understanding the DeepWiki-Open project and establishing a memory bank to document the project's structure, functionality, and technical details. This is the initial setup phase where we're creating comprehensive documentation to serve as a foundation for future development work.

Key areas of focus include:
- Understanding the overall architecture and component relationships
- Documenting the technology stack and dependencies
- Identifying core patterns and implementation details
- Establishing a clear picture of the project's current state

## Recent Changes

### Added Swagger API Documentation
- Created comprehensive OpenAPI specification (openapi.yaml) documenting all API endpoints
- Implemented a Swagger UI interface for interactive API exploration and testing
- Added new endpoints to serve the OpenAPI specification and Swagger UI
- Updated the root endpoint to include links to the documentation
- Organized API endpoints into logical tags (Chat, Wiki, Local Project, Models, Health)
- Documented all request/response models with detailed descriptions

### Added Local Project Analysis API
- Created a new API for analyzing local project directories
- Implemented functionality to extract statistics and insights from local projects
- Added endpoints for analyzing projects, listing reports, and retrieving specific reports
- Created test cases to ensure the API works correctly
- Added example scripts and documentation for using the API
- Integrated with the RAG system to store project reports for later querying

### Fixed Turbopack Google Font Issue
- Fixed error: "Module not found: Can't resolve '@vercel/turbopack-next/internal/font/google/font'"
- Removed the `--turbopack` flag from the dev script in package.json
- The project now uses the default webpack bundler instead of Turbopack
- This resolved a compatibility issue between Turbopack and Google Fonts in Next.js 15.3.1

### UI Theme Update
- Changed the UI theme from a Japanese aesthetic to a modern blue theme
- Updated color variables in `src/app/globals.css` for both light and dark modes
- Modified background patterns to use a subtle grid instead of paper texture
- Enhanced button, input, and card styles with modern design elements
- Improved hover and focus states for better user interaction

### Memory Bank Creation
- Created the memory bank with the following structure:
  - `projectbrief.md`: Foundation document defining core requirements and goals
  - `productContext.md`: Why the project exists, problems it solves, user experience goals
  - `systemPatterns.md`: System architecture, key technical decisions, design patterns
  - `techContext.md`: Technologies used, development setup, technical constraints
  - `activeContext.md`: Current work focus, recent changes, next steps (this file)
  - `progress.md`: What works, what's left to build, current status

## Next Steps

After implementing the Swagger API documentation and local project analysis API, potential next steps could include:

1. **Enhance Local Project Analysis**:
   - Add support for more programming languages
   - Improve the project summary generation with more detailed insights
   - Create a frontend interface for the local project analysis feature
   - Add visualization capabilities for project statistics

2. **Explore Key Components**:
   - Examine the RAG implementation in `api/rag.py`
   - Understand the data pipeline in `api/data_pipeline.py`
   - Review the WebSocket implementation in `api/websocket_wiki.py`
   - Analyze the frontend components, especially the Ask feature

2. **Potential Enhancements**:
   - Improve model provider integration
   - Enhance caching mechanisms
   - Optimize embedding generation
   - Expand language support
   - Add new visualization types
   - Further refine UI components to match the new modern blue theme
   - Create additional UI themes for user customization
   - Improve responsive design for mobile devices

3. **Documentation Improvements**:
   - Create additional memory bank files for complex features
   - Expand API documentation with more examples and use cases
   - Add detailed component diagrams
   - Create user guides
   - Keep the OpenAPI specification updated as new endpoints are added

4. **Testing and Validation**:
   - Test with various repository types and sizes
   - Validate performance with different model providers
   - Ensure security of private repository handling

## Active Decisions and Considerations

### API Documentation Design
- Created a comprehensive OpenAPI 3.0.3 specification documenting all endpoints
- Organized endpoints into logical tags for better navigation
- Used Swagger UI for interactive API exploration and testing
- Added detailed descriptions for all parameters, request bodies, and responses
- Ensured documentation is accessible through dedicated endpoints
- Updated the FastAPI application to serve the documentation files

### Local Project Analysis API Design
- Created a modular design with clear separation of concerns
- Used Pydantic models for request and response validation
- Implemented a command-line interface for easy usage
- Added comprehensive test cases to ensure reliability
- Created example scripts and documentation for better usability
- Integrated with the RAG system for storing and retrieving project reports

### UI Theme Design
- Changed from a Japanese aesthetic to a modern blue theme
- Selected a clean, professional color palette suitable for a technical application
- Enhanced interactive elements with subtle animations and transitions
- Improved visual hierarchy and readability with refined spacing and typography
- Maintained consistent design language across components

### Memory Bank Structure
- Decided to follow the hierarchical structure outlined in the custom instructions
- Created comprehensive documentation for each aspect of the project
- Used Mermaid diagrams to visualize system architecture and processes

### Project Understanding
- Focused on understanding both frontend and backend components
- Identified key patterns and architectural decisions
- Documented the configuration system and environment variables

### Documentation Approach
- Prioritized clarity and comprehensiveness
- Included both high-level overviews and detailed technical information
- Used consistent formatting and structure across files

## Important Patterns and Preferences

### Documentation Patterns
- Clear hierarchical structure
- Comprehensive coverage of all system aspects
- Visual diagrams for complex relationships
- Consistent formatting and terminology

### System Design Patterns
- Client-server architecture with clear separation of concerns
- Provider-based model selection system
- Configuration-driven behavior
- Caching at multiple levels for performance
- WebSocket-based streaming for real-time interaction

### Code Organization Preferences
- Modular components with clear responsibilities
- Configuration files for customizable behavior
- Environment variables for sensitive information
- Clear separation between frontend and backend

## Learnings and Project Insights

### Local Project Analysis
- The local project analysis API provides valuable insights into project structure and statistics
- The API can analyze various aspects of a project, including file types, lines of code, and directory structure
- The integration with the RAG system allows for storing and retrieving project reports
- The command-line interface and example scripts make it easy to use the API

### Project Structure
- DeepWiki-Open follows a modern web application architecture with Next.js frontend and FastAPI backend
- The system is designed to be flexible, supporting multiple model providers and repository sources
- Configuration is centralized in JSON files and environment variables

### Key Features
- The RAG system is central to the project's functionality, enabling context-aware responses
- WebSocket streaming provides real-time interaction for the Ask and DeepResearch features
- Caching mechanisms optimize performance for repeated access to repositories
- Multi-language support enhances accessibility

### Technical Insights
- The provider-based model selection system allows for flexibility in AI model usage
- The repository processing pipeline is designed to handle various repository types and sizes
- Security considerations are addressed through environment variables and token-based authentication
- Performance optimizations include parallel processing, caching, and efficient resource management

### Project Evolution
- The project appears to be actively developed, with support for multiple model providers
- Recent additions include OpenRouter integration and DeepResearch functionality
- The configuration system has evolved to support more customization options
