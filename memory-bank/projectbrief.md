# Project Brief: DeepWiki-Open

## Project Overview
DeepWiki-Open is an AI-powered tool that automatically creates comprehensive, interactive wikis for any GitHub, GitLab, or BitBucket repository. The system analyzes code structure, generates documentation, creates visual diagrams, and organizes everything into an easy-to-navigate wiki format.

## Core Objectives
1. Provide instant, high-quality documentation for code repositories
2. Generate visual representations of code architecture and relationships
3. Create an intuitive navigation system for exploring repository documentation
4. Enable intelligent querying of repositories through RAG-powered chat
5. Support deep, multi-turn research for complex topics
6. Accommodate multiple AI model providers for flexibility and performance

## Target Users
- Software developers seeking to understand new codebases
- Development teams needing to maintain documentation
- Open-source project maintainers wanting to improve accessibility
- Technical managers requiring overview of project architecture
- New team members onboarding to existing projects

## Key Requirements

### Functional Requirements
- Repository analysis for GitHub, GitLab, and BitBucket (including private repos)
- Automatic documentation generation with context-aware AI
- Visual diagram creation for code relationships and architecture
- Structured wiki organization with intuitive navigation
- RAG-powered Q&A functionality for repository-specific queries
- Multi-turn research capability for in-depth investigations
- Support for multiple model providers (Google, OpenAI, OpenRouter, Ollama)
- Multi-language support for generated content

### Technical Requirements
- Efficient repository cloning and analysis
- Effective embedding generation for code retrieval
- Streaming response capability for real-time interaction
- Caching system for performance optimization
- Docker support for easy deployment
- Configurable model selection system
- Secure handling of API keys and access tokens

## Success Criteria
- Generate accurate, comprehensive documentation for repositories of various sizes and complexities
- Produce clear, informative visual diagrams that correctly represent code relationships
- Provide relevant, context-aware responses to queries about the repository
- Support multiple model providers with seamless switching between them
- Maintain performance and responsiveness even with large repositories
- Enable easy deployment and configuration for users with varying technical expertise

## Constraints
- Dependency on external AI model providers and their APIs
- Repository size and complexity limitations based on embedding and processing capabilities
- Need for API keys from supported providers
- Processing time constraints for large repositories

## Timeline
This is an ongoing open-source project with continuous development and improvement.
