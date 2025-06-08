import os
import logging
from fastapi import FastAPI, HTTPException, Query, Request, WebSocket, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import List, Optional, Dict, Any, Literal
import json
from datetime import datetime
from pydantic import BaseModel, Field
import google.generativeai as genai
import asyncio
import glob

# Get a logger for this module
logger = logging.getLogger(__name__)

# Get API keys from environment variables
google_api_key = os.environ.get('GOOGLE_API_KEY')

# Configure Google Generative AI
if google_api_key:
    genai.configure(api_key=google_api_key)
else:
    logger.warning("GOOGLE_API_KEY not found in environment variables")

# Initialize FastAPI app
app = FastAPI(
    title="DeepWiki-Open API",
    description="API for DeepWiki-Open, an AI-powered tool that automatically creates comprehensive, interactive wikis for any GitHub, GitLab, or BitBucket repository",
    version="1.0.0",
    docs_url=None,  # Disable default Swagger UI
    redoc_url=None  # Disable default ReDoc
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Helper function to get adalflow root path
def get_adalflow_default_root_path():
    return os.path.expanduser(os.path.join("~", ".adalflow"))

# --- Local Project Analysis Models ---
class LocalProjectAnalysisRequest(BaseModel):
    """
    Model for requesting a local project analysis.
    """
    project_path: str = Field(..., description="Path to the local project directory")
    excluded_dirs: Optional[List[str]] = Field(None, description="Directories to exclude from analysis")
    excluded_files: Optional[List[str]] = Field(None, description="File patterns to exclude from analysis")
    model_provider: str = Field("google", description="Model provider to use for summary generation")
    model_name: Optional[str] = Field(None, description="Model name to use with the provider")
    store_in_rag: bool = Field(True, description="Whether to store the report in the RAG system")

class LocalProjectReportSummary(BaseModel):
    """
    Model for a summary of a local project report.
    """
    id: str = Field(..., description="Unique identifier for the report")
    project_name: str = Field(..., description="Name of the project")
    analysis_timestamp: str = Field(..., description="Timestamp of the analysis")
    total_files: int = Field(..., description="Total number of files in the project")
    total_lines_of_code: int = Field(..., description="Total lines of code in the project")
    total_size_bytes: int = Field(..., description="Total size of the project in bytes")
    languages: Dict[str, int] = Field(..., description="Languages used in the project and their file counts")

# --- Pydantic Models ---
class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class ProcessedProjectEntry(BaseModel):
    id: str  # Filename
    owner: str
    repo: str
    name: str  # owner/repo
    repo_type: str # Renamed from type to repo_type for clarity with existing models
    submittedAt: int # Timestamp
    language: str # Extracted from filename

class WikiStructureModel(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]

class WikiCacheData(BaseModel):
    """
    Model for the data to be stored in the wiki cache.
    """
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  # Add repo_url to cache

class WikiCacheRequest(BaseModel):
    """
    Model for the request body when saving wiki cache.
    """
    owner: str
    repo: str
    repo_type: str
    language: str
    wiki_structure: WikiStructureModel
    generated_pages: Dict[str, WikiPage]
    repo_url: Optional[str] = None  # Add repo_url to cache request

class WikiExportRequest(BaseModel):
    """
    Model for requesting a wiki export.
    """
    repo_url: str = Field(..., description="URL of the repository")
    pages: List[WikiPage] = Field(..., description="List of wiki pages to export")
    format: Literal["markdown", "json"] = Field(..., description="Export format (markdown or json)")

# --- Model Configuration Models ---
class Model(BaseModel):
    """
    Model for LLM model configuration
    """
    id: str = Field(..., description="Model identifier")
    name: str = Field(..., description="Display name for the model")

class Provider(BaseModel):
    """
    Model for LLM provider configuration
    """
    id: str = Field(..., description="Provider identifier")
    name: str = Field(..., description="Display name for the provider")
    models: List[Model] = Field(..., description="List of available models for this provider")
    supportsCustomModel: Optional[bool] = Field(False, description="Whether this provider supports custom models")

class ModelConfig(BaseModel):
    """
    Model for the entire model configuration
    """
    providers: List[Provider] = Field(..., description="List of available model providers")
    defaultProvider: str = Field(..., description="ID of the default provider")

from api.config import configs
from api.local_report import analyze_project, generate_project_summary, store_report_in_rag, analyze_and_store_project

@app.get("/models/config", response_model=ModelConfig)
async def get_model_config():
    """
    Get available model providers and their models.

    This endpoint returns the configuration of available model providers and their
    respective models that can be used throughout the application.

    Returns:
        ModelConfig: A configuration object containing providers and their models
    """
    try:
        logger.info("Fetching model configurations")

        # Create providers from the config file
        providers = []
        default_provider = configs.get("default_provider", "google")

        # Add provider configuration based on config.py
        for provider_id, provider_config in configs["providers"].items():
            models = []
            # Add models from config
            for model_id in provider_config["models"].keys():
                # Get a more user-friendly display name if possible
                models.append(Model(id=model_id, name=model_id))

            # Add provider with its models
            providers.append(
                Provider(
                    id=provider_id,
                    name=f"{provider_id.capitalize()}",
                    supportsCustomModel=provider_config.get("supportsCustomModel", False),
                    models=models
                )
            )

        # Create and return the full configuration
        config = ModelConfig(
            providers=providers,
            defaultProvider=default_provider
        )
        return config

    except Exception as e:
        logger.error(f"Error creating model configuration: {str(e)}")
        # Return some default configuration in case of error
        return ModelConfig(
            providers=[
                Provider(
                    id="google",
                    name="Google",
                    supportsCustomModel=True,
                    models=[
                        Model(id="gemini-2.0-flash", name="Gemini 2.0 Flash")
                    ]
                )
            ],
            defaultProvider="google"
        )

@app.post("/export/wiki")
async def export_wiki(request: WikiExportRequest):
    """
    Export wiki content as Markdown or JSON.

    Args:
        request: The export request containing wiki pages and format

    Returns:
        A downloadable file in the requested format
    """
    try:
        logger.info(f"Exporting wiki for {request.repo_url} in {request.format} format")

        # Extract repository name from URL for the filename
        repo_parts = request.repo_url.rstrip('/').split('/')
        repo_name = repo_parts[-1] if len(repo_parts) > 0 else "wiki"

        # Get current timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if request.format == "markdown":
            # Generate Markdown content
            content = generate_markdown_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.md"
            media_type = "text/markdown"
        else:  # JSON format
            # Generate JSON content
            content = generate_json_export(request.repo_url, request.pages)
            filename = f"{repo_name}_wiki_{timestamp}.json"
            media_type = "application/json"

        # Create response with appropriate headers for file download
        response = Response(
            content=content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename={filename}"
            }
        )

        return response

    except Exception as e:
        error_msg = f"Error exporting wiki: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/local_repo/structure")
async def get_local_repo_structure(path: str = Query(None, description="Path to local repository")):
    """Return the file tree and README content for a local repository."""
    if not path:
        return JSONResponse(
            status_code=400,
            content={"error": "No path provided. Please provide a 'path' query parameter."}
        )

    if not os.path.isdir(path):
        return JSONResponse(
            status_code=404,
            content={"error": f"Directory not found: {path}"}
        )

    try:
        logger.info(f"Processing local repository at: {path}")
        file_tree_lines = []
        readme_content = ""

        for root, dirs, files in os.walk(path):
            # Exclude hidden dirs/files and virtual envs
            dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__' and d != 'node_modules' and d != '.venv']
            for file in files:
                if file.startswith('.') or file == '__init__.py' or file == '.DS_Store':
                    continue
                rel_dir = os.path.relpath(root, path)
                rel_file = os.path.join(rel_dir, file) if rel_dir != '.' else file
                file_tree_lines.append(rel_file)
                # Find README.md (case-insensitive)
                if file.lower() == 'readme.md' and not readme_content:
                    try:
                        with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                            readme_content = f.read()
                    except Exception as e:
                        logger.warning(f"Could not read README.md: {str(e)}")
                        readme_content = ""

        file_tree_str = '\n'.join(sorted(file_tree_lines))
        return {"file_tree": file_tree_str, "readme": readme_content}
    except Exception as e:
        logger.error(f"Error processing local repository: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing local repository: {str(e)}"}
        )

def generate_markdown_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate Markdown export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        Markdown content as string
    """
    # Start with metadata
    markdown = f"# Wiki Documentation for {repo_url}\n\n"
    markdown += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    # Add table of contents
    markdown += "## Table of Contents\n\n"
    for page in pages:
        markdown += f"- [{page.title}](#{page.id})\n"
    markdown += "\n"

    # Add each page
    for page in pages:
        markdown += f"<a id='{page.id}'></a>\n\n"
        markdown += f"## {page.title}\n\n"



        # Add related pages
        if page.relatedPages and len(page.relatedPages) > 0:
            markdown += "### Related Pages\n\n"
            related_titles = []
            for related_id in page.relatedPages:
                # Find the title of the related page
                related_page = next((p for p in pages if p.id == related_id), None)
                if related_page:
                    related_titles.append(f"[{related_page.title}](#{related_id})")

            if related_titles:
                markdown += "Related topics: " + ", ".join(related_titles) + "\n\n"

        # Add page content
        markdown += f"{page.content}\n\n"
        markdown += "---\n\n"

    return markdown

def generate_json_export(repo_url: str, pages: List[WikiPage]) -> str:
    """
    Generate JSON export of wiki pages.

    Args:
        repo_url: The repository URL
        pages: List of wiki pages

    Returns:
        JSON content as string
    """
    # Create a dictionary with metadata and pages
    export_data = {
        "metadata": {
            "repository": repo_url,
            "generated_at": datetime.now().isoformat(),
            "page_count": len(pages)
        },
        "pages": [page.model_dump() for page in pages]
    }

    # Convert to JSON string with pretty formatting
    return json.dumps(export_data, indent=2)

# Import the simplified chat implementation
from api.simple_chat import chat_completions_stream
from api.websocket_wiki import handle_websocket_chat

# Add the chat_completions_stream endpoint to the main app
app.add_api_route("/chat/completions/stream", chat_completions_stream, methods=["POST"])

# Add the WebSocket endpoint
app.add_websocket_route("/ws/chat", handle_websocket_chat)

# Serve OpenAPI specification and Swagger UI
@app.get("/api/openapi.yaml", include_in_schema=False)
async def get_openapi_yaml():
    """Serve the OpenAPI specification file."""
    return FileResponse(
        path="api/openapi.yaml",
        media_type="text/yaml",
        filename="openapi.yaml"
    )

@app.get("/api/docs", include_in_schema=False)
async def get_swagger_ui():
    """Serve the Swagger UI HTML."""
    return FileResponse(
        path="api/swagger-ui.html",
        media_type="text/html"
    )

# --- Wiki Cache Helper Functions ---

WIKI_CACHE_DIR = os.path.join(get_adalflow_default_root_path(), "wikicache")
os.makedirs(WIKI_CACHE_DIR, exist_ok=True)

def get_wiki_cache_path(owner: str, repo: str, repo_type: str, language: str) -> str:
    """Generates the file path for a given wiki cache."""
    filename = f"deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json"
    return os.path.join(WIKI_CACHE_DIR, filename)

async def read_wiki_cache(owner: str, repo: str, repo_type: str, language: str) -> Optional[WikiCacheData]:
    """Reads wiki cache data from the file system."""
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return WikiCacheData(**data)
        except Exception as e:
            logger.error(f"Error reading wiki cache from {cache_path}: {e}")
            return None
    return None

async def save_wiki_cache(data: WikiCacheRequest) -> bool:
    """Saves wiki cache data to the file system."""
    cache_path = get_wiki_cache_path(data.owner, data.repo, data.repo_type, data.language)
    logger.info(f"Attempting to save wiki cache. Path: {cache_path}")
    try:
        payload = WikiCacheData(
            wiki_structure=data.wiki_structure,
            generated_pages=data.generated_pages,
            repo_url=data.repo_url
        )
        # Log size of data to be cached for debugging (avoid logging full content if large)
        try:
            payload_json = payload.model_dump_json()
            payload_size = len(payload_json.encode('utf-8'))
            logger.info(f"Payload prepared for caching. Size: {payload_size} bytes.")
        except Exception as ser_e:
            logger.warning(f"Could not serialize payload for size logging: {ser_e}")


        logger.info(f"Writing cache file to: {cache_path}")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(payload.model_dump(), f, indent=2)
        logger.info(f"Wiki cache successfully saved to {cache_path}")
        return True
    except IOError as e:
        logger.error(f"IOError saving wiki cache to {cache_path}: {e.strerror} (errno: {e.errno})", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error saving wiki cache to {cache_path}: {e}", exc_info=True)
        return False

# --- Wiki Cache API Endpoints ---

@app.get("/api/wiki_cache", response_model=Optional[WikiCacheData])
async def get_cached_wiki(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Retrieves cached wiki data (structure and generated pages) for a repository.
    """
    logger.info(f"Attempting to retrieve wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cached_data = await read_wiki_cache(owner, repo, repo_type, language)
    if cached_data:
        return cached_data
    else:
        # Return 200 with null body if not found, as frontend expects this behavior
        # Or, raise HTTPException(status_code=404, detail="Wiki cache not found") if preferred
        logger.info(f"Wiki cache not found for {owner}/{repo} ({repo_type}), lang: {language}")
        return None

@app.post("/api/wiki_cache")
async def store_wiki_cache(request_data: WikiCacheRequest):
    """
    Stores generated wiki data (structure and pages) to the server-side cache.
    """
    logger.info(f"Attempting to save wiki cache for {request_data.owner}/{request_data.repo} ({request_data.repo_type}), lang: {request_data.language}")
    success = await save_wiki_cache(request_data)
    if success:
        return {"message": "Wiki cache saved successfully"}
    else:
        raise HTTPException(status_code=500, detail="Failed to save wiki cache")

@app.delete("/api/wiki_cache")
async def delete_wiki_cache(
    owner: str = Query(..., description="Repository owner"),
    repo: str = Query(..., description="Repository name"),
    repo_type: str = Query(..., description="Repository type (e.g., github, gitlab)"),
    language: str = Query(..., description="Language of the wiki content")
):
    """
    Deletes a specific wiki cache from the file system.
    """
    logger.info(f"Attempting to delete wiki cache for {owner}/{repo} ({repo_type}), lang: {language}")
    cache_path = get_wiki_cache_path(owner, repo, repo_type, language)

    if os.path.exists(cache_path):
        try:
            os.remove(cache_path)
            logger.info(f"Successfully deleted wiki cache: {cache_path}")
            return {"message": f"Wiki cache for {owner}/{repo} ({language}) deleted successfully"}
        except Exception as e:
            logger.error(f"Error deleting wiki cache {cache_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete wiki cache: {str(e)}")
    else:
        logger.warning(f"Wiki cache not found, cannot delete: {cache_path}")
        raise HTTPException(status_code=404, detail="Wiki cache not found")

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker and monitoring"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "deepwiki-api"
    }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running"""
    return {
        "message": "Welcome to DeepWiki-Open API",
        "version": "1.0.0",
        "documentation": "/api/docs",
        "openapi_spec": "/api/openapi.yaml",
        "endpoints": {
            "Documentation": [
                "GET /api/docs - Swagger UI documentation",
                "GET /api/openapi.yaml - OpenAPI specification"
            ],
            "Chat": [
                "POST /chat/completions/stream - Streaming chat completion (HTTP)",
                "WebSocket /ws/chat - WebSocket chat completion",
            ],
            "Wiki": [
                "POST /export/wiki - Export wiki content as Markdown or JSON",
                "GET /api/wiki_cache - Retrieve cached wiki data",
                "POST /api/wiki_cache - Store wiki data to cache",
                "DELETE /api/wiki_cache - Delete wiki cache",
                "GET /api/processed_projects - List all processed projects"
            ],
            "LocalRepo": [
                "GET /local_repo/structure - Get structure of a local repository (with path parameter)",
            ],
            "LocalProject": [
                "POST /api/local_project/analyze - Analyze a local project directory",
                "GET /api/local_project/reports - List all stored local project reports",
                "GET /api/local_project/report/{id} - Get a specific local project report"
            ],
            "Models": [
                "GET /models/config - Get available model providers and their models"
            ],
            "Health": [
                "GET /health - Health check endpoint"
            ]
        }
    }

# --- Local Project Analysis Endpoints ---

@app.post("/api/local_project/analyze", response_model=Dict[str, Any])
async def analyze_local_project(request: LocalProjectAnalysisRequest):
    """
    Analyze a local project directory and generate a comprehensive report.
    
    Args:
        request: The analysis request containing the project path and options
        
    Returns:
        The generated project report
    """
    try:
        logger.info(f"Analyzing local project at {request.project_path}")
        
        # Validate project path
        if not os.path.isdir(request.project_path):
            raise HTTPException(
                status_code=400,
                detail=f"Project path does not exist or is not a directory: {request.project_path}"
            )
        
        # Analyze the project and store in RAG if requested
        report, rag_success = analyze_and_store_project(
            project_path=request.project_path,
            model_provider=request.model_provider,
            model_name=request.model_name,
            excluded_dirs=request.excluded_dirs,
            excluded_files=request.excluded_files
        )
        
        # Convert the report to a dictionary
        report_dict = report.to_dict()
        
        # Add RAG storage status
        report_dict["stored_in_rag"] = rag_success
        
        return report_dict
    
    except Exception as e:
        logger.error(f"Error analyzing local project: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing local project: {str(e)}"
        )

@app.get("/api/local_project/reports", response_model=List[LocalProjectReportSummary])
async def list_local_project_reports():
    """
    List all stored local project reports.
    
    Returns:
        A list of local project report summaries
    """
    try:
        # Get the path to the database directory
        root_path = get_adalflow_default_root_path()
        db_dir = os.path.join(root_path, "databases")
        
        # Find all local project report databases
        report_summaries = []
        pattern = os.path.join(db_dir, "local_*.pkl")
        
        # Import pickle for loading the database files
        import pickle
        
        for db_path in glob.glob(pattern):
            try:
                # Extract the project ID from the filename
                filename = os.path.basename(db_path)
                project_id = os.path.splitext(filename)[0]
                
                # Extract project name and timestamp from the ID
                # Format: local_<project_name>_<timestamp>
                parts = project_id.split('_')
                if len(parts) >= 3:
                    project_name = '_'.join(parts[1:-1])
                    timestamp = parts[-1]
                    
                    # Try to load the actual data from the database
                    try:
                        with open(db_path, 'rb') as f:
                            db_data = pickle.load(f)
                        
                        # Extract the project report from the database
                        project_report = None
                        
                        # Try to find a ProjectReport object in the database
                        if hasattr(db_data, 'get') and callable(db_data.get):
                            # If db_data is a dictionary-like object, look for the report
                            if 'report' in db_data:
                                project_report = db_data['report']
                            elif 'project_report' in db_data:
                                project_report = db_data['project_report']
                        elif hasattr(db_data, 'to_dict') and callable(db_data.to_dict):
                            # If db_data is a ProjectReport object or has a to_dict method
                            project_report = db_data
                        
                        # If we found a project report, extract the summary information
                        if project_report:
                            # Get the report data
                            if hasattr(project_report, 'to_dict') and callable(project_report.to_dict):
                                report_data = project_report.to_dict()
                            elif isinstance(project_report, dict):
                                report_data = project_report
                            else:
                                # If we can't get the report data, use default values
                                report_data = {
                                    'total_files': 0,
                                    'total_lines_of_code': 0,
                                    'total_size_bytes': 0,
                                    'languages': {}
                                }
                            
                            # Create the report summary
                            report_summaries.append(
                                LocalProjectReportSummary(
                                    id=project_id,
                                    project_name=project_name,
                                    analysis_timestamp=datetime.fromtimestamp(int(timestamp)).isoformat(),
                                    total_files=report_data.get('total_files', 0),
                                    total_lines_of_code=report_data.get('total_lines_of_code', 0),
                                    total_size_bytes=report_data.get('total_size_bytes', 0),
                                    languages=report_data.get('languages', {})
                                )
                            )
                        else:
                            # If we couldn't find a project report, use default values
                            logger.warning(f"Could not find ProjectReport in database: {project_id}")
                            report_summaries.append(
                                LocalProjectReportSummary(
                                    id=project_id,
                                    project_name=project_name,
                                    analysis_timestamp=datetime.fromtimestamp(int(timestamp)).isoformat(),
                                    total_files=0,
                                    total_lines_of_code=0,
                                    total_size_bytes=0,
                                    languages={}
                                )
                            )
                    except (pickle.PickleError, EOFError) as pe:
                        logger.warning(f"Error unpickling database file {db_path}: {pe}")
                        # Add a basic entry with default values
                        report_summaries.append(
                            LocalProjectReportSummary(
                                id=project_id,
                                project_name=project_name,
                                analysis_timestamp=datetime.fromtimestamp(int(timestamp)).isoformat(),
                                total_files=0,
                                total_lines_of_code=0,
                                total_size_bytes=0,
                                languages={}
                            )
                        )
            except Exception as e:
                logger.warning(f"Error processing report database {db_path}: {e}")
        
        # Sort by timestamp (newest first)
        report_summaries.sort(key=lambda x: x.analysis_timestamp, reverse=True)
        
        return report_summaries
    
    except Exception as e:
        logger.error(f"Error listing local project reports: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error listing local project reports: {str(e)}"
        )

@app.get("/api/local_project/report/{id}", response_model=Dict[str, Any])
async def get_local_project_report(id: str):
    """
    Get a specific local project report.
    
    Args:
        id: The ID of the report to retrieve
        
    Returns:
        The project report as JSON
    """
    try:
        # Get the path to the database file
        root_path = get_adalflow_default_root_path()
        db_path = os.path.join(root_path, "databases", f"{id}.pkl")
        
        # Check if the database exists
        if not os.path.exists(db_path):
            raise HTTPException(
                status_code=404,
                detail=f"Report not found: {id}"
            )
        
        # Load the database using pickle
        import pickle
        try:
            with open(db_path, 'rb') as f:
                db_data = pickle.load(f)
                
            # Extract the project report from the database
            # The database structure might vary, so we need to handle different formats
            project_report = None
            
            # Try to find a ProjectReport object in the database
            if hasattr(db_data, 'get') and callable(db_data.get):
                # If db_data is a dictionary-like object, look for the report
                if 'report' in db_data:
                    project_report = db_data['report']
                elif 'project_report' in db_data:
                    project_report = db_data['project_report']
            elif hasattr(db_data, 'to_dict') and callable(db_data.to_dict):
                # If db_data is a ProjectReport object or has a to_dict method
                project_report = db_data
            else:
                # If we can't find a ProjectReport, return the raw data
                logger.warning(f"Could not find ProjectReport in database: {id}")
                return {
                    "id": id,
                    "raw_data": str(db_data)
                }
            
            # Convert the project report to a dictionary
            if hasattr(project_report, 'to_dict') and callable(project_report.to_dict):
                report_dict = project_report.to_dict()
            elif isinstance(project_report, dict):
                report_dict = project_report
            else:
                logger.warning(f"Could not convert ProjectReport to dictionary: {id}")
                return {
                    "id": id,
                    "raw_data": str(project_report)
                }
            
            # Add the report ID to the dictionary
            report_dict['id'] = id
            
            # Ensure all values are JSON serializable
            def make_json_serializable(obj):
                if isinstance(obj, dict):
                    return {k: make_json_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_json_serializable(item) for item in obj]
                elif isinstance(obj, (str, int, float, bool, type(None))):
                    return obj
                else:
                    # Convert non-serializable objects to strings
                    return str(obj)
            
            # Process the report dictionary to ensure all values are JSON serializable
            report_dict = make_json_serializable(report_dict)
            
            return report_dict
            
        except (pickle.PickleError, EOFError) as pe:
            logger.error(f"Error unpickling database file {db_path}: {pe}")
            raise HTTPException(
                status_code=500,
                detail=f"Error reading report data: {str(pe)}"
            )
    
    except HTTPException:
        raise
    
    except Exception as e:
        logger.error(f"Error retrieving local project report: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving local project report: {str(e)}"
        )

# --- Processed Projects Endpoint --- (New Endpoint)
@app.get("/api/processed_projects", response_model=List[ProcessedProjectEntry])
async def get_processed_projects():
    """
    Lists all processed projects found in the wiki cache directory.
    Projects are identified by files named like: deepwiki_cache_{repo_type}_{owner}_{repo}_{language}.json
    """
    project_entries: List[ProcessedProjectEntry] = []
    # WIKI_CACHE_DIR is already defined globally in the file

    try:
        if not os.path.exists(WIKI_CACHE_DIR):
            logger.info(f"Cache directory {WIKI_CACHE_DIR} not found. Returning empty list.")
            return []

        logger.info(f"Scanning for project cache files in: {WIKI_CACHE_DIR}")
        filenames = await asyncio.to_thread(os.listdir, WIKI_CACHE_DIR) # Use asyncio.to_thread for os.listdir

        for filename in filenames:
            if filename.startswith("deepwiki_cache_") and filename.endswith(".json"):
                file_path = os.path.join(WIKI_CACHE_DIR, filename)
                try:
                    stats = await asyncio.to_thread(os.stat, file_path) # Use asyncio.to_thread for os.stat
                    parts = filename.replace("deepwiki_cache_", "").replace(".json", "").split('_')

                    # Expecting repo_type_owner_repo_language
                    # Example: deepwiki_cache_github_AsyncFuncAI_deepwiki-open_en.json
                    # parts = [github, AsyncFuncAI, deepwiki-open, en]
                    if len(parts) >= 4:
                        repo_type = parts[0]
                        owner = parts[1]
                        language = parts[-1] # language is the last part
                        repo = "_".join(parts[2:-1]) # repo can contain underscores

                        project_entries.append(
                            ProcessedProjectEntry(
                                id=filename,
                                owner=owner,
                                repo=repo,
                                name=f"{owner}/{repo}",
                                repo_type=repo_type,
                                submittedAt=int(stats.st_mtime * 1000), # Convert to milliseconds
                                language=language
                            )
                        )
                    else:
                        logger.warning(f"Could not parse project details from filename: {filename}")
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
                    continue # Skip this file on error

        # Sort by most recent first
        project_entries.sort(key=lambda p: p.submittedAt, reverse=True)
        logger.info(f"Found {len(project_entries)} processed project entries.")
        return project_entries

    except Exception as e:
        logger.error(f"Error listing processed projects from {WIKI_CACHE_DIR}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list processed projects from server cache.")
