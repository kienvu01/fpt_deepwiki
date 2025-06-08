import os
import logging
import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import re
import adalflow as adal
from adalflow.core.types import Document

from api.data_pipeline import (
    read_all_documents,
    DatabaseManager,
    count_tokens,
    transform_documents_and_save_to_db
)
from api.rag import RAG
from api.config import configs, get_embedder_config, is_ollama_embedder
from api.tools.embedder import get_embedder

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class FileStats:
    """Statistics about a file in the project."""
    path: str
    size_bytes: int
    lines_of_code: int
    token_count: int
    language: str
    is_test: bool = False
    is_config: bool = False
    is_documentation: bool = False

@dataclass
class DirectoryStats:
    """Statistics about a directory in the project."""
    path: str
    file_count: int
    total_size_bytes: int
    total_lines_of_code: int
    languages: Dict[str, int] = field(default_factory=dict)
    
@dataclass
class ProjectReport:
    """Comprehensive report about a project."""
    project_name: str
    analysis_timestamp: str
    total_files: int
    total_size_bytes: int
    total_lines_of_code: int
    languages: Dict[str, int] = field(default_factory=dict)
    file_stats: List[FileStats] = field(default_factory=list)
    directory_stats: List[DirectoryStats] = field(default_factory=list)
    top_files_by_size: List[str] = field(default_factory=list)
    top_files_by_loc: List[str] = field(default_factory=list)
    readme_content: str = ""
    project_structure: str = ""
    summary: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Convert the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def save_to_file(self, file_path: str) -> None:
        """Save the report to a file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

def detect_language(file_path: str) -> str:
    """
    Detect the programming language of a file based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        The detected language or "Unknown"
    """
    extension_map = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript (React)',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript (React)',
        '.html': 'HTML',
        '.css': 'CSS',
        '.scss': 'SCSS',
        '.sass': 'Sass',
        '.less': 'Less',
        '.java': 'Java',
        '.c': 'C',
        '.cpp': 'C++',
        '.h': 'C/C++ Header',
        '.hpp': 'C++ Header',
        '.cs': 'C#',
        '.go': 'Go',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.rs': 'Rust',
        '.scala': 'Scala',
        '.m': 'Objective-C',
        '.mm': 'Objective-C++',
        '.pl': 'Perl',
        '.sh': 'Shell',
        '.bash': 'Bash',
        '.zsh': 'Zsh',
        '.fish': 'Fish',
        '.sql': 'SQL',
        '.r': 'R',
        '.json': 'JSON',
        '.xml': 'XML',
        '.yaml': 'YAML',
        '.yml': 'YAML',
        '.toml': 'TOML',
        '.ini': 'INI',
        '.md': 'Markdown',
        '.rst': 'reStructuredText',
        '.tex': 'LaTeX',
        '.dockerfile': 'Dockerfile',
        '.vue': 'Vue',
        '.svelte': 'Svelte',
        '.dart': 'Dart',
        '.lua': 'Lua',
        '.ex': 'Elixir',
        '.exs': 'Elixir',
        '.erl': 'Erlang',
        '.hrl': 'Erlang',
        '.clj': 'Clojure',
        '.elm': 'Elm',
        '.hs': 'Haskell',
        '.fs': 'F#',
        '.fsx': 'F#',
        '.ml': 'OCaml',
        '.mli': 'OCaml',
    }
    
    _, ext = os.path.splitext(file_path.lower())
    if not ext and file_path.lower().endswith('dockerfile'):
        return 'Dockerfile'
    
    return extension_map.get(ext, "Unknown")

def count_lines_of_code(file_path: str) -> int:
    """
    Count the number of lines of code in a file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        Number of lines of code
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except Exception as e:
        logger.warning(f"Error counting lines in {file_path}: {e}")
        return 0

def is_test_file(file_path: str) -> bool:
    """
    Determine if a file is a test file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a test file, False otherwise
    """
    test_patterns = [
        r'test_.*\.py$',
        r'.*_test\.py$',
        r'.*\.test\.(js|jsx|ts|tsx)$',
        r'.*spec\.(js|jsx|ts|tsx)$',
        r'.*Test\.(java|kt|cs|scala)$',
        r'.*_test\.(go|rb|rs)$',
        r'test/.*',
        r'tests/.*',
        r'spec/.*',
    ]
    
    for pattern in test_patterns:
        if re.search(pattern, file_path):
            return True
    
    return False

def is_config_file(file_path: str) -> bool:
    """
    Determine if a file is a configuration file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a configuration file, False otherwise
    """
    config_patterns = [
        r'.*\.json$',
        r'.*\.yaml$',
        r'.*\.yml$',
        r'.*\.toml$',
        r'.*\.ini$',
        r'.*\.conf$',
        r'.*\.config$',
        r'.*\.properties$',
        r'.*\.env$',
        r'.*\.lock$',
        r'.*rc$',
        r'Dockerfile',
        r'docker-compose\.yml',
        r'package\.json',
        r'tsconfig\.json',
        r'webpack\.config\.js',
        r'babel\.config\.js',
        r'jest\.config\.js',
        r'\.gitignore',
        r'\.dockerignore',
        r'requirements\.txt',
        r'Pipfile',
        r'pyproject\.toml',
        r'setup\.py',
        r'setup\.cfg',
        r'CMakeLists\.txt',
        r'Makefile',
    ]
    
    for pattern in config_patterns:
        if re.search(pattern, file_path):
            return True
    
    return False

def is_documentation_file(file_path: str) -> bool:
    """
    Determine if a file is a documentation file.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file is a documentation file, False otherwise
    """
    doc_patterns = [
        r'.*\.md$',
        r'.*\.rst$',
        r'.*\.txt$',
        r'.*\.adoc$',
        r'.*\.asciidoc$',
        r'.*\.tex$',
        r'.*\.wiki$',
        r'.*\.rdoc$',
        r'README.*',
        r'LICENSE.*',
        r'CONTRIBUTING.*',
        r'CHANGELOG.*',
        r'AUTHORS.*',
        r'NOTICE.*',
        r'docs/.*',
        r'documentation/.*',
        r'wiki/.*',
    ]
    
    for pattern in doc_patterns:
        if re.search(pattern, file_path):
            return True
    
    return False

def analyze_project(project_path: str, excluded_dirs: List[str] = None, excluded_files: List[str] = None) -> ProjectReport:
    """
    Analyze a project directory and generate a comprehensive report.
    
    Args:
        project_path: Path to the project directory
        excluded_dirs: List of directories to exclude from analysis
        excluded_files: List of file patterns to exclude from analysis
        
    Returns:
        A ProjectReport object containing the analysis results
    """
    if excluded_dirs is None:
        excluded_dirs = [
            '.git', '.github', 'node_modules', 'venv', '.venv', 'env',
            '__pycache__', '.pytest_cache', '.idea', '.vscode', 'dist',
            'build', 'target', 'out', 'bin', 'obj', 'coverage'
        ]
    
    if excluded_files is None:
        excluded_files = [
            '*.min.js', '*.min.css', '*.map', '*.pyc', '*.pyo', '*.pyd',
            '*.so', '*.dylib', '*.dll', '*.exe', '*.o', '*.a', '*.lib',
            '*.zip', '*.tar', '*.gz', '*.rar', '*.jar', '*.war', '*.ear',
            '*.class', '*.log', '*.tmp', '*.bak', '*.swp', '*.DS_Store'
        ]
    
    # Initialize report
    project_name = os.path.basename(os.path.abspath(project_path))
    report = ProjectReport(
        project_name=project_name,
        analysis_timestamp=datetime.now().isoformat(),
        total_files=0,
        total_size_bytes=0,
        total_lines_of_code=0,
        languages={},
    )
    
    # Track directory statistics
    dir_stats = {}
    
    # Find README file
    readme_path = None
    for root, _, files in os.walk(project_path):
        for file in files:
            if file.lower().startswith('readme.'):
                readme_path = os.path.join(root, file)
                break
        if readme_path:
            break
    
    # Read README content if found
    if readme_path:
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                report.readme_content = f.read()
        except Exception as e:
            logger.warning(f"Error reading README file: {e}")
    
    # Generate project structure
    structure_lines = []
    for root, dirs, files in os.walk(project_path):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not any(d.startswith(ed) for ed in excluded_dirs)]
        
        # Calculate relative path
        rel_path = os.path.relpath(root, project_path)
        if rel_path == '.':
            rel_path = ''
        
        # Add directory to structure
        if rel_path:
            depth = rel_path.count(os.sep)
            structure_lines.append('  ' * depth + '└── ' + os.path.basename(root) + '/')
        
        # Initialize directory stats
        if rel_path not in dir_stats:
            dir_stats[rel_path] = DirectoryStats(
                path=rel_path if rel_path else '.',
                file_count=0,
                total_size_bytes=0,
                total_lines_of_code=0,
                languages={}
            )
        
        # Process files
        for file in sorted(files):
            # Skip excluded files
            if any(re.match(pattern.replace('*', '.*'), file) for pattern in excluded_files):
                continue
            
            file_path = os.path.join(root, file)
            rel_file_path = os.path.relpath(file_path, project_path)
            
            # Add file to structure
            if rel_path:
                depth = rel_path.count(os.sep) + 1
            else:
                depth = 0
            structure_lines.append('  ' * depth + '└── ' + file)
            
            # Get file stats
            try:
                size_bytes = os.path.getsize(file_path)
                lines_of_code = count_lines_of_code(file_path)
                language = detect_language(file_path)
                
                # Count tokens
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    token_count = count_tokens(content)
                except Exception as e:
                    logger.warning(f"Error counting tokens in {file_path}: {e}")
                    token_count = 0
                
                # Create file stats
                file_stat = FileStats(
                    path=rel_file_path,
                    size_bytes=size_bytes,
                    lines_of_code=lines_of_code,
                    token_count=token_count,
                    language=language,
                    is_test=is_test_file(rel_file_path),
                    is_config=is_config_file(rel_file_path),
                    is_documentation=is_documentation_file(rel_file_path)
                )
                report.file_stats.append(file_stat)
                
                # Update directory stats
                dir_stats[rel_path].file_count += 1
                dir_stats[rel_path].total_size_bytes += size_bytes
                dir_stats[rel_path].total_lines_of_code += lines_of_code
                dir_stats[rel_path].languages[language] = dir_stats[rel_path].languages.get(language, 0) + 1
                
                # Update project stats
                report.total_files += 1
                report.total_size_bytes += size_bytes
                report.total_lines_of_code += lines_of_code
                report.languages[language] = report.languages.get(language, 0) + 1
                
            except Exception as e:
                logger.warning(f"Error processing file {file_path}: {e}")
    
    # Add directory stats to report
    report.directory_stats = list(dir_stats.values())
    
    # Set project structure
    report.project_structure = '\n'.join(structure_lines)
    
    # Get top files by size
    report.top_files_by_size = [
        f.path for f in sorted(report.file_stats, key=lambda x: x.size_bytes, reverse=True)[:10]
    ]
    
    # Get top files by lines of code
    report.top_files_by_loc = [
        f.path for f in sorted(report.file_stats, key=lambda x: x.lines_of_code, reverse=True)[:10]
    ]
    
    return report

def generate_project_summary(report: ProjectReport, model_provider: str = "google", model_name: str = None) -> str:
    """
    Generate a summary of the project using an AI model.
    
    Args:
        report: The project report
        model_provider: The model provider to use (google, openai, openrouter, ollama)
        model_name: The model name to use with the provider
        
    Returns:
        A summary of the project
    """
    # Create a RAG instance
    rag = RAG(provider=model_provider, model=model_name)
    
    # Create a prompt for the model
    prompt = f"""
    Generate a comprehensive summary of the following software project:
    
    Project Name: {report.project_name}
    Total Files: {report.total_files}
    Total Lines of Code: {report.total_lines_of_code}
    
    Languages Used:
    {json.dumps(report.languages, indent=2)}
    
    Top Files by Size:
    {json.dumps(report.top_files_by_size, indent=2)}
    
    Top Files by Lines of Code:
    {json.dumps(report.top_files_by_loc, indent=2)}
    
    Project Structure:
    {report.project_structure}
    
    README Content:
    {report.readme_content}
    
    Based on this information, provide a detailed summary of the project that includes:
    1. The main purpose and functionality of the project
    2. The key technologies and languages used
    3. The architecture and organization of the codebase
    4. Any notable patterns or design choices
    5. Potential areas for improvement or optimization
    
    Format your response as a well-structured markdown document with appropriate headings and sections.
    """
    
    # Get the response from the model
    try:
        # Use the RAG system to generate a response
        # Since we're not using retrieval here, we'll just use the call method
        response = rag.call(query=prompt)
        
        # Extract the answer from the response
        if hasattr(response, 'answer'):
            return response.answer
        elif isinstance(response, dict) and 'answer' in response:
            return response['answer']
        else:
            return str(response)
    except Exception as e:
        logger.error(f"Error generating project summary: {e}")
        return "Error generating project summary. Please try again later."

def store_report_in_rag(report: ProjectReport, project_path: str) -> bool:
    """
    Store the project report in the RAG system.
    
    Args:
        report: The project report
        project_path: Path to the project directory
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the database manager
        db_manager = DatabaseManager()
        
        # Create documents from the report
        documents = []
        
        # Add the summary as a document
        if report.summary:
            summary_doc = Document(
                text=report.summary,
                meta_data={
                    "file_path": "project_summary.md",
                    "type": "md",
                    "is_code": False,
                    "is_implementation": False,
                    "title": f"{report.project_name} Summary",
                    "token_count": count_tokens(report.summary),
                }
            )
            documents.append(summary_doc)
        
        # Add the README as a document
        if report.readme_content:
            readme_doc = Document(
                text=report.readme_content,
                meta_data={
                    "file_path": "README.md",
                    "type": "md",
                    "is_code": False,
                    "is_implementation": False,
                    "title": f"{report.project_name} README",
                    "token_count": count_tokens(report.readme_content),
                }
            )
            documents.append(readme_doc)
        
        # Add the project structure as a document
        if report.project_structure:
            structure_doc = Document(
                text=report.project_structure,
                meta_data={
                    "file_path": "project_structure.txt",
                    "type": "txt",
                    "is_code": False,
                    "is_implementation": False,
                    "title": f"{report.project_name} Structure",
                    "token_count": count_tokens(report.project_structure),
                }
            )
            documents.append(structure_doc)
        
        # Add the report statistics as a document
        stats_text = json.dumps({
            "project_name": report.project_name,
            "total_files": report.total_files,
            "total_size_bytes": report.total_size_bytes,
            "total_lines_of_code": report.total_lines_of_code,
            "languages": report.languages,
            "top_files_by_size": report.top_files_by_size,
            "top_files_by_loc": report.top_files_by_loc,
        }, indent=2)
        
        stats_doc = Document(
            text=stats_text,
            meta_data={
                "file_path": "project_stats.json",
                "type": "json",
                "is_code": False,
                "is_implementation": False,
                "title": f"{report.project_name} Statistics",
                "token_count": count_tokens(stats_text),
            }
        )
        documents.append(stats_doc)
        
        # Create a unique identifier for the project
        project_id = f"local_{os.path.basename(os.path.abspath(project_path))}_{int(time.time())}"
        
        # Create a database for the project
        is_ollama = is_ollama_embedder()
        transformed_docs = db_manager._create_repo(project_id)
        
        # Transform the documents and save to the database
        db_path = db_manager.repo_paths["save_db_file"]
        db = transform_documents_and_save_to_db(documents, db_path, is_ollama_embedder=is_ollama)
        
        logger.info(f"Project report stored in RAG system at {db_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error storing project report in RAG: {e}")
        return False

def analyze_and_store_project(project_path: str, model_provider: str = "google", model_name: str = None,
                             excluded_dirs: List[str] = None, excluded_files: List[str] = None) -> Tuple[ProjectReport, bool]:
    """
    Analyze a project directory, generate a report, and store it in the RAG system.
    
    Args:
        project_path: Path to the project directory
        model_provider: The model provider to use for summary generation
        model_name: The model name to use with the provider
        excluded_dirs: List of directories to exclude from analysis
        excluded_files: List of file patterns to exclude from analysis
        
    Returns:
        A tuple of (ProjectReport, bool) where the bool indicates if the report was successfully stored in RAG
    """
    # Analyze the project
    logger.info(f"Analyzing project at {project_path}")
    report = analyze_project(project_path, excluded_dirs, excluded_files)
    
    # Generate a summary
    logger.info("Generating project summary")
    summary = generate_project_summary(report, model_provider, model_name)
    report.summary = summary
    
    # Store the report in RAG
    logger.info("Storing project report in RAG")
    success = store_report_in_rag(report, project_path)
    
    return report, success
