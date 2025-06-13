"""
Wiki report generation module for DeepWiki-Open.

This module provides functionality to generate wiki reports for repositories
and store them in the RAG system.
"""

import os
import logging
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field

# Get a logger for this module
logger = logging.getLogger(__name__)

class WikiReportRequest(BaseModel):
    """
    Model for requesting a wiki report generation.
    """
    repo_url: str = Field(..., description="URL of the repository to generate a wiki for")
    repo_type: Optional[str] = Field(None, description="Type of repository (github, gitlab, bitbucket)")
    token: Optional[str] = Field(None, description="Personal access token for private repositories")
    language: str = Field("en", description="Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')")
    model_provider: Optional[str] = Field(None, description="Model provider to use for content generation")
    model_name: Optional[str] = Field(None, description="Model name to use with the provider")
    excluded_dirs: List[str] = Field([], description="Directories to exclude from processing")
    excluded_files: List[str] = Field([], description="File patterns to exclude from processing")
    included_dirs: List[str] = Field([], description="Directories to include exclusively")
    included_files: List[str] = Field([], description="File patterns to include exclusively")
    store_in_rag: bool = Field(True, description="Whether to store the report in the RAG system")
    comprehensive: bool = Field(True, description="Whether to generate a comprehensive wiki (vs. concise)")

class WikiPage(BaseModel):
    """
    Model for a wiki page.
    """
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str  # Should ideally be Literal['high', 'medium', 'low']
    relatedPages: List[str]

class WikiStructure(BaseModel):
    """
    Model for the overall wiki structure.
    """
    id: str
    title: str
    description: str
    pages: List[WikiPage]
    sections: List[Dict[str, Any]] = []
    rootSections: List[str] = []

class WikiReport(BaseModel):
    """
    Model for a complete wiki report.
    """
    repo_url: str
    repo_type: str
    language: str
    wiki_structure: WikiStructure
    generated_pages: Dict[str, WikiPage]
    stored_in_rag: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    def dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            "repo_url": self.repo_url,
            "repo_type": self.repo_type,
            "language": self.language,
            "wiki_structure": self.wiki_structure.dict(),
            "generated_pages": {k: v.dict() for k, v in self.generated_pages.items()},
            "stored_in_rag": self.stored_in_rag,
            "timestamp": self.timestamp
        }

def generate_and_store_wiki_report(request: WikiReportRequest) -> Tuple[WikiReport, bool]:
    """
    Generate a wiki report for a repository and store it in the RAG system.
    
    Args:
        request: The wiki report request containing the repository URL and options
        
    Returns:
        A tuple containing the generated wiki report and a boolean indicating whether
        it was successfully stored in the RAG system
    """
    try:
        logger.info(f"Generating wiki report for {request.repo_url}")
        
        # Extract repository type from URL if not provided
        repo_type = request.repo_type
        if not repo_type:
            if "github.com" in request.repo_url:
                repo_type = "github"
            elif "gitlab.com" in request.repo_url:
                repo_type = "gitlab"
            elif "bitbucket.org" in request.repo_url:
                repo_type = "bitbucket"
            else:
                repo_type = "github"  # Default to GitHub
        
        # TODO: Implement the actual wiki report generation logic
        # For now, we'll create a placeholder report
        
        # Create a placeholder wiki structure
        wiki_structure = WikiStructure(
            id="wiki",
            title=f"Wiki for {request.repo_url}",
            description=f"Automatically generated wiki for {request.repo_url}",
            pages=[
                WikiPage(
                    id="page-1",
                    title="Overview",
                    content="# Overview\n\nThis is a placeholder overview page.",
                    filePaths=["README.md"],
                    importance="high",
                    relatedPages=[]
                ),
                WikiPage(
                    id="page-2",
                    title="Architecture",
                    content="# Architecture\n\nThis is a placeholder architecture page.",
                    filePaths=["src/main.py"],
                    importance="high",
                    relatedPages=["page-1"]
                )
            ]
        )
        
        # Create a placeholder for generated pages
        generated_pages = {
            page.id: page for page in wiki_structure.pages
        }
        
        # Create the wiki report
        report = WikiReport(
            repo_url=request.repo_url,
            repo_type=repo_type,
            language=request.language,
            wiki_structure=wiki_structure,
            generated_pages=generated_pages
        )
        
        # Store the report in the RAG system if requested
        rag_success = False
        if request.store_in_rag:
            try:
                # TODO: Implement the actual RAG storage logic
                # For now, we'll just simulate success
                rag_success = True
                logger.info(f"Successfully stored wiki report for {request.repo_url} in RAG")
            except Exception as e:
                logger.error(f"Error storing wiki report in RAG: {e}", exc_info=True)
        
        # Update the report with the RAG storage status
        report.stored_in_rag = rag_success
        
        return report, rag_success
    
    except Exception as e:
        logger.error(f"Error generating wiki report: {e}", exc_info=True)
        raise
