"""
Module for generating wiki documentation from local project analysis.
"""

import os
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

from api.local_report import analyze_project, generate_project_summary, store_report_in_rag
from api.rag import RAG
from api.data_pipeline import DatabaseManager
from api.wiki_structure import WikiStructure, WikiPage

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class LocalProjectWiki:
    """Class for generating wiki documentation from local project analysis."""
    project_path: str
    model_provider: str = "google"
    model_name: Optional[str] = None
    excluded_dirs: Optional[List[str]] = None
    excluded_files: Optional[List[str]] = None

    def __post_init__(self):
        """Initialize the RAG system and database manager."""
        self.rag = RAG(provider=self.model_provider, model=self.model_name)
        self.db_manager = DatabaseManager()

    def generate_wiki(self) -> Tuple[WikiStructure, Dict[str, WikiPage]]:
        """
        Generate wiki documentation from local project analysis.

        Returns:
            Tuple[WikiStructure, Dict[str, WikiPage]]: The wiki structure and generated pages
        """
        try:
            # Analyze the project
            logger.info(f"Analyzing project at {self.project_path}")
            report = analyze_project(
                self.project_path,
                excluded_dirs=self.excluded_dirs,
                excluded_files=self.excluded_files
            )

            # Generate project summary
            logger.info("Generating project summary")
            summary = generate_project_summary(report, self.model_provider, self.model_name)
            report.summary = summary

            # Store the report in RAG
            logger.info("Storing project report in RAG")
            success = store_report_in_rag(report, self.project_path)
            if not success:
                logger.warning("Failed to store report in RAG system")

            # Create wiki structure
            wiki_structure = WikiStructure(
                id=f"local_{os.path.basename(self.project_path)}",
                title=report.project_name,
                description=report.summary,
                pages=[]
            )

            # Generate overview page
            overview_page = WikiPage(
                id="overview",
                title="Project Overview",
                content=report.summary,
                filePaths=[],
                importance="high",
                relatedPages=[],
                isSection=True,
                children=[]
            )
            wiki_structure.pages.append(overview_page)

            # Generate architecture page
            architecture_content = self._generate_architecture_page(report)
            architecture_page = WikiPage(
                id="architecture",
                title="Architecture",
                content=architecture_content,
                filePaths=[],
                importance="high",
                relatedPages=["overview"],
                isSection=True,
                children=[]
            )
            wiki_structure.pages.append(architecture_page)

            # Generate pages for each major component/directory
            for dir_stat in report.directory_stats:
                if dir_stat.path == '.':
                    continue

                # Skip empty directories
                if dir_stat.file_count == 0:
                    continue

                # Generate content for directory
                content = self._generate_directory_page(dir_stat, report)
                
                # Split path into parts to handle nested directories
                path_parts = dir_stat.path.split('/')
                
                if len(path_parts) > 1:
                    # This is a nested directory, set up parent-child relationship
                    parent_path = '/'.join(path_parts[:-1])
                    parent_id = parent_path.replace('/', '_')
                    
                    page = WikiPage(
                        id=dir_stat.path.replace('/', '_'),
                        title=os.path.basename(dir_stat.path),
                        content=content,
                        filePaths=[f.path for f in report.file_stats if f.path.startswith(dir_stat.path)],
                        importance="medium",
                        relatedPages=["architecture"],
                        parentId=parent_id,
                        isSection=False,
                        children=[]
                    )
                    
                    # Find parent page and add this page as a child
                    for p in wiki_structure.pages:
                        if p.id == parent_id:
                            p.children.append(page.id)
                            break
                else:
                    # Top-level directory
                    page = WikiPage(
                        id=dir_stat.path.replace('/', '_'),
                        title=os.path.basename(dir_stat.path),
                        content=content,
                        filePaths=[f.path for f in report.file_stats if f.path.startswith(dir_stat.path)],
                        importance="medium",
                        relatedPages=["architecture"],
                        isSection=True,
                        children=[]
                    )
                
                wiki_structure.pages.append(page)

            # Create pages dictionary
            generated_pages = {page.id: page for page in wiki_structure.pages}

            return wiki_structure, generated_pages

        except Exception as e:
            logger.error(f"Error generating wiki: {e}")
            raise

    def _generate_architecture_page(self, report) -> str:
        """Generate content for the architecture page."""
        content = f"""# Architecture Overview

## Project Structure

The project is organized into the following structure:

```
{report.project_structure}
```

## Key Components

The project consists of {report.total_files} files across {len(report.languages)} programming languages:

"""
        # Add language statistics
        for lang, count in sorted(report.languages.items(), key=lambda x: x[1], reverse=True):
            content += f"- {lang}: {count} files\n"

        content += "\n## Implementation Details\n\n"

        # Add information about major directories
        for dir_stat in sorted(report.directory_stats, key=lambda x: x.file_count, reverse=True):
            if dir_stat.path == '.' or dir_stat.file_count == 0:
                continue
            content += f"### {dir_stat.path}\n\n"
            content += f"- Files: {dir_stat.file_count}\n"
            content += f"- Lines of Code: {dir_stat.total_lines_of_code:,}\n"
            content += f"- Size: {dir_stat.total_size_bytes / 1024:.1f} KB\n"
            if dir_stat.languages:
                content += "- Languages:\n"
                for lang, count in sorted(dir_stat.languages.items(), key=lambda x: x[1], reverse=True):
                    content += f"  - {lang}: {count} files\n"
            content += "\n"

        return content

    def _generate_directory_page(self, dir_stat, report) -> str:
        """Generate content for a directory page."""
        content = f"""# {os.path.basename(dir_stat.path)}

This directory contains {dir_stat.file_count} files with a total of {dir_stat.total_lines_of_code:,} lines of code.

## Files

The following files are included in this directory:

"""
        # Add file information
        for file_stat in sorted(report.file_stats, key=lambda x: x.lines_of_code, reverse=True):
            if not file_stat.path.startswith(dir_stat.path):
                continue

            content += f"### {os.path.basename(file_stat.path)}\n\n"
            content += f"- Path: `{file_stat.path}`\n"
            content += f"- Language: {file_stat.language}\n"
            content += f"- Lines of Code: {file_stat.lines_of_code:,}\n"
            content += f"- Size: {file_stat.size_bytes / 1024:.1f} KB\n"

            if file_stat.is_test:
                content += "- Type: Test File\n"
            elif file_stat.is_config:
                content += "- Type: Configuration File\n"
            elif file_stat.is_documentation:
                content += "- Type: Documentation\n"
            content += "\n"

        return content
