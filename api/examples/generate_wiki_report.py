"""
Example script for generating a wiki report using the backend API.

This script demonstrates how to use the wiki report generation API to generate
a wiki report for a repository and store it in the RAG system.
"""

import os
import sys
import json
import requests
from typing import Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the WikiReportRequest model for type checking
from api.wiki_report import WikiReportRequest

def generate_wiki_report(
    repo_url: str,
    repo_type: Optional[str] = None,
    token: Optional[str] = None,
    language: str = "en",
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    excluded_dirs: Optional[list] = None,
    excluded_files: Optional[list] = None,
    included_dirs: Optional[list] = None,
    included_files: Optional[list] = None,
    store_in_rag: bool = True,
    api_url: str = "http://localhost:8000"
) -> Dict[str, Any]:
    """
    Generate a wiki report for a repository using the backend API.
    
    Args:
        repo_url: URL of the repository to generate a wiki for
        repo_type: Type of repository (github, gitlab, bitbucket)
        token: Personal access token for private repositories
        language: Language for content generation (e.g., 'en', 'ja', 'zh', 'es', 'kr', 'vi')
        model_provider: Model provider to use for content generation
        model_name: Model name to use with the provider
        excluded_dirs: Directories to exclude from processing
        excluded_files: File patterns to exclude from processing
        included_dirs: Directories to include exclusively
        included_files: File patterns to include exclusively
        store_in_rag: Whether to store the report in the RAG system
        api_url: URL of the API server
        
    Returns:
        The generated wiki report as a dictionary
    """
    # Create the request payload
    payload = {
        "repo_url": repo_url,
        "repo_type": repo_type,
        "token": token,
        "language": language,
        "model_provider": model_provider,
        "model_name": model_name,
        "excluded_dirs": excluded_dirs or [],
        "excluded_files": excluded_files or [],
        "included_dirs": included_dirs or [],
        "included_files": included_files or [],
        "store_in_rag": store_in_rag
    }
    
    # Send the request to the API
    response = requests.post(
        f"{api_url}/api/wiki/generate",
        json=payload,
        headers={"Content-Type": "application/json"}
    )
    
    # Check if the request was successful
    if response.status_code != 200:
        error_message = f"Error generating wiki report: {response.status_code} - {response.text}"
        print(error_message)
        raise Exception(error_message)
    
    # Return the response data
    return response.json()

def save_wiki_report(report: Dict[str, Any], output_file: str) -> None:
    """
    Save the wiki report to a JSON file.
    
    Args:
        report: The wiki report to save
        output_file: The path to save the report to
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Wiki report saved to {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a wiki report for a repository")
    parser.add_argument("repo_url", help="URL of the repository to generate a wiki for")
    parser.add_argument("--repo-type", help="Type of repository (github, gitlab, bitbucket)")
    parser.add_argument("--token", help="Personal access token for private repositories")
    parser.add_argument("--language", default="en", help="Language for content generation")
    parser.add_argument("--model-provider", help="Model provider to use for content generation")
    parser.add_argument("--model-name", help="Model name to use with the provider")
    parser.add_argument("--excluded-dirs", nargs="*", help="Directories to exclude from processing")
    parser.add_argument("--excluded-files", nargs="*", help="File patterns to exclude from processing")
    parser.add_argument("--included-dirs", nargs="*", help="Directories to include exclusively")
    parser.add_argument("--included-files", nargs="*", help="File patterns to include exclusively")
    parser.add_argument("--no-rag", action="store_true", help="Don't store the report in the RAG system")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of the API server")
    parser.add_argument("--output", help="Path to save the report to")
    
    args = parser.parse_args()
    
    # Generate the wiki report
    report = generate_wiki_report(
        repo_url=args.repo_url,
        repo_type=args.repo_type,
        token=args.token,
        language=args.language,
        model_provider=args.model_provider,
        model_name=args.model_name,
        excluded_dirs=args.excluded_dirs,
        excluded_files=args.excluded_files,
        included_dirs=args.included_dirs,
        included_files=args.included_files,
        store_in_rag=not args.no_rag,
        api_url=args.api_url
    )
    
    # Print the report summary
    print(f"Wiki report generated for {args.repo_url}")
    print(f"Number of pages: {len(report['generated_pages'])}")
    print(f"Stored in RAG: {report['stored_in_rag']}")
    
    # Save the report to a file if requested
    if args.output:
        save_wiki_report(report, args.output)
