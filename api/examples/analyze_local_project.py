#!/usr/bin/env python3
"""
Example script for using the local project analysis API.

This script demonstrates how to use the local project analysis API to analyze
a local directory, generate a report, and store it in the RAG system.
"""

import os
import sys
import argparse
import json
import requests
from pprint import pprint

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Example script for using the local project analysis API.'
    )
    
    parser.add_argument(
        'project_path',
        type=str,
        help='Path to the project directory to analyze'
    )
    
    parser.add_argument(
        '--api-url',
        type=str,
        default='http://localhost:8000',
        help='URL of the API server (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--exclude-dirs',
        '-d',
        type=str,
        nargs='+',
        help='Directories to exclude from analysis'
    )
    
    parser.add_argument(
        '--exclude-files',
        '-f',
        type=str,
        nargs='+',
        help='File patterns to exclude from analysis'
    )
    
    parser.add_argument(
        '--model-provider',
        '-p',
        type=str,
        choices=['google', 'openai', 'openrouter', 'ollama'],
        default='google',
        help='Model provider to use for summary generation (default: google)'
    )
    
    parser.add_argument(
        '--model-name',
        '-m',
        type=str,
        help='Model name to use with the provider'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Skip storing the report in the RAG system'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Path to save the report JSON file (default: report_<project_name>.json)'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Validate project path
    project_path = os.path.abspath(args.project_path)
    if not os.path.isdir(project_path):
        print(f"Error: Project path does not exist or is not a directory: {project_path}")
        return 1
    
    # Prepare the request payload
    payload = {
        "project_path": project_path,
        "excluded_dirs": args.exclude_dirs,
        "excluded_files": args.exclude_files,
        "model_provider": args.model_provider,
        "model_name": args.model_name,
        "store_in_rag": not args.no_rag
    }
    
    # Make the API request
    api_url = f"{args.api_url}/api/local_project/analyze"
    print(f"Sending request to {api_url}")
    print(f"Analyzing project at {project_path}")
    
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        
        # Parse the response
        report = response.json()
        
        # Print summary statistics
        print("\n=== Project Report Summary ===")
        print(f"Project Name: {report['project_name']}")
        print(f"Total Files: {report['total_files']}")
        print(f"Total Lines of Code: {report['total_lines_of_code']}")
        print(f"Total Size: {report['total_size_bytes'] / 1024 / 1024:.2f} MB")
        
        print("\nLanguages:")
        for lang, count in sorted(report['languages'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count} files")
        
        print("\nTop Files by Size:")
        for i, path in enumerate(report['top_files_by_size'][:5], 1):
            size = next((f['size_bytes'] for f in report['file_stats'] if f['path'] == path), 0)
            print(f"  {i}. {path} ({size / 1024:.2f} KB)")
        
        print("\nTop Files by Lines of Code:")
        for i, path in enumerate(report['top_files_by_loc'][:5], 1):
            loc = next((f['lines_of_code'] for f in report['file_stats'] if f['path'] == path), 0)
            print(f"  {i}. {path} ({loc} lines)")
        
        # Save the report to a file if requested
        if args.output:
            output_path = args.output
        else:
            output_path = f"report_{report['project_name']}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nFull report saved to: {output_path}")
        
        # Print the summary if available
        if 'summary' in report and report['summary']:
            print("\n=== Project Summary ===")
            print(report['summary'])
        
        return 0
    
    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status code: {e.response.status_code}")
            try:
                error_data = e.response.json()
                print(f"Error details: {error_data}")
            except:
                print(f"Response text: {e.response.text}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
