#!/usr/bin/env python3
"""
Command-line interface for generating project reports.

This script provides a command-line interface for analyzing local directories,
generating comprehensive project reports, and storing them in the RAG system.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Optional
from datetime import datetime

from api.local_report import (
    analyze_project,
    generate_project_summary,
    store_report_in_rag,
    analyze_and_store_project
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze a local directory and generate a project report.'
    )
    
    parser.add_argument(
        'project_path',
        type=str,
        help='Path to the project directory to analyze'
    )
    
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Path to save the report JSON file (default: report_<project_name>_<timestamp>.json)'
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
        '--no-summary',
        action='store_true',
        help='Skip generating a project summary'
    )
    
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Skip storing the report in the RAG system'
    )
    
    parser.add_argument(
        '--verbose',
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate project path
    project_path = os.path.abspath(args.project_path)
    if not os.path.isdir(project_path):
        logger.error(f"Project path does not exist or is not a directory: {project_path}")
        sys.exit(1)
    
    try:
        # Analyze the project
        logger.info(f"Analyzing project at {project_path}")
        report = analyze_project(
            project_path,
            excluded_dirs=args.exclude_dirs,
            excluded_files=args.exclude_files
        )
        
        # Generate summary if requested
        if not args.no_summary:
            logger.info("Generating project summary")
            summary = generate_project_summary(
                report,
                model_provider=args.model_provider,
                model_name=args.model_name
            )
            report.summary = summary
        
        # Store in RAG if requested
        if not args.no_rag:
            logger.info("Storing project report in RAG")
            success = store_report_in_rag(report, project_path)
            if success:
                logger.info("Report successfully stored in RAG")
            else:
                logger.warning("Failed to store report in RAG")
        
        # Save report to file
        if args.output:
            output_path = args.output
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"report_{report.project_name}_{timestamp}.json"
        
        report.save_to_file(output_path)
        logger.info(f"Report saved to {output_path}")
        
        # Print summary statistics
        print("\n=== Project Report Summary ===")
        print(f"Project Name: {report.project_name}")
        print(f"Total Files: {report.total_files}")
        print(f"Total Lines of Code: {report.total_lines_of_code}")
        print(f"Total Size: {report.total_size_bytes / 1024 / 1024:.2f} MB")
        
        print("\nLanguages:")
        for lang, count in sorted(report.languages.items(), key=lambda x: x[1], reverse=True):
            print(f"  {lang}: {count} files")
        
        print("\nTop Files by Size:")
        for i, path in enumerate(report.top_files_by_size[:5], 1):
            size = next((f.size_bytes for f in report.file_stats if f.path == path), 0)
            print(f"  {i}. {path} ({size / 1024:.2f} KB)")
        
        print("\nTop Files by Lines of Code:")
        for i, path in enumerate(report.top_files_by_loc[:5], 1):
            loc = next((f.lines_of_code for f in report.file_stats if f.path == path), 0)
            print(f"  {i}. {path} ({loc} lines)")
        
        print(f"\nFull report saved to: {output_path}")
        if not args.no_rag:
            print("Report stored in RAG system for querying")
        
        return 0
    
    except Exception as e:
        logger.error(f"Error analyzing project: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())
