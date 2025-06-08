# Local Project Analysis API

This API provides functionality for analyzing local project directories, generating comprehensive reports, and storing them in the RAG (Retrieval-Augmented Generation) system for later querying.

## Features

- Analyze local project directories to extract statistics and insights
- Generate comprehensive reports with project structure, file statistics, and language breakdown
- Generate AI-powered summaries of projects
- Store reports in the RAG system for later querying
- Command-line interface for easy usage
- RESTful API for integration with other applications

## Installation

The Local Project Analysis API is part of the DeepWiki project. To install it, follow these steps:

1. Clone the DeepWiki repository:
   ```bash
   git clone https://github.com/AsyncFuncAI/deepwiki-open.git
   cd deepwiki-open
   ```

2. Install the required dependencies:
   ```bash
   pip install -r api/requirements.txt
   ```

3. Start the API server:
   ```bash
   cd api
   python main.py
   ```

## Command-Line Usage

The API includes a command-line interface for analyzing local projects:

```bash
python report_cli.py /path/to/project [options]
```

### Options

- `--output`, `-o`: Path to save the report JSON file (default: `report_<project_name>_<timestamp>.json`)
- `--exclude-dirs`, `-d`: Directories to exclude from analysis (e.g., `node_modules`, `venv`)
- `--exclude-files`, `-f`: File patterns to exclude from analysis (e.g., `*.min.js`, `*.pyc`)
- `--model-provider`, `-p`: Model provider to use for summary generation (default: `google`)
- `--model-name`, `-m`: Model name to use with the provider
- `--no-summary`: Skip generating a project summary
- `--no-rag`: Skip storing the report in the RAG system
- `--verbose`, `-v`: Enable verbose logging

### Example

```bash
python report_cli.py /path/to/project -o report.json -d node_modules .git -f *.min.js *.map
```

## API Usage

The API provides several endpoints for analyzing local projects:

### Analyze a Local Project

```http
POST /api/local_project/analyze
```

#### Request Body

```json
{
  "project_path": "/path/to/project",
  "excluded_dirs": ["node_modules", ".git"],
  "excluded_files": ["*.min.js", "*.map"],
  "model_provider": "google",
  "model_name": null,
  "store_in_rag": true
}
```

#### Response

```json
{
  "project_name": "project-name",
  "analysis_timestamp": "2025-06-05T20:00:00",
  "total_files": 100,
  "total_size_bytes": 1000000,
  "total_lines_of_code": 5000,
  "languages": {
    "Python": 50,
    "JavaScript": 30,
    "HTML": 10,
    "CSS": 10
  },
  "file_stats": [...],
  "directory_stats": [...],
  "top_files_by_size": [...],
  "top_files_by_loc": [...],
  "readme_content": "...",
  "project_structure": "...",
  "summary": "...",
  "stored_in_rag": true
}
```

### List Local Project Reports

```http
GET /api/local_project/reports
```

#### Response

```json
[
  {
    "id": "local_project1_1717500000",
    "project_name": "project1",
    "analysis_timestamp": "2025-06-05T20:00:00",
    "total_files": 100,
    "total_lines_of_code": 5000,
    "total_size_bytes": 1000000,
    "languages": {
      "Python": 50,
      "JavaScript": 30
    }
  },
  {
    "id": "local_project2_1717400000",
    "project_name": "project2",
    "analysis_timestamp": "2025-06-04T20:00:00",
    "total_files": 200,
    "total_lines_of_code": 10000,
    "total_size_bytes": 2000000,
    "languages": {
      "Java": 100,
      "XML": 50
    }
  }
]
```

### Get a Specific Local Project Report

```http
GET /api/local_project/report/{id}
```

#### Response

```json
{
  "id": "local_project1_1717500000",
  "project_name": "project1",
  "analysis_timestamp": "2025-06-05T20:00:00",
  "total_files": 100,
  "total_size_bytes": 1000000,
  "total_lines_of_code": 5000,
  "languages": {
    "Python": 50,
    "JavaScript": 30,
    "HTML": 10,
    "CSS": 10
  },
  "file_stats": [...],
  "directory_stats": [...],
  "top_files_by_size": [...],
  "top_files_by_loc": [...],
  "readme_content": "...",
  "project_structure": "...",
  "summary": "..."
}
```

## Example Script

The `analyze_local_project.py` script demonstrates how to use the API from Python:

```bash
python analyze_local_project.py /path/to/project [options]
```

### Options

- `--api-url`: URL of the API server (default: `http://localhost:8000`)
- `--exclude-dirs`, `-d`: Directories to exclude from analysis
- `--exclude-files`, `-f`: File patterns to exclude from analysis
- `--model-provider`, `-p`: Model provider to use for summary generation (default: `google`)
- `--model-name`, `-m`: Model name to use with the provider
- `--no-rag`: Skip storing the report in the RAG system
- `--output`, `-o`: Path to save the report JSON file (default: `report_<project_name>.json`)

### Example

```bash
python analyze_local_project.py /path/to/project --api-url http://localhost:8000 -o report.json
```

## Report Structure

The generated report includes the following information:

- **Project Name**: The name of the project (derived from the directory name)
- **Analysis Timestamp**: The time when the analysis was performed
- **Total Files**: The total number of files in the project
- **Total Size**: The total size of the project in bytes
- **Total Lines of Code**: The total number of lines of code in the project
- **Languages**: A breakdown of programming languages used in the project
- **File Stats**: Statistics for each file in the project
- **Directory Stats**: Statistics for each directory in the project
- **Top Files by Size**: The largest files in the project
- **Top Files by Lines of Code**: The files with the most lines of code
- **README Content**: The content of the project's README file
- **Project Structure**: A tree representation of the project's file structure
- **Summary**: An AI-generated summary of the project

## License

This project is licensed under the MIT License - see the LICENSE file for details.
