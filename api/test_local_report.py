#!/usr/bin/env python3
"""
Test cases for the local project analysis API.

This module contains test cases for the local project analysis functionality,
including the API endpoints and the core analysis functions.
"""

import os
import sys
import unittest
import tempfile
import shutil
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from api.api import app
from api.local_report import (
    FileStats,
    DirectoryStats,
    ProjectReport,
    detect_language,
    count_lines_of_code,
    is_test_file,
    is_config_file,
    is_documentation_file,
    analyze_project,
    generate_project_summary,
    store_report_in_rag,
    analyze_and_store_project
)

# Create a test client
client = TestClient(app)

class TestLocalReportUtils(unittest.TestCase):
    """Test cases for utility functions in local_report.py."""
    
    def test_detect_language(self):
        """Test language detection based on file extension."""
        self.assertEqual(detect_language("test.py"), "Python")
        self.assertEqual(detect_language("test.js"), "JavaScript")
        self.assertEqual(detect_language("test.jsx"), "JavaScript (React)")
        self.assertEqual(detect_language("test.ts"), "TypeScript")
        self.assertEqual(detect_language("test.tsx"), "TypeScript (React)")
        self.assertEqual(detect_language("test.html"), "HTML")
        self.assertEqual(detect_language("test.css"), "CSS")
        self.assertEqual(detect_language("test.unknown"), "Unknown")
        self.assertEqual(detect_language("Dockerfile"), "Dockerfile")
    
    def test_is_test_file(self):
        """Test detection of test files."""
        self.assertTrue(is_test_file("test_file.py"))
        self.assertTrue(is_test_file("file_test.py"))
        self.assertTrue(is_test_file("file.test.js"))
        self.assertTrue(is_test_file("file.spec.js"))
        self.assertTrue(is_test_file("FileTest.java"))
        self.assertTrue(is_test_file("test/file.py"))
        self.assertTrue(is_test_file("tests/file.py"))
        self.assertFalse(is_test_file("file.py"))
        self.assertFalse(is_test_file("testing.py"))
    
    def test_is_config_file(self):
        """Test detection of configuration files."""
        self.assertTrue(is_config_file("config.json"))
        self.assertTrue(is_config_file("config.yaml"))
        self.assertTrue(is_config_file("config.yml"))
        self.assertTrue(is_config_file("config.toml"))
        self.assertTrue(is_config_file("config.ini"))
        self.assertTrue(is_config_file(".env"))
        self.assertTrue(is_config_file("Dockerfile"))
        self.assertTrue(is_config_file("docker-compose.yml"))
        self.assertTrue(is_config_file("package.json"))
        self.assertTrue(is_config_file(".gitignore"))
        self.assertFalse(is_config_file("file.py"))
        self.assertFalse(is_config_file("file.js"))
    
    def test_is_documentation_file(self):
        """Test detection of documentation files."""
        self.assertTrue(is_documentation_file("README.md"))
        self.assertTrue(is_documentation_file("readme.txt"))
        self.assertTrue(is_documentation_file("LICENSE"))
        self.assertTrue(is_documentation_file("CONTRIBUTING.md"))
        self.assertTrue(is_documentation_file("CHANGELOG.md"))
        self.assertTrue(is_documentation_file("docs/file.md"))
        self.assertTrue(is_documentation_file("documentation/file.txt"))
        self.assertFalse(is_documentation_file("file.py"))
        self.assertFalse(is_documentation_file("file.js"))

class TestLocalReportCore(unittest.TestCase):
    """Test cases for core functionality in local_report.py."""
    
    def setUp(self):
        """Set up a temporary project directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple project structure
        os.makedirs(os.path.join(self.temp_dir, "src"))
        os.makedirs(os.path.join(self.temp_dir, "tests"))
        os.makedirs(os.path.join(self.temp_dir, "docs"))
        
        # Create some files
        with open(os.path.join(self.temp_dir, "README.md"), "w") as f:
            f.write("# Test Project\n\nThis is a test project for unit tests.")
        
        with open(os.path.join(self.temp_dir, "src", "main.py"), "w") as f:
            f.write("def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()")
        
        with open(os.path.join(self.temp_dir, "src", "utils.py"), "w") as f:
            f.write("def add(a, b):\n    return a + b\n\ndef subtract(a, b):\n    return a - b")
        
        with open(os.path.join(self.temp_dir, "tests", "test_utils.py"), "w") as f:
            f.write("import unittest\nfrom src.utils import add, subtract\n\nclass TestUtils(unittest.TestCase):\n    def test_add(self):\n        self.assertEqual(add(1, 2), 3)\n\n    def test_subtract(self):\n        self.assertEqual(subtract(3, 1), 2)")
        
        with open(os.path.join(self.temp_dir, "docs", "usage.md"), "w") as f:
            f.write("# Usage\n\nThis document describes how to use the project.")
        
        with open(os.path.join(self.temp_dir, "config.json"), "w") as f:
            f.write('{"name": "test-project", "version": "1.0.0"}')
    
    def tearDown(self):
        """Clean up the temporary project directory."""
        shutil.rmtree(self.temp_dir)
    
    def test_count_lines_of_code(self):
        """Test counting lines of code in a file."""
        main_py_path = os.path.join(self.temp_dir, "src", "main.py")
        self.assertEqual(count_lines_of_code(main_py_path), 5)
        
        utils_py_path = os.path.join(self.temp_dir, "src", "utils.py")
        self.assertEqual(count_lines_of_code(utils_py_path), 5)
    
    def test_analyze_project(self):
        """Test analyzing a project directory."""
        report = analyze_project(self.temp_dir)
        
        # Check basic report properties
        self.assertEqual(report.project_name, os.path.basename(self.temp_dir))
        self.assertEqual(report.total_files, 3)
        self.assertGreater(report.total_size_bytes, 0)
        self.assertGreater(report.total_lines_of_code, 0)
        
        # Check languages
        self.assertIn("Python", report.languages)
        self.assertIn("Markdown", report.languages)
        
        # Check file stats
        self.assertEqual(len(report.file_stats), 3)
        
        # Check directory stats
        self.assertEqual(len(report.directory_stats), 4)  # root, src, tests, docs
        
        # Check README content
        self.assertIn("# Test Project", report.readme_content)
        
        # Check project structure
        self.assertIn("src/", report.project_structure)
        self.assertIn("tests/", report.project_structure)
        self.assertIn("docs/", report.project_structure)
    
    @patch("api.local_report.RAG")
    def test_generate_project_summary(self, mock_rag):
        """Test generating a project summary."""
        # Create a mock RAG instance
        mock_rag_instance = MagicMock()
        mock_rag.return_value = mock_rag_instance
        
        # Configure the mock to return a summary
        mock_response = MagicMock()
        mock_response.answer = "# Project Summary\n\nThis is a test project."
        mock_rag_instance.generator.return_value = mock_response
        
        # Create a simple report
        report = ProjectReport(
            project_name="test-project",
            analysis_timestamp="2025-06-05T20:00:00",
            total_files=6,
            total_size_bytes=1000,
            total_lines_of_code=100,
            languages={"Python": 3, "Markdown": 2, "JSON": 1},
            top_files_by_size=["src/main.py", "src/utils.py"],
            top_files_by_loc=["tests/test_utils.py", "src/main.py"],
            readme_content="# Test Project\n\nThis is a test project for unit tests."
        )
        
        # Generate a summary
        summary = generate_project_summary(report)
        
        # Check that the summary was generated
        self.assertEqual(summary, "# Project Summary\n\nThis is a test project.")
        
        # Check that the RAG instance was created with the correct parameters
        mock_rag.assert_called_once_with(provider="google", model=None)
        
        # Check that the generator was called with the correct prompt
        mock_rag_instance.generator.assert_called_once()
        prompt = mock_rag_instance.generator.call_args[1]["input_str"]
        self.assertIn("test-project", prompt)
        self.assertIn("6", prompt)
        self.assertIn("100", prompt)

class TestLocalReportAPI(unittest.TestCase):
    """Test cases for the local project analysis API endpoints."""
    
    def setUp(self):
        """Set up a temporary project directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a simple project structure
        os.makedirs(os.path.join(self.temp_dir, "src"))
        
        # Create some files
        with open(os.path.join(self.temp_dir, "README.md"), "w") as f:
            f.write("# Test Project\n\nThis is a test project for API tests.")
        
        with open(os.path.join(self.temp_dir, "src", "main.py"), "w") as f:
            f.write("def main():\n    print('Hello, world!')\n\nif __name__ == '__main__':\n    main()")
    
    def tearDown(self):
        """Clean up the temporary project directory."""
        shutil.rmtree(self.temp_dir)
    
    @patch("api.api.analyze_and_store_project")
    def test_analyze_local_project(self, mock_analyze):
        """Test the analyze_local_project endpoint."""
        # Configure the mock to return a report and success status
        mock_report = ProjectReport(
            project_name="test-project",
            analysis_timestamp="2025-06-05T20:00:00",
            total_files=2,
            total_size_bytes=500,
            total_lines_of_code=50,
            languages={"Python": 1, "Markdown": 1}
        )
        mock_analyze.return_value = (mock_report, True)
        
        # Make a request to the endpoint
        response = client.post(
            "/api/local_project/analyze",
            json={
                "project_path": self.temp_dir,
                "model_provider": "google",
                "store_in_rag": True
            }
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["project_name"], "test-project")
        self.assertEqual(data["total_files"], 2)
        self.assertEqual(data["stored_in_rag"], True)
        
        # Check that the analyze_and_store_project function was called with the correct parameters
        mock_analyze.assert_called_once_with(
            project_path=self.temp_dir,
            model_provider="google",
            model_name=None,
            excluded_dirs=None,
            excluded_files=None
        )
    
    @patch("api.api.glob.glob")
    def test_list_local_project_reports(self, mock_glob):
        """Test the list_local_project_reports endpoint."""
        # Configure the mock to return some database paths
        mock_glob.return_value = [
            "/home/user/.adalflow/databases/local_project1_1717500000.pkl",
            "/home/user/.adalflow/databases/local_project2_1717400000.pkl"
        ]
        
        # Make a request to the endpoint
        response = client.get("/api/local_project/reports")
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["project_name"], "project1")
        self.assertEqual(data[1]["project_name"], "project2")
    
    @patch("api.api.os.path.exists")
    def test_get_local_project_report(self, mock_exists):
        """Test the get_local_project_report endpoint."""
        # Configure the mock to return True (file exists)
        mock_exists.return_value = True
        
        # Make a request to the endpoint
        response = client.get("/api/local_project/report/local_project1_1717500000")
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["id"], "local_project1_1717500000")
        self.assertIn("message", data)

if __name__ == "__main__":
    unittest.main()
