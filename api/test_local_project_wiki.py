"""
Test module for local project wiki generation.
"""

import os
import pytest
from api.local_project_wiki import LocalProjectWiki
from api.wiki_structure import WikiStructure, WikiPage

def test_local_project_wiki_generation(tmp_path):
    """Test generating wiki from a local project."""
    # Create a test project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create some test files and directories
    src_dir = project_dir / "src"
    src_dir.mkdir()
    
    # Create a Python file
    py_file = src_dir / "main.py"
    py_file.write_text("""
def main():
    print("Hello, World!")

if __name__ == "__main__":
    main()
""")
    
    # Create a nested directory with files
    utils_dir = src_dir / "utils"
    utils_dir.mkdir()
    
    utils_file = utils_dir / "helpers.py"
    utils_file.write_text("""
def helper_function():
    return "I'm helping!"
""")
    
    # Create a README
    readme = project_dir / "README.md"
    readme.write_text("""
# Test Project

This is a test project for wiki generation.
""")
    
    # Initialize wiki generator
    wiki_gen = LocalProjectWiki(
        project_path=str(project_dir),
        model_provider="google"
    )
    
    # Generate wiki
    wiki_structure, generated_pages = wiki_gen.generate_wiki()
    
    # Test wiki structure
    assert isinstance(wiki_structure, WikiStructure)
    assert wiki_structure.title == "test_project"
    assert len(wiki_structure.pages) > 0
    
    # Test overview page
    overview_page = next((p for p in wiki_structure.pages if p.id == "overview"), None)
    assert overview_page is not None
    assert overview_page.title == "Project Overview"
    assert overview_page.importance == "high"
    assert overview_page.isSection is True
    
    # Test architecture page
    arch_page = next((p for p in wiki_structure.pages if p.id == "architecture"), None)
    assert arch_page is not None
    assert arch_page.title == "Architecture"
    assert arch_page.importance == "high"
    assert arch_page.isSection is True
    assert "overview" in arch_page.relatedPages
    
    # Test src directory page
    src_page = next((p for p in wiki_structure.pages if p.id == "src"), None)
    assert src_page is not None
    assert src_page.title == "src"
    assert src_page.importance == "medium"
    assert src_page.isSection is True
    assert "architecture" in src_page.relatedPages
    assert len(src_page.children) > 0  # Should have utils as child
    
    # Test utils directory page
    utils_page = next((p for p in wiki_structure.pages if p.id == "src_utils"), None)
    assert utils_page is not None
    assert utils_page.title == "utils"
    assert utils_page.importance == "medium"
    assert utils_page.isSection is False
    assert utils_page.parentId == "src"
    assert "architecture" in utils_page.relatedPages
    
    # Test generated pages dictionary
    assert isinstance(generated_pages, dict)
    assert "overview" in generated_pages
    assert "architecture" in generated_pages
    assert "src" in generated_pages
    assert "src_utils" in generated_pages
    
    # Test page content
    assert "test_project" in generated_pages["overview"].content.lower()
    assert "Project Structure" in generated_pages["architecture"].content
    assert "main.py" in generated_pages["src"].content
    assert "helpers.py" in generated_pages["src_utils"].content

def test_local_project_wiki_with_exclusions(tmp_path):
    """Test wiki generation with excluded directories and files."""
    # Create a test project structure
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create directories
    src_dir = project_dir / "src"
    src_dir.mkdir()
    test_dir = project_dir / "tests"
    test_dir.mkdir()
    
    # Create files
    src_file = src_dir / "main.py"
    src_file.write_text("print('main')")
    
    test_file = test_dir / "test_main.py"
    test_file.write_text("print('test')")
    
    # Initialize wiki generator with exclusions
    wiki_gen = LocalProjectWiki(
        project_path=str(project_dir),
        excluded_dirs=["tests"],
        excluded_files=["*.pyc"]
    )
    
    # Generate wiki
    wiki_structure, generated_pages = wiki_gen.generate_wiki()
    
    # Test that tests directory is excluded
    test_page = next((p for p in wiki_structure.pages if p.id == "tests"), None)
    assert test_page is None
    
    # Test that src directory is included
    src_page = next((p for p in wiki_structure.pages if p.id == "src"), None)
    assert src_page is not None
    assert "main.py" in src_page.content
    assert "test_main.py" not in src_page.content

def test_local_project_wiki_rag_integration(tmp_path):
    """Test RAG system integration with wiki generation."""
    # Create a test project
    project_dir = tmp_path / "test_project"
    project_dir.mkdir()
    
    # Create a file with meaningful content
    src_dir = project_dir / "src"
    src_dir.mkdir()
    
    main_file = src_dir / "main.py"
    main_file.write_text("""
# This is a sample Python script
# It demonstrates basic functionality

def calculate_sum(a: int, b: int) -> int:
    '''Calculate the sum of two integers.'''
    return a + b

def main():
    result = calculate_sum(5, 3)
    print(f"The sum is: {result}")

if __name__ == "__main__":
    main()
""")
    
    # Initialize wiki generator
    wiki_gen = LocalProjectWiki(
        project_path=str(project_dir),
        model_provider="google"
    )
    
    # Generate wiki
    wiki_structure, generated_pages = wiki_gen.generate_wiki()
    
    # Test that RAG system contains the project information
    # This can be verified by checking the content of the overview page
    overview_page = generated_pages["overview"]
    assert "Python" in overview_page.content
    assert "calculate_sum" in overview_page.content
    
    # Test that architecture page includes file structure
    arch_page = generated_pages["architecture"]
    assert "src" in arch_page.content
    assert "main.py" in arch_page.content
    
    # Test that src directory page includes file details
    src_page = generated_pages["src"]
    assert "calculate_sum" in src_page.content
    assert "Lines of Code" in src_page.content
