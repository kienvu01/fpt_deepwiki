"""
Module defining the structure of wiki pages and sections.
"""

from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class WikiPage:
    """Class representing a wiki page."""
    id: str
    title: str
    content: str
    filePaths: List[str]
    importance: str  # 'high', 'medium', or 'low'
    relatedPages: List[str]
    parentId: Optional[str] = None
    isSection: bool = False
    children: List[str] = field(default_factory=list)

@dataclass
class WikiStructure:
    """Class representing the overall wiki structure."""
    id: str
    title: str
    description: str
    pages: List[WikiPage]
