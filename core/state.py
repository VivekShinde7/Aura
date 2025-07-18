from typing import TypedDict, List, Dict, Optional
from pydantic import BaseModel

class SubjectProfile(BaseModel):
    name: str
    entity_type: str  # 'person' or 'company'
    dob: Optional[str] = None
    locations: List[str] = []
    keywords: List[str] = []

class Document(BaseModel):
    url: str
    title: str
    raw_content: str
    source: str

class ExtractedEntity(BaseModel):
    """Represents a single entity extracted from a document."""
    name: str
    type: str # e.g., 'Person', 'Company', 'Location'
    source_document_url: str # To trace back where it was found

class SummarizedEntity(BaseModel):
    """Represents a unique entity and all its sources."""
    name: str
    type: str
    count: int # How many times it was found
    source_urls: List[str] # List of all unique URLs where it was mentioned


class InvestigationState(TypedDict):
    """
    Represents the shared state of our investigation workflow.
    This acts as our "MCP Packet".
    """
    subject: SubjectProfile
    documents: List[Document]
    extracted_entities: List[ExtractedEntity]
    summarized_entities: List[SummarizedEntity] 