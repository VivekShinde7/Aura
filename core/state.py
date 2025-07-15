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

class InvestigationState(TypedDict):
    """
    Represents the shared state of our investigation workflow.
    This acts as our "MCP Packet".
    """
    subject: SubjectProfile
    documents: List[Document]
    # We will add more fields here later, like 'extracted_entities', etc.