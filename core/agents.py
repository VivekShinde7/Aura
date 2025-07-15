import os
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from .state import InvestigationState, Document

class SearchResults(BaseModel):
    """Structured data model for search results."""
    results: List[Document] = Field(description="A list of relevant documents found from the web search.")

# ... (keep all existing imports and Pydantic models)

# --- The FINAL Production-Ready Web Search Agent ---
def web_search_agent(state: InvestigationState) -> InvestigationState:
    """
    A production-ready agent that uses a two-phase search strategy.
    1. Broad Search: To establish a digital footprint.
    2. Deep Search: If warranted, performs targeted searches for specific signals.
    """
    print("---AGENT: Production Web Search Agent (v3 - Two-Phase)---")
    subject = state['subject']
    all_found_documents = []
    unique_urls = set()

    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
    structured_llm = llm.with_structured_output(SearchResults)

    # --- PHASE 1: BROAD SEARCH ---
    # Goal: Find any core information or official profiles about the subject.
    print(f"--- PHASE 1: BROAD SEARCH for '{subject.name}' ---")
    
    # A simple, powerful query using the core identifiers.
    broad_query_parts = [f'"{subject.name}"']
    if subject.dob:
        broad_query_parts.append(f'born {subject.dob}')
    if subject.locations:
        # For the broad search, let's just use the first, most likely location.
        broad_query_parts.append(f'"{subject.locations[0]}"')
        
    broad_query = " ".join(broad_query_parts)
    print(f"  - Executing broad query: {broad_query}")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a researcher. Your goal is to find the most authoritative "
                   "web pages (like official profiles, LinkedIn, company pages, or top-tier news mentions) "
                   "for the given subject. This is a preliminary search to establish their digital footprint."),
        ("human", "Please perform a web search for: {search_query}")
    ])
    final_prompt = prompt_template.format_prompt(search_query=broad_query)

    try:
        response = structured_llm.invoke(final_prompt)
        if response.results:
            for doc in response.results:
                if doc.url not in unique_urls:
                    all_found_documents.append(doc)
                    unique_urls.add(doc.url)
            print(f"    - Broad search found {len(response.results)} initial documents.")
        else:
            print("    - Broad search found no initial documents. Subject may have a low digital footprint.")
    except Exception as e:
        print(f"    - Error during broad search: {e}")

    # --- PHASE 2: DEEP, TARGETED SEARCH ---
    # Goal: Only if the subject exists, probe for specific risk signals.
    # We proceed if the broad search found anything, or if it's a high-profile subject.
    if len(all_found_documents) > 0 or subject.name.lower() in ["elon musk", "tesla"]: # Add other high-profile names if needed
        print(f"--- PHASE 2: DEEP SEARCH for '{subject.name}' ---")
        
        search_angles = {
            "legal_issues": f'"{subject.name}" lawsuit OR legal OR court OR settlement OR investigation',
            "negative_press": f'"{subject.name}" controversy OR scandal OR criticism OR complaint',
            "financial_distress": f'"{subject.name}" bankruptcy OR debt OR financial issues',
        }

        for angle, query in search_angles.items():
            print(f"  - Executing deep search on angle '{angle}'")
            # Same prompt logic as before
            deep_prompt = prompt_template.format_prompt(search_query=query)
            try:
                response = structured_llm.invoke(deep_prompt)
                if response.results:
                    for doc in response.results:
                        if doc.url not in unique_urls:
                            all_found_documents.append(doc)
                            unique_urls.add(doc.url)
                    print(f"    - Deep search found {len(response.results)} new documents.")
            except Exception as e:
                print(f"    - Error during deep search for angle {angle}: {e}")
                continue
    else:
        print("--- SKIPPING PHASE 2: DEEP SEARCH (insufficient initial findings) ---")

    print(f"Total unique documents found across all phases: {len(all_found_documents)}")
    return {"documents": all_found_documents}