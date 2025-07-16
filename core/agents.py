from collections import defaultdict
import os
import time
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from .state import InvestigationState, Document, ExtractedEntity, SummarizedEntity
from constants import MAX_DOCS_TO_PROCESS, API_CALL_DELAY_SECONDS



class SearchResults(BaseModel):
    """Structured data model for search results."""
    results: List[Document] = Field(description="A list of relevant documents found from the web search.")

class Entities(BaseModel):
    """A list of entities found in a body of text."""
    entities: List[ExtractedEntity]

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

            # --- ADD A DELAY BEFORE EACH DEEP SEARCH CALL ---
            print(f"Waiting {API_CALL_DELAY_SECONDS} seconds before next API call...")
            time.sleep(API_CALL_DELAY_SECONDS)

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

def entity_extraction_agent(state: InvestigationState) -> InvestigationState:
    """
    Extracts entities (People, Companies, Locations) from the documents' raw content.
    """
    print("---AGENT: Entity Extraction Agent---")
    
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
    # Important: We tell the model to structure its output according to our Pydantic class
    structured_llm = llm.with_structured_output(Entities)
    
    all_extracted_entities = []
    documents_to_process = state['documents']

    print(f"Found {len(documents_to_process)} documents. Processing a maximum of {MAX_DOCS_TO_PROCESS} for this test run.")
    # We slice the list to only process the first N documents
    limited_documents = documents_to_process[:MAX_DOCS_TO_PROCESS]

    # We process each document individually to keep the context small and focused
    for i, doc in enumerate(limited_documents):
        print(f"  - ({i+1}/{len(limited_documents)}) Extracting from: {doc.title[:50]}...")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert data analyst. Your task is to extract all named entities "
                       "(specifically People, Companies, and Locations) from the provided text. For each entity, provide its name, "
                       "its type, and the source document URL it was found in. Do not extract the main subject of the investigation, only other entities mentioned."),
            ("human", f"Please extract entities from the following text. The source URL is {doc.url}. "
                      f"The main subject is '{state['subject'].name}', do not include them in the output.\n\nText:\n---\n{doc.raw_content}")
        ])

        final_prompt = prompt_template.format_prompt()
        
        try:
            response = structured_llm.invoke(final_prompt)
            # Add the found entities to our master list
            if response.entities:
                all_extracted_entities.extend(response.entities)
                print(f"- Found {len(response.entities)} entities in this document.")
        except Exception as e:
            print(f"- Error extracting entities from {doc.url}: {e}")
            continue

    print(f"Extracted a total of {len(all_extracted_entities)} entities across all documents.")
    
    # Return the update to the state
    return {"extracted_entities": all_extracted_entities}

def summarize_entities_agent(state: InvestigationState) -> InvestigationState:
    """
    Deduplicates and aggregates the extracted entities.
    """
    print("---AGENT: Entity Summarization Agent---")
    
    raw_entities = state['extracted_entities']
    
    # Use a defaultdict to easily group entities by name and type
    # The key will be a tuple: (name, type)
    aggregated_data = defaultdict(lambda: {"count": 0, "urls": set()})
    
    for entity in raw_entities:
        key = (entity.name.strip().lower(), entity.type.lower())
        aggregated_data[key]["count"] += 1
        aggregated_data[key]["urls"].add(entity.source_document_url)

    # Convert the aggregated data into our Pydantic model
    summarized_list = []
    for (name, entity_type), data in aggregated_data.items():
        summarized_list.append(
            SummarizedEntity(
                name=name.title(), # Capitalize for clean display
                type=entity_type.title(),
                count=data["count"],
                source_urls=list(data["urls"])
            )
        )
        
    # Sort the list by count, so the most mentioned entities are at the top
    summarized_list.sort(key=lambda x: x.count, reverse=True)
    
    print(f"Summarized {len(raw_entities)} raw entities into {len(summarized_list)} unique entities.")

    state['summarized_entities'] = summarized_list
    return state