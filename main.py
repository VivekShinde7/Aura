from dotenv import load_dotenv
from core.graph import app
from core.state import SubjectProfile, Document, ExtractedEntity, SummarizedEntity
import json

# Load environment variables from .env file
load_dotenv()

# Define a custom JSON encoder to handle Pydantic models for clean printing
class PydanticEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (Document, ExtractedEntity, SummarizedEntity, SubjectProfile)):
                # Use .model_dump() which is the Pydantic v2 standard
                return obj.model_dump() 
        return json.JSONEncoder.default(self, obj)

def run_investigation():
    """
    Main function to define a subject and run the full investigation workflow.
    This version correctly uses stream() for a single execution run,
    provides real-time logs, and accumulates the final state.
    """
    print("--- Starting New Investigation ---")

    # Define the subject profile for the investigation
    subject_profile = SubjectProfile(
        name="Elon Musk",
        entity_type="person",
        dob="1971-06-28",
        locations=["Texas, USA", "California, USA"],
        keywords=["Tesla", "SpaceX", "lawsuit", "neuralink"]
    )

    # This is the initial state that will be passed to the graph.
    # It will be updated at each step of the workflow.
    thread_state = {
        "subject": subject_profile,
        "documents": [],
        "extracted_entities": [],
        "summarized_entities": [],
    }

    print("Invoking the investigation workflow...")
    
    # --- Single, Efficient Execution with State Accumulation ---
    for output in app.stream(thread_state):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(json.dumps(value, indent=2, cls=PydanticEncoder))
            print("---")
            
            # This is the crucial step: Merge the partial update (value) 
            # from the current node into our main state dictionary.
            thread_state.update(value)
    
    print("--- Investigation Workflow Finished ---")
    
    # 'thread_state' now holds the complete and final state of the graph.
    # We can now confidently access any part of it.
    print("--- FINAL SUMMARIZED ENTITIES ---")
    if thread_state.get("summarized_entities"):
        print(json.dumps(thread_state["summarized_entities"], indent=2, cls=PydanticEncoder))
    else:
        print("Workflow did not produce a final summarized entity list.")
    
    # For full visibility, let's also print the entire final state
    print("--- COMPLETE FINAL STATE FOR DEBUGGING ---")
    print(json.dumps(thread_state, indent=2, cls=PydanticEncoder))


if __name__ == "__main__":
    run_investigation()