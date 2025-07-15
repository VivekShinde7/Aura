from dotenv import load_dotenv
from core.graph import app
from core.state import SubjectProfile, Document, ExtractedEntity
import json

# Load environment variables (like MISTRAL_API_KEY)
load_dotenv()

def run_investigation():
    """
    Main function to define a subject and run the investigation workflow.
    """
    print("--- Starting New Investigation ---")

    # 1. Define the subject profile (this will come from the user/API later)
    subject_profile = SubjectProfile(
        name="Elon Musk",
        entity_type="person",
        dob="1971-06-28",
        locations=["Texas, USA", "California, USA"],
        keywords=["Tesla", "SpaceX", "lawsuit", "neuralink"]
    )

    # 2. Define the initial state for the graph
    initial_state = {
        "subject": subject_profile,
        "documents": [],
        "extracted_entities": [], # Initialize as an empty list
    }

    print("Invoking the investigation workflow...")
    for output in app.stream(initial_state):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            class PydanticEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, (Document, ExtractedEntity, SubjectProfile)):
                         return obj.dict()
                    return json.JSONEncoder.default(self, obj)
            print(json.dumps(value, indent=2, cls=PydanticEncoder))
    
    print("--- Investigation Workflow Finished ---")


if __name__ == "__main__":
    run_investigation()