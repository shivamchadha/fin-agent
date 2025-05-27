import json
from datetime import datetime
import os

TRANSCRIPT_FILE = "transcripts.json"

def load_transcripts():
    if os.path.exists(TRANSCRIPT_FILE):
        with open(TRANSCRIPT_FILE, "r") as f:
            return json.load(f)
    return []

def save_transcript(entry):
    transcripts = load_transcripts()
    transcripts.append(entry)
    with open(TRANSCRIPT_FILE, "w") as f:
        json.dump(transcripts, f, indent=2)



def log_interaction(user_message, response, retrieved_context,tools_used):
    transcript_entry = {
        "timestamp": datetime.now().isoformat(),
        "retrieved_context": retrieved_context,
        "user_question": user_message,
        "agent_response": response,
        "tools_used": tools_used
    }
    save_transcript(transcript_entry)