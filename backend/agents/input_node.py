"""input_node.py: LangGraph node that turns the customer's input
(recorded call and/or free text) into structured requirements.

Reads from state:
    audio_path : str | None   path to a recorded call
    user_text  : str | None   free text (Arabic or English)

Writes to state:
    conversation_text : str | None   speaker-labeled call transcript
    requirements      : dict | None  CustomerRequirements as a dict
    missing_fields    : list[str]
    error             : str | None
"""
from backend.services.user_input import audio_to_conversation
from backend.services.user_input_text import extract_requirements
from backend.schema.state import AgentState, PropertyItem
from typing import Optional, Literal, List, Dict, Any


def input_node(state: AgentState) -> Dict[str, Any]:
    input_type = state.get("user_input_type", "text")
    audio_path = state.get("audio_path")
    raw_text = (state.get("user_query") or "").strip() or None

    conversation_text = audio_to_conversation(audio_path) if audio_path else None

    if not any([conversation_text, raw_text]):
        return {
            # "conversation_text": None,1
            "customer_reqs": None,
            # "missing_fields": [],
            "error": "Please provide at least one input: a recorded call or text.",
        }

    req = extract_requirements(call_transcript=conversation_text, raw_text=raw_text)

    return {
        # "conversation_text": conversation_text,
        "customer_reqs": req.model_dump(),
        # "missing_fields": req.missing_fields,
        "error_message": None,
    }
