from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.schema.state import AgentState, PropertyItem
from backend.guardrails.pii import safe_print
from backend.ai_config import model, prompt, prompt_version

import json
import re
from pathlib import Path
# from langchain_google_genai import ChatGoogleGenerativeAI
import re
from typing import Dict, Any, Set
# from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
# from langchain_groq import ChatGroq

from dotenv import load_dotenv, find_dotenv

# تحميل المفاتيح من ملف .env تلقائياً
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

class IntentDetection(BaseModel):
    
    intent: str = Field(
        description=(
            "The detected customer intent. "
            "Must be one of: "
            "new_search, modify_search, alternative_search, "
            "positive_feedback, negative_feedback, "
            "property_inquiry, booking_request, "
            "general_question, unknown."
        )
    )
    
    confidence: float = Field(
        description=(
            "Confidence score between 0 and 1."
        )
    )
    
    reason: str = Field(
        description=(
            "Short explanation for why this intent "
            "was selected."
        )
    )

intent_prompt = prompt("intent")   # text lives in config/ai_config.yaml (prompts.intent)
 
VALID_INTENTS : Set[str] = {
    "new_search",
    "modify_search",
    "alternative_search",
    "positive_feedback",
    "negative_feedback",
    "property_inquiry",
    "booking_request", 
    "general_question",
    "unknown"
}


from backend.agents.structured_llm import structured_llm
llm = ChatGroq(model_name=model("intent"), temperature=0)
# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
# llm = ChatGroq(model_name="openai/gpt-oss-20b", temperature=0)
structured_micro_llm = structured_llm(IntentDetection, model("intent"))   # فيه fallback لو الـ tool call رجع فاضي
# structured_intent_llm = llm.with_structured_output(IntentDetection)

intent_chain = (
    ChatPromptTemplate.from_template(intent_prompt)
    | structured_micro_llm
)


def normalize_arabic_text(text: str) -> str:
    
    text = str(text)
    
    # Remove Tatweel
    text = text.replace("ـ", "")
    
    # Normalize Alef variations
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    
    # Normalize Yeh
    text = text.replace("ى", "ي")
    
    # Remove Arabic diacritics
    text = re.sub(
        r"[\u0617-\u061A\u064B-\u0652]",
        "",
        text
    )
    
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    
    return text.strip()

try:
    from langsmith_ops.tracer import langsmith_trace
except Exception:
    def langsmith_trace(*args, **kwargs):
        return lambda fn: fn

@langsmith_trace(agent_name="intent_detector", prompt_version=prompt_version("intent"), default_model=model("intent"), stage_name="detect_micro_intent")
def detect_intent(conversation: str) -> Dict[str, Any]:
    
    normalized_text = normalize_arabic_text(conversation)
    
    result = intent_chain.invoke({"conversation": normalized_text})
    
    result_dict = result.model_dump()
    
    detected_intent = result_dict.get("intent")
    
    if detected_intent not in VALID_INTENTS:
        detected_intent = "unknown"
        result_dict["intent"] = "unknown"
    
    confidence = float(result_dict.get("confidence", 0.0))
    
    confidence = max(0.0, min(1.0, confidence))
    
    result_dict["confidence"] = confidence
    
    return result_dict



# =====================================================================
# 5. Official LangGraph Node Wrapper
# =====================================================================
def detect_intent_node(state: AgentState) -> Dict[str, Any]:
    """
    LangGraph Node: Reads the current conversation query from AgentState,
    detects the micro-intent, and updates the State.
    """
    if state.get("micro_intent") == "new_search":
        print("⚡ [Micro Intent Node] Fast-pass: 'new_search' already set by Supervisor. Skipping LLM call.")
        return {
            "micro_intent": "new_search",
            "intent_confidence": 1.0
        }
    user_query = state.get("user_query") or ""
    
    if not user_query.strip():
        print("⚠️ [Micro Intent Node] Input query is empty. Defaulting to 'unknown'.")
        return {
            "micro_intent": "unknown",
            "intent_confidence": 0.0,
            "intent_reason": "No query provided"
        }
    
    safe_print(f"\n💬 [Micro Intent Node] Analyzing text: '{user_query}'")   # Privacy: الاسم والرقم بيتخفوا
    intent_res = detect_intent(user_query)

    print(f"🎯 Detected Intent: {intent_res['intent']} (Confidence: {intent_res['confidence']})")
    safe_print(f"💡 Reason: {intent_res['reason']}")
    
    return {
        "micro_intent": intent_res["intent"],
        "intent_confidence": intent_res["confidence"]
    }