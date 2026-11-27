"""routing_node.py: LangGraph node that decides what happens after the 3D
model step. There is no "classic" / 2D fallback path here - once the GLB
model is ready, the only next stop is furniture placement.

Reads from state:
    model_state : ModelState

Writes to state:
    next_step : "3d_staging" | "retry" | "error" | "furniture"
    error     : str | None
"""

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from backend.schema.state import AgentState, PropertyItem
from typing import List, Optional
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
from backend.guardrails.actions import strip_unconfirmed_claims
from backend.guardrails.grounding import strip_ungrounded_numbers
from backend.agents.structured_llm import structured_llm
from backend.guardrails.pii import safe_print
from backend.ai_config import model, prompt

PROJECT_ROOT = Path(__file__).resolve().parents[2]      # SmartSemsar/ (فيه .env)، مش backend/agents
load_dotenv(PROJECT_ROOT / ".env")

class MacroIntentClassifier(BaseModel):
    macro_intent: str = Field(
        description=(
            "Must be strictly one of: "
            "'search_by_reqs', "
            "'sketch_based_search', "
            "'direct_sketch_staging', "
            "'general_browse', "
            "'request_sketch_upload'"
        )
    )
    reasoning: str = Field(description="سبب اختيار هذه النية باختصار")
    suggested_reply: str = Field(
        description="رد ترحيبي/توجيهي بالعامية المصرية يناسب نية العميل"
    )
    quick_actions: List[str] = Field(
        default_factory=lambda: [
            "🔍 بحث بمواصفات محددة",
            "📐 بحث عن شقة شبه السكيتش",
            "🛋️ فرش وتحويل السكيتش لـ 3D",
            "🏢 تصفح كل الشقق المتاحة"
        ],
        description="الخيارات الأربعة الرئيسية للـ UI"
    )

_structured_orchestrator = None


def get_structured_orchestrator():
    """الموديل بيتعمل أول ما يتطلب مش وقت الـ import، فالملف بيتحمل حتى لو GROQ_API_KEY مش موجود (التستات مثلاً)"""
    global _structured_orchestrator
    if _structured_orchestrator is None:
        _structured_orchestrator = structured_llm(MacroIntentClassifier, model("orchestrator"))   # فيه fallback لو الـ tool call رجع فاضي
    return _structured_orchestrator

ORCHESTRATOR_PROMPT = prompt("orchestrator")   # text lives in config/ai_config.yaml (prompts.orchestrator)

DEFAULT_QUICK_ACTIONS = [
    "🔍 بحث بمواصفات محددة",
    "📐 بحث عن شقة شبه السكيتش",
    "🛋️ فرش وتحويل السكيتش لـ 3D",
    "🏢 تصفح كل الشقق المتاحة"
]

def supervisor_router_node(state: AgentState) -> dict:
    print("\n🧠 [Supervisor Router] Evaluating macro entry point...")
    user_query = state.get("user_query") or ""
    current_state = state.get("current_stage") or ""
    macro_intent = state.get("macro_intent") or ""
    
    if (current_state=="property_results_displayed") and (macro_intent=="general_browse"):
        return {
        "macro_intent": "direct_sketch_staging",
        "micro_intent": "",
        # "agent_response": result.suggested_reply,
    }

    if current_state=="property_results_displayed":
        return {
        "macro_intent": "search_by_reqs",
        "micro_intent": "",
        # "agent_response": result.suggested_reply,
    }

    
        
        
    has_sketch = bool(state.get("uploaded_sketch_path"))
    if not user_query:
        print("📌 Initial load: Fetching default properties catalog.")
        return {
            "macro_intent": "general_browse"
        }
    
    formatted_prompt = ORCHESTRATOR_PROMPT.format(
        query=user_query,
        has_sketch=has_sketch
    )

    result = get_structured_orchestrator().invoke(formatted_prompt)

    # الموديل ميقدرش يحجز ولا يبعت: أي رد بيدعي كده بيتشال (No false claims of actions)
    result.suggested_reply = strip_unconfirmed_claims(result.suggested_reply)
    # Grounded replies: الموديل مبيشوفش الداتابيز، فميكتبش سعر ولا مساحة ولا أوض العميل مقالهاش
    result.suggested_reply = strip_ungrounded_numbers(result.suggested_reply, user_query)

    print(f"🎯 Detected Path: {result.macro_intent}")
    safe_print(f"💡 Reason: {result.reasoning}")
    
    if result.macro_intent == "search_by_reqs":
        return {
        "macro_intent": result.macro_intent,
        "micro_intent": "new_search",
        "agent_response": result.suggested_reply,
        "quick_actions": DEFAULT_QUICK_ACTIONS  # إرجاع الأزرار الأربعة للـ UI دائماً
    }

    return {
        "macro_intent": result.macro_intent,
        "agent_response": result.suggested_reply,
        "quick_actions": DEFAULT_QUICK_ACTIONS  # إرجاع الأزرار الأربعة للـ UI دائماً
    }


def routing_node(state: AgentState) -> dict:
    model_state = state.get("model_state")

    if model_state is None or model_state.status == "pending":
        return {"next_step": "3d_staging"}

    if model_state.status == "building":
        return {"next_step": "retry"}

    if model_state.status == "failed":
        return {"next_step": "error", "error": model_state.error or "3D model build failed."}

    # status == "ready" -> hand off to furniture placement
    return {"next_step": "furniture"}








    # current_stage = state.get("current_stage")
    

    # if state.get("selected_property_id") or state.get("micro_intent") == "select_property":
    #     print("📌 Direct button click: Jumping straight to property selection & 3D staging.")
    #     return {
    #         "macro_intent": "select_property",
    #         "micro_intent": "select_property"
    #     }
    
    

    # if current_stage == "property_results_displayed" and not has_sketch:
    #     print("📌 [Supervisor] User is replying to displayed properties -> Routing directly to Micro Intent.")
    #     return {
    #         "macro_intent": "search_by_reqs"
    #     }
    
    

